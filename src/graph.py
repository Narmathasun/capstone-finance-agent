"""
Orchestration graph. Wires: entry -> router -> one of six agent nodes -> END.

Checkpointer backend is configurable via CHECKPOINTER_BACKEND in .env:
  - "memory" (default): LangGraph's MemorySaver, in-process only, wiped on
    restart. Appropriate for ephemeral hosts (e.g. Streamlit Community
    Cloud) where local disk writes don't survive a redeploy anyway.
  - "sqlite": persists conversation state to a local .sqlite file, durable
    across restarts on a machine/server you control.

Both are wired through the same build_graph()/get_app()/invoke() interface
so callers (Streamlit UI, MCP server) never need to know which backend is
active.
"""
import os
import sqlite3
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from src.state import AgentState
from src.router import route_query, route_selector, ROUTES
from src.agents import (
    finance_qa_agent,
    portfolio_agent,
    market_analysis_agent,
    goal_planning_agent,
    news_synthesizer_agent,
    tax_education_agent,
)
from config import settings, get_logger

logger = get_logger(__name__)

_AGENT_FN = {
    "finance_qa": finance_qa_agent.run,
    "portfolio_analysis": portfolio_agent.run,
    "market_analysis": market_analysis_agent.run,
    "goal_planning": goal_planning_agent.run,
    "news_synthesizer": news_synthesizer_agent.run,
    "tax_education": tax_education_agent.run,
}

# Anchors relative SQLITE_CHECKPOINT_PATH values to this file's own
# directory (the project root), rather than the process's actual working
# directory. This matters because MCP clients — Claude Desktop in
# particular, on the sandboxed Microsoft Store build — do not reliably
# honor the configured `cwd` when spawning the server subprocess (the
# same root cause behind an earlier `ModuleNotFoundError: No module named
# 'src'` bug, fixed there via an explicit PYTHONPATH). A relative sqlite
# path resolved against the wrong working directory produces
# `sqlite3.OperationalError: unable to open database file` — reproduced
# and confirmed as the exact failure mode this anchor avoids.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _resolve_sqlite_path(configured_path: str) -> str:
    if os.path.isabs(configured_path):
        return configured_path
    return os.path.join(_PROJECT_ROOT, configured_path)


def _build_default_checkpointer():
    """
    Constructs the checkpointer based on CHECKPOINTER_BACKEND. Uses a raw
    sqlite3.Connection passed directly to SqliteSaver's constructor (rather
    than SqliteSaver.from_conn_string(), which is a context manager meant
    for short-lived `with` blocks) since this app is long-running — the
    connection needs to stay open for the process's whole lifetime, not
    close after a single call.

    If the sqlite backend is configured but the file genuinely can't be
    opened for any reason (permissions, a still-wrong resolved path on an
    unusual setup, disk issues), this falls back to MemorySaver rather
    than crashing every single agent call — matching this project's
    established resilience philosophy (see market_data.py's fallback
    chain, every agent's try/except, etc.) rather than introducing a
    single new hard-failure point.
    """
    if settings.CHECKPOINTER_BACKEND == "sqlite":
        resolved_path = _resolve_sqlite_path(settings.SQLITE_CHECKPOINT_PATH)
        try:
            os.makedirs(os.path.dirname(resolved_path) or ".", exist_ok=True)
            from langgraph.checkpoint.sqlite import SqliteSaver
            conn = sqlite3.connect(resolved_path, check_same_thread=False)
            saver = SqliteSaver(conn)
            saver.setup()  # idempotent — creates tables on first run, no-op after
            logger.info(f"Using SqliteSaver checkpointer at {resolved_path}")
            return saver
        except Exception as e:
            logger.error(
                f"Could not open sqlite checkpoint database at {resolved_path} "
                f"({e}); falling back to in-memory checkpointer for this run. "
                "Conversation memory will not persist across restarts until "
                "this is resolved."
            )
            return MemorySaver()

    logger.info("Using in-memory MemorySaver checkpointer (not persisted across restarts)")
    return MemorySaver()


def build_graph(checkpointer=None):
    graph = StateGraph(AgentState)

    graph.add_node("router", route_query)
    for name, fn in _AGENT_FN.items():
        graph.add_node(name, fn)

    graph.set_entry_point("router")
    graph.add_conditional_edges("router", route_selector, {r: r for r in ROUTES})
    for name in _AGENT_FN:
        graph.add_edge(name, END)

    return graph.compile(checkpointer=checkpointer or _build_default_checkpointer())


# Singleton compiled app used by the UI / MCP server
_app = None

def get_app():
    global _app
    if _app is None:
        _app = build_graph()
    return _app


def invoke(query: str, user_id: str, session_id: str, portfolio: list | None = None) -> dict:
    """Single entry point the UI / MCP server call."""
    app = get_app()
    config = {"configurable": {"thread_id": session_id}}
    initial_state = {
        "messages": [{"role": "user", "content": query}],
        "user_id": user_id,
        "session_id": session_id,
        "query": query,
        "portfolio": portfolio or [],
    }
    result = app.invoke(initial_state, config=config)
    return {
        "response": result.get("final_response", "I couldn't generate a response."),
        "route": result.get("route"),
        "sources": result.get("sources", []),
        "error": result.get("error"),
    }