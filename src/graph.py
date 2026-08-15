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


def _build_default_checkpointer():
    """
    Constructs the checkpointer based on CHECKPOINTER_BACKEND. Uses a raw
    sqlite3.Connection passed directly to SqliteSaver's constructor (rather
    than SqliteSaver.from_conn_string(), which is a context manager meant
    for short-lived `with` blocks) since this app is long-running — the
    connection needs to stay open for the process's whole lifetime, not
    close after a single call.
    """
    if settings.CHECKPOINTER_BACKEND == "sqlite":
        from langgraph.checkpoint.sqlite import SqliteSaver
        conn = sqlite3.connect(settings.SQLITE_CHECKPOINT_PATH, check_same_thread=False)
        saver = SqliteSaver(conn)
        saver.setup()  # idempotent — creates tables on first run, no-op after
        logger.info(f"Using SqliteSaver checkpointer at {settings.SQLITE_CHECKPOINT_PATH}")
        return saver

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
