"""
Orchestration graph. Wires: entry -> router -> one of six agent nodes -> END.
Uses LangGraph's MemorySaver checkpointer for conversation memory across
turns, keyed by thread_id (== our session_id). Swap MemorySaver for
SqliteSaver/PostgresSaver in production for durable, multi-instance state.
"""
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

_AGENT_FN = {
    "finance_qa": finance_qa_agent.run,
    "portfolio_analysis": portfolio_agent.run,
    "market_analysis": market_analysis_agent.run,
    "goal_planning": goal_planning_agent.run,
    "news_synthesizer": news_synthesizer_agent.run,
    "tax_education": tax_education_agent.run,
}


def build_graph(checkpointer=None):
    graph = StateGraph(AgentState)

    graph.add_node("router", route_query)
    for name, fn in _AGENT_FN.items():
        graph.add_node(name, fn)

    graph.set_entry_point("router")
    graph.add_conditional_edges("router", route_selector, {r: r for r in ROUTES})
    for name in _AGENT_FN:
        graph.add_edge(name, END)

    return graph.compile(checkpointer=checkpointer or MemorySaver())


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
