"""
Shared graph state. Every node (agent) reads/writes this TypedDict.
LangGraph persists this via a checkpointer, which is how we get
multi-turn conversation memory and per-user sessions "for free".
"""
from typing import TypedDict, List, Dict, Any, Optional, Annotated
from langgraph.graph.message import add_messages


class PortfolioHolding(TypedDict):
    ticker: str
    shares: float
    cost_basis: float


class AgentState(TypedDict):
    # conversation memory — add_messages appends instead of overwriting
    messages: Annotated[List[Dict[str, Any]], add_messages]

    # session / identity
    user_id: str
    session_id: str

    # routing
    query: str
    route: Optional[str]           # which agent(s) the router chose
    route_confidence: Optional[float]

    # working data agents fill in
    retrieved_docs: Optional[List[Dict[str, Any]]]   # RAG hits
    market_data: Optional[Dict[str, Any]]            # live quotes / fundamentals
    portfolio: Optional[List[PortfolioHolding]]
    news_items: Optional[List[Dict[str, Any]]]

    # output
    final_response: Optional[str]
    sources: Optional[List[str]]
    error: Optional[str]
