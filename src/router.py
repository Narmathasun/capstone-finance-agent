"""
Workflow Router — the first hop in the graph.
Classifies the incoming user query into one of the six agent routes
using a lightweight structured-output LLM call (cheap + fast model).
Falls back to a keyword heuristic if the LLM call fails, so the
system degrades gracefully instead of crashing.
"""
from typing import Literal
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from config import settings, get_logger
from src.state import AgentState

logger = get_logger(__name__)

ROUTES = [
    "finance_qa",
    "portfolio_analysis",
    "market_analysis",
    "goal_planning",
    "news_synthesizer",
    "tax_education",
]

class RouteDecision(BaseModel):
    route: Literal[
        "finance_qa", "portfolio_analysis", "market_analysis",
        "goal_planning", "news_synthesizer", "tax_education"
    ] = Field(description="The single best-fit specialist agent for this query")
    confidence: float = Field(description="0-1 confidence in this routing decision")

_router_llm = ChatOpenAI(
    model=settings.OPENAI_MODEL,
    api_key=settings.OPENAI_API_KEY,
    temperature=0,
).with_structured_output(RouteDecision)

_KEYWORD_FALLBACK = {
    "portfolio_analysis": ["my portfolio", "holdings", "diversif", "allocation", "rebalance"],
    "market_analysis": ["stock price", "quote", "ticker", "market today", "trend", "chart"],
    "goal_planning": ["retire", "goal", "save for", "how much should i save", "financial plan"],
    "news_synthesizer": ["news", "headline", "latest on", "happened to"],
    "tax_education": ["tax", "401k", "ira", "roth", "capital gains", "deduction"],
}


def _keyword_route(query: str) -> str:
    q = query.lower()
    for route, kws in _KEYWORD_FALLBACK.items():
        if any(kw in q for kw in kws):
            return route
    return "finance_qa"  # safe default


def route_query(state: AgentState) -> AgentState:
    query = state["query"]
    try:
        decision = _router_llm.invoke(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a routing classifier for a financial assistant. "
                        "Choose exactly one specialist agent best suited to handle the user's query:\n"
                        "- finance_qa: general financial education / definitions / how-things-work questions\n"
                        "- portfolio_analysis: user wants their own holdings reviewed, diversification/risk assessed\n"
                        "- market_analysis: real-time quotes, price trends, technical/fundamental market data\n"
                        "- goal_planning: retirement, savings goals, budgeting timelines, projections\n"
                        "- news_synthesizer: 'what's happening with X', recent news, headline summaries\n"
                        "- tax_education: tax rules, account types (401k/IRA/Roth), deductions, capital gains tax"
                    ),
                },
                {"role": "user", "content": query},
            ]
        )
        state["route"] = decision.route
        state["route_confidence"] = decision.confidence
        logger.info(f"Routed '{query[:50]}...' -> {decision.route} ({decision.confidence:.2f})")
    except Exception as e:
        logger.warning(f"Router LLM failed ({e}); falling back to keyword routing")
        state["route"] = _keyword_route(query)
        state["route_confidence"] = 0.3
    return state


def route_selector(state: AgentState) -> str:
    """Conditional-edge function LangGraph uses to pick the next node."""
    return state.get("route") or "finance_qa"
