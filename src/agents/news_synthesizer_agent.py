import re
from src.state import AgentState
from src.agents.base import get_llm
from src.tools.news_tool import get_news
from config import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are the News Synthesizer Agent. You are given recent
headlines/summaries for a topic or ticker. Synthesize them into a short,
neutral briefing: what happened, why it matters, and any consensus or
disagreement across sources. Do not editorialize or predict price direction.
Always note this is a summary of recent headlines, not investment advice,
and that the user should read full articles for details."""

TICKER_RE = re.compile(r"\b[A-Z]{1,5}\b")
COMMON_WORDS = {
    "I", "A", "THE", "IS", "OF", "FOR", "TO", "IN", "ON", "NEWS", "WHAT",
    "LATEST", "HAPPENED", "ON", "ABOUT", "WITH", "TODAY",
}


def _extract_ticker(query: str) -> str | None:
    """Only match already-uppercase tokens in the original query (see the
    same rationale in market_analysis_agent._extract_tickers)."""
    for c in TICKER_RE.findall(query):
        if c not in COMMON_WORDS:
            return c
    return None


def run(state: AgentState) -> AgentState:
    query = state["query"]
    ticker = _extract_ticker(query)
    try:
        items = get_news(ticker=ticker, limit=6)
        if not items:
            state["final_response"] = (
                "I couldn't retrieve news right now (feed unavailable). "
                "Please try again shortly, or check a financial news site directly."
            )
            return state

        state["news_items"] = items
        headlines_block = "\n".join(f"- {n['title']}: {n['summary']}" for n in items)
        llm = get_llm()
        response = llm.invoke([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Topic: {ticker or 'general market'}\n\nHeadlines:\n{headlines_block}"},
        ])
        state["final_response"] = response.content
        state["sources"] = [n["link"] for n in items]
    except Exception as e:
        logger.error(f"News synthesizer agent failed: {e}")
        state["final_response"] = "I couldn't synthesize news right now. Please try again."
        state["error"] = str(e)
    return state
