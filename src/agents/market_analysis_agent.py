import re
from src.state import AgentState
from src.agents.base import get_llm
from src.tools.market_data import get_live_quote, get_fundamentals, MarketDataError
from config import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are the Market Analysis Agent. You are given live quote
and fundamental data for one or more tickers. Provide a concise market
insight: price action, valuation context (P/E vs sector norms if known),
52-week range position, and any notable signal. Avoid definitive buy/sell
calls — frame observations as informational, not investment advice."""

TICKER_RE = re.compile(r"\b[A-Z]{1,5}\b")
COMMON_WORDS = {
    "I", "A", "THE", "IS", "OF", "FOR", "TO", "IN", "ON", "AND", "HOW", "DOING",
    "WHAT", "WHY", "ARE", "AM", "IT", "MY", "ME", "DO", "AN", "AT", "BE", "BY",
    "OR", "SO", "UP", "US", "GO", "NOT", "CAN", "ALL", "NEW", "NOW", "TODAY",
}


def _extract_tickers(query: str) -> list[str]:
    """
    Only treat ALREADY-UPPERCASE tokens in the original query as candidate
    tickers (e.g. user wrote "AAPL" or "MSFT"). We deliberately do NOT
    upper-case the whole query first — doing so turns ordinary lowercase
    words like "how" or "doing" into false-positive "tickers" once matched
    against the 1-5-letter pattern.
    """
    candidates = TICKER_RE.findall(query)
    return [c for c in candidates if c not in COMMON_WORDS][:5]


def run(state: AgentState) -> AgentState:
    query = state["query"]
    tickers = _extract_tickers(query)
    if not tickers:
        state["final_response"] = (
            "I can pull real-time market data — just include a ticker symbol "
            "(e.g. AAPL, MSFT, TSLA) in your question and I'll analyze it."
        )
        return state

    try:
        data = {}
        errors = {}
        for tk in tickers:
            try:
                quote = get_live_quote(tk)
                fund = get_fundamentals(tk)
                data[tk] = {**quote, **fund}
            except MarketDataError as e:
                errors[tk] = str(e)

        if not data:
            state["final_response"] = (
                f"I couldn't fetch live data for {', '.join(tickers)} right now "
                "(market data provider issue). Please try again shortly."
            )
            return state

        state["market_data"] = data
        prompt_data = "\n\n".join(
            f"{tk}: price=${d['price']}, change={d['change_percent']}, "
            f"52w range=${d.get('52w_low')}-${d.get('52w_high')}, "
            f"P/E={d.get('pe_ratio')}, sector={d.get('sector')}, beta={d.get('beta')}"
            for tk, d in data.items()
        )
        llm = get_llm()
        response = llm.invoke([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Query: {query}\n\nLive data:\n{prompt_data}"},
        ])
        state["final_response"] = response.content
        state["sources"] = [f"{d.get('source', 'market data')} (live)" for d in data.values()]
    except Exception as e:
        logger.error(f"Market analysis agent failed: {e}")
        state["final_response"] = "I hit an issue pulling market data. Please try again."
        state["error"] = str(e)
    return state
