from src.state import AgentState
from src.agents.base import get_llm
from src.tools.market_data import get_portfolio_quotes, get_fundamentals, MarketDataError
from config import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are the Portfolio Analysis Agent. You are given a user's
current holdings along with live quotes and fundamentals. Analyze:
1) current allocation / concentration risk, 2) sector diversification,
3) unrealized gain/loss per position, 4) notable risk flags (e.g. one stock
>25% of portfolio, high-beta concentration). Be specific and reference the
actual numbers given. This is educational analysis, not personalized
investment advice — say so briefly at the end. Keep it well organized with
headers or bullets."""


def run(state: AgentState) -> AgentState:
    portfolio = state.get("portfolio") or []
    if not portfolio:
        state["final_response"] = (
            "I don't see any portfolio holdings yet. Add your positions (ticker, shares, "
            "cost basis) in the Portfolio tab and I'll analyze allocation, risk, and "
            "performance for you."
        )
        return state

    tickers = [h["ticker"] for h in portfolio]
    try:
        quote_data = get_portfolio_quotes(tickers)
        quotes, errors = quote_data["quotes"], quote_data["errors"]

        enriched = []
        for h in portfolio:
            tk = h["ticker"]
            q = quotes.get(tk)
            if not q:
                continue
            try:
                fund = get_fundamentals(tk)
            except MarketDataError:
                fund = {}
            market_value = q["price"] * h["shares"]
            cost_value = h["cost_basis"] * h["shares"]
            enriched.append({
                "ticker": tk, "shares": h["shares"], "price": q["price"],
                "market_value": round(market_value, 2),
                "gain_loss": round(market_value - cost_value, 2),
                "gain_loss_pct": round((market_value - cost_value) / cost_value * 100, 2) if cost_value else 0,
                "sector": fund.get("sector", "Unknown"),
                "beta": fund.get("beta"),
            })

        state["market_data"] = {"holdings": enriched, "errors": errors}

        summary_lines = [
            f"- {e['ticker']}: {e['shares']} shares @ ${e['price']} = ${e['market_value']:,} "
            f"(P/L {e['gain_loss_pct']}%), sector={e['sector']}, beta={e['beta']}"
            for e in enriched
        ]
        total_value = sum(e["market_value"] for e in enriched)
        prompt_data = "\n".join(summary_lines) + f"\n\nTotal portfolio value: ${total_value:,.2f}"
        if errors:
            prompt_data += f"\n\nNote: could not fetch live data for: {list(errors.keys())}"

        llm = get_llm()
        response = llm.invoke([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt_data},
        ])
        state["final_response"] = response.content
        state["sources"] = ["yfinance / Alpha Vantage live data"]
    except Exception as e:
        logger.error(f"Portfolio agent failed: {e}")
        state["final_response"] = (
            "I couldn't complete the portfolio analysis due to a data issue. "
            "Please try again shortly."
        )
        state["error"] = str(e)
    return state
