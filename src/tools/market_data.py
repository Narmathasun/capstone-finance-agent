"""
Real-time market data layer.
- yfinance is the primary source (no key, generous limits, good for OHLCV/fundamentals)
- Alpha Vantage is used for data yfinance doesn't cover well (e.g. some intraday /
  earnings calendars) and as a secondary source
- cachetools TTLCache avoids hammering APIs and smooths over rate limits
- tenacity retries transient failures with exponential backoff
"""
import time
import yfinance as yf
import requests
from cachetools import TTLCache
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from config import settings, get_logger

logger = get_logger(__name__)

_quote_cache = TTLCache(maxsize=500, ttl=settings.CACHE_TTL_SECONDS)
_fundamentals_cache = TTLCache(maxsize=200, ttl=settings.CACHE_TTL_SECONDS * 4)

ALPHA_VANTAGE_BASE = "https://www.alphavantage.co/query"


class MarketDataError(Exception):
    pass


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception_type((requests.RequestException,)),
    reraise=True,
)
def _alpha_vantage_quote(ticker: str) -> dict:
    params = {
        "function": "GLOBAL_QUOTE",
        "symbol": ticker,
        "apikey": settings.ALPHA_VANTAGE_API_KEY,
    }
    resp = requests.get(ALPHA_VANTAGE_BASE, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    if "Note" in data or "Information" in data:
        # Alpha Vantage rate-limit message comes back as HTTP 200 with a "Note"
        raise MarketDataError("Alpha Vantage rate limit hit")
    quote = data.get("Global Quote", {})
    if not quote:
        raise MarketDataError(f"No Alpha Vantage data for {ticker}")
    return {
        "ticker": ticker,
        "price": float(quote.get("05. price", 0)),
        "change": float(quote.get("09. change", 0)),
        "change_percent": quote.get("10. change percent", "0%"),
        "volume": int(quote.get("06. volume", 0)),
        "source": "alpha_vantage",
    }


def get_live_quote(ticker: str) -> dict:
    """
    Returns a normalized live quote dict.
    Tries yfinance first (fast, no rate limit issues in practice),
    falls back to Alpha Vantage, and finally raises a clean error
    the calling agent can turn into a user-friendly message.
    """
    ticker = ticker.upper().strip()
    cache_key = f"quote:{ticker}"
    if cache_key in _quote_cache:
        return _quote_cache[cache_key]

    try:
        t = yf.Ticker(ticker)
        fast = t.fast_info
        price = fast.get("last_price") or fast.get("lastPrice")
        prev_close = fast.get("previous_close") or fast.get("previousClose")
        if price is None:
            raise MarketDataError("yfinance returned no price")
        change = (price - prev_close) if prev_close else 0
        change_pct = (change / prev_close * 100) if prev_close else 0
        result = {
            "ticker": ticker,
            "price": round(float(price), 2),
            "change": round(float(change), 2),
            "change_percent": f"{change_pct:.2f}%",
            "volume": int(fast.get("last_volume", 0) or 0),
            "day_high": fast.get("day_high"),
            "day_low": fast.get("day_low"),
            "source": "yfinance",
            "timestamp": time.time(),
        }
        _quote_cache[cache_key] = result
        return result
    except Exception as e:
        logger.warning(f"yfinance failed for {ticker} ({e}); trying Alpha Vantage")
        try:
            result = _alpha_vantage_quote(ticker)
            result["timestamp"] = time.time()
            _quote_cache[cache_key] = result
            return result
        except Exception as e2:
            logger.error(f"All market data sources failed for {ticker}: {e2}")
            raise MarketDataError(
                f"Unable to fetch live data for {ticker} right now. "
                "Both yfinance and Alpha Vantage are unavailable — please try again shortly."
            )


def get_fundamentals(ticker: str) -> dict:
    """Company fundamentals for portfolio/market analysis agents."""
    ticker = ticker.upper().strip()
    cache_key = f"fund:{ticker}"
    if cache_key in _fundamentals_cache:
        return _fundamentals_cache[cache_key]
    try:
        t = yf.Ticker(ticker)
        info = t.info
        result = {
            "ticker": ticker,
            "name": info.get("shortName"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "market_cap": info.get("marketCap"),
            "pe_ratio": info.get("trailingPE"),
            "dividend_yield": info.get("dividendYield"),
            "52w_high": info.get("fiftyTwoWeekHigh"),
            "52w_low": info.get("fiftyTwoWeekLow"),
            "beta": info.get("beta"),
        }
        _fundamentals_cache[cache_key] = result
        return result
    except Exception as e:
        logger.error(f"Fundamentals fetch failed for {ticker}: {e}")
        raise MarketDataError(f"Unable to fetch fundamentals for {ticker}.")


def get_price_history(ticker: str, period: str = "6mo", interval: str = "1d"):
    """Returns a pandas DataFrame for charting. Used by the Streamlit dashboard."""
    try:
        t = yf.Ticker(ticker.upper().strip())
        hist = t.history(period=period, interval=interval)
        if hist.empty:
            raise MarketDataError(f"No history for {ticker}")
        return hist
    except Exception as e:
        logger.error(f"History fetch failed for {ticker}: {e}")
        raise MarketDataError(f"Unable to fetch price history for {ticker}.")


def get_portfolio_quotes(tickers: list[str]) -> dict:
    """Batch fetch — used by the Portfolio Analysis Agent. Partial failures don't kill the batch."""
    results, errors = {}, {}
    for tk in tickers:
        try:
            results[tk] = get_live_quote(tk)
        except MarketDataError as e:
            errors[tk] = str(e)
    return {"quotes": results, "errors": errors}
