"""
Lightweight financial news fetcher.
Primary source: Yahoo Finance RSS (no key required) via feedparser.
Keeps the project dependency-light; swap in NewsAPI/Benzinga easily
by adding a branch here — the rest of the app is source-agnostic.
"""
import feedparser
from cachetools import TTLCache
from config import settings, get_logger

logger = get_logger(__name__)
_news_cache = TTLCache(maxsize=100, ttl=settings.CACHE_TTL_SECONDS)

YF_RSS_TEMPLATE = "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
GENERAL_MARKET_RSS = "https://feeds.finance.yahoo.com/rss/2.0/headline?s=%5EGSPC&region=US&lang=en-US"


def get_news(ticker: str | None = None, limit: int = 5) -> list[dict]:
    cache_key = f"news:{ticker or 'market'}"
    if cache_key in _news_cache:
        return _news_cache[cache_key]

    url = YF_RSS_TEMPLATE.format(ticker=ticker) if ticker else GENERAL_MARKET_RSS
    try:
        feed = feedparser.parse(url)
        items = [
            {
                "title": e.get("title"),
                "link": e.get("link"),
                "published": e.get("published"),
                "summary": e.get("summary", "")[:400],
                "source": "Yahoo Finance RSS",
            }
            for e in feed.entries[:limit]
        ]
        _news_cache[cache_key] = items
        return items
    except Exception as e:
        logger.error(f"News fetch failed ({ticker}): {e}")
        return []
