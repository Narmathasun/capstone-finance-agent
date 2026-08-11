from unittest.mock import patch, MagicMock
from src.tools.news_tool import get_news, _news_cache


class TestGetNews:
    def setup_method(self):
        _news_cache.clear()

    @patch("src.tools.news_tool.feedparser.parse")
    def test_fetches_ticker_news(self, mock_parse):
        mock_entry = MagicMock()
        mock_entry.get.side_effect = lambda k, d=None: {
            "title": "AAPL rises on earnings",
            "link": "http://example.com/1",
            "published": "2026-08-10",
            "summary": "Apple reported strong Q3 results.",
        }.get(k, d)
        mock_feed = MagicMock()
        mock_feed.entries = [mock_entry]
        mock_parse.return_value = mock_feed

        items = get_news(ticker="AAPL", limit=5)
        assert len(items) == 1
        assert items[0]["title"] == "AAPL rises on earnings"
        assert items[0]["source"] == "Yahoo Finance RSS"

    @patch("src.tools.news_tool.feedparser.parse")
    def test_returns_empty_list_on_failure(self, mock_parse):
        mock_parse.side_effect = Exception("feed unavailable")
        items = get_news(ticker="AAPL")
        assert items == []

    @patch("src.tools.news_tool.feedparser.parse")
    def test_uses_general_market_feed_without_ticker(self, mock_parse):
        mock_feed = MagicMock()
        mock_feed.entries = []
        mock_parse.return_value = mock_feed

        get_news(ticker=None)
        called_url = mock_parse.call_args[0][0]
        assert "%5EGSPC" in called_url  # S&P 500 general market feed

    @patch("src.tools.news_tool.feedparser.parse")
    def test_uses_cache_on_second_call(self, mock_parse):
        mock_feed = MagicMock()
        mock_feed.entries = []
        mock_parse.return_value = mock_feed

        get_news(ticker="MSFT")
        get_news(ticker="MSFT")
        assert mock_parse.call_count == 1
