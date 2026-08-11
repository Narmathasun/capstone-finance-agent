import json
from unittest.mock import patch
from src.mcp.server import (
    get_stock_quote,
    get_stock_fundamentals,
    get_financial_news,
    search_finance_knowledge_base,
    ask_financial_assistant,
)
from src.tools.market_data import MarketDataError


class TestGetStockQuote:
    @patch("src.mcp.server.get_live_quote")
    def test_returns_json_quote(self, mock_quote):
        mock_quote.return_value = {"ticker": "AAPL", "price": 200.0}
        result = json.loads(get_stock_quote("AAPL"))
        assert result["ticker"] == "AAPL"

    @patch("src.mcp.server.get_live_quote")
    def test_returns_error_json_on_failure(self, mock_quote):
        mock_quote.side_effect = MarketDataError("no data")
        result = json.loads(get_stock_quote("BADTICKER"))
        assert "error" in result


class TestGetStockFundamentals:
    @patch("src.mcp.server.get_fundamentals")
    def test_returns_json_fundamentals(self, mock_fund):
        mock_fund.return_value = {"ticker": "AAPL", "sector": "Technology"}
        result = json.loads(get_stock_fundamentals("AAPL"))
        assert result["sector"] == "Technology"

    @patch("src.mcp.server.get_fundamentals")
    def test_returns_error_json_on_failure(self, mock_fund):
        mock_fund.side_effect = MarketDataError("no data")
        result = json.loads(get_stock_fundamentals("BADTICKER"))
        assert "error" in result


class TestGetFinancialNews:
    @patch("src.mcp.server.get_news")
    def test_returns_json_news_list(self, mock_news):
        mock_news.return_value = [{"title": "headline"}]
        result = json.loads(get_financial_news("AAPL"))
        assert len(result) == 1

    @patch("src.mcp.server.get_news")
    def test_empty_ticker_uses_general_market(self, mock_news):
        mock_news.return_value = []
        get_financial_news("")
        mock_news.assert_called_once_with(ticker=None, limit=5)


class TestSearchKnowledgeBase:
    @patch("src.mcp.server.similarity_search")
    def test_returns_json_docs(self, mock_search):
        mock_search.return_value = [{"content": "text", "source": "kb"}]
        result = json.loads(search_finance_knowledge_base("Roth IRA", "tax_education"))
        assert result[0]["source"] == "kb"


class TestAskFinancialAssistant:
    @patch("src.mcp.server.run_assistant")
    def test_generates_session_id_when_missing(self, mock_run):
        mock_run.return_value = {"response": "answer", "route": "finance_qa", "sources": []}
        result = json.loads(ask_financial_assistant("What is diversification?"))
        assert result["response"] == "answer"
        # a session_id was generated and passed through
        _, kwargs = mock_run.call_args
        assert kwargs["session_id"]  # non-empty
