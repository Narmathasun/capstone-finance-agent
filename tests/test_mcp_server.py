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

# NOTE: server.py imports its dependencies lazily (inside each tool
# function, not at module top level — see the docstring in
# src/mcp/server.py for why: keeping startup fast for MCP clients that
# enforce a strict startup timeout, e.g. Cowork/Code sessions). Because of
# that, patches must target the ORIGINAL defining module (e.g.
# "src.tools.market_data.get_live_quote"), not "src.mcp.server.get_live_quote"
# — the latter name is never bound at module level anymore, so patching it
# there would silently patch nothing and the real function would still run.


class TestGetStockQuote:
    @patch("src.tools.market_data.get_live_quote")
    def test_returns_json_quote(self, mock_quote):
        mock_quote.return_value = {"ticker": "AAPL", "price": 200.0}
        result = json.loads(get_stock_quote("AAPL"))
        assert result["ticker"] == "AAPL"

    @patch("src.tools.market_data.get_live_quote")
    def test_returns_error_json_on_failure(self, mock_quote):
        mock_quote.side_effect = MarketDataError("no data")
        result = json.loads(get_stock_quote("BADTICKER"))
        assert "error" in result


class TestGetStockFundamentals:
    @patch("src.tools.market_data.get_fundamentals")
    def test_returns_json_fundamentals(self, mock_fund):
        mock_fund.return_value = {"ticker": "AAPL", "sector": "Technology"}
        result = json.loads(get_stock_fundamentals("AAPL"))
        assert result["sector"] == "Technology"

    @patch("src.tools.market_data.get_fundamentals")
    def test_returns_error_json_on_failure(self, mock_fund):
        mock_fund.side_effect = MarketDataError("no data")
        result = json.loads(get_stock_fundamentals("BADTICKER"))
        assert "error" in result


class TestGetFinancialNews:
    @patch("src.tools.news_tool.get_news")
    def test_returns_json_news_list(self, mock_news):
        mock_news.return_value = [{"title": "headline"}]
        result = json.loads(get_financial_news("AAPL"))
        assert len(result) == 1

    @patch("src.tools.news_tool.get_news")
    def test_empty_ticker_uses_general_market(self, mock_news):
        mock_news.return_value = []
        get_financial_news("")
        mock_news.assert_called_once_with(ticker=None, limit=5)


class TestSearchKnowledgeBase:
    @patch("src.rag.vector_store.similarity_search")
    def test_returns_json_docs(self, mock_search):
        mock_search.return_value = [{"content": "text", "source": "kb"}]
        result = json.loads(search_finance_knowledge_base("Roth IRA", "tax_education"))
        assert result[0]["source"] == "kb"


class TestAskFinancialAssistant:
    @patch("src.graph.invoke")
    def test_generates_session_id_when_missing(self, mock_run):
        mock_run.return_value = {"response": "answer", "route": "finance_qa", "sources": []}
        result = json.loads(ask_financial_assistant("What is diversification?"))
        assert result["response"] == "answer"
        # a session_id was generated and passed through
        _, kwargs = mock_run.call_args
        assert kwargs["session_id"]  # non-empty
