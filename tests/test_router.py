from unittest.mock import patch, MagicMock
from src.router import route_query, route_selector, _keyword_route


class TestKeywordFallback:
    def test_portfolio_keywords(self):
        assert _keyword_route("How is my portfolio diversified?") == "portfolio_analysis"

    def test_market_keywords(self):
        assert _keyword_route("What's the stock price of AAPL?") == "market_analysis"

    def test_tax_keywords(self):
        assert _keyword_route("How does a Roth IRA work for taxes?") == "tax_education"

    def test_goal_keywords(self):
        assert _keyword_route("How much should I save for retirement?") == "goal_planning"

    def test_news_keywords(self):
        assert _keyword_route("What's the latest news on Tesla?") == "news_synthesizer"

    def test_default_fallback(self):
        assert _keyword_route("random unrelated query xyz") == "finance_qa"


class TestRouteQuery:
    @patch("src.router._router_llm")
    def test_llm_routing_success(self, mock_llm, sample_state):
        mock_decision = MagicMock(route="tax_education", confidence=0.92)
        mock_llm.invoke.return_value = mock_decision
        sample_state["query"] = "Explain 401k contribution limits"

        result = route_query(sample_state)
        assert result["route"] == "tax_education"
        assert result["route_confidence"] == 0.92

    @patch("src.router._router_llm")
    def test_llm_failure_falls_back_to_keywords(self, mock_llm, sample_state):
        mock_llm.invoke.side_effect = Exception("API down")
        sample_state["query"] = "How is my portfolio doing?"

        result = route_query(sample_state)
        assert result["route"] == "portfolio_analysis"
        assert result["route_confidence"] == 0.3


class TestRouteSelector:
    def test_returns_route_from_state(self, sample_state):
        sample_state["route"] = "market_analysis"
        assert route_selector(sample_state) == "market_analysis"

    def test_defaults_when_missing(self, sample_state):
        sample_state["route"] = None
        assert route_selector(sample_state) == "finance_qa"
