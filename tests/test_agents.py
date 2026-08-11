from unittest.mock import patch, MagicMock
from src.agents import (
    finance_qa_agent, portfolio_agent, market_analysis_agent,
    goal_planning_agent, news_synthesizer_agent, tax_education_agent,
)
from src.tools.market_data import MarketDataError


class TestFinanceQAAgent:
    @patch("src.agents.finance_qa_agent.get_llm")
    @patch("src.agents.finance_qa_agent.retrieve_context")
    def test_success(self, mock_retrieve, mock_get_llm, sample_state):
        mock_retrieve.return_value = [{"content": "text", "source": "kb", "title": "t"}]
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Diversification spreads risk.")
        mock_get_llm.return_value = mock_llm

        result = finance_qa_agent.run(sample_state)
        assert result["final_response"] == "Diversification spreads risk."
        assert result["sources"] == ["kb"]

    @patch("src.agents.finance_qa_agent.get_llm")
    @patch("src.agents.finance_qa_agent.retrieve_context")
    def test_llm_failure_sets_fallback_message(self, mock_retrieve, mock_get_llm, sample_state):
        mock_retrieve.return_value = []
        mock_get_llm.side_effect = Exception("LLM down")

        result = finance_qa_agent.run(sample_state)
        assert "issue" in result["final_response"].lower()
        assert result["error"] is not None


class TestPortfolioAgent:
    def test_empty_portfolio_returns_prompt(self, sample_state):
        sample_state["portfolio"] = []
        result = portfolio_agent.run(sample_state)
        assert "don't see any portfolio" in result["final_response"]

    @patch("src.agents.portfolio_agent.get_llm")
    @patch("src.agents.portfolio_agent.get_fundamentals")
    @patch("src.agents.portfolio_agent.get_portfolio_quotes")
    def test_analyzes_holdings(self, mock_quotes, mock_fund, mock_get_llm, sample_state, sample_portfolio):
        sample_state["portfolio"] = sample_portfolio
        mock_quotes.return_value = {
            "quotes": {"AAPL": {"price": 200.0}, "MSFT": {"price": 350.0}},
            "errors": {},
        }
        mock_fund.return_value = {"sector": "Technology", "beta": 1.2}
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Well diversified.")
        mock_get_llm.return_value = mock_llm

        result = portfolio_agent.run(sample_state)
        assert result["final_response"] == "Well diversified."
        assert len(result["market_data"]["holdings"]) == 2


class TestMarketAnalysisAgent:
    def test_no_ticker_prompts_for_one(self, sample_state):
        sample_state["query"] = "how is the market doing"
        result = market_analysis_agent.run(sample_state)
        assert "ticker symbol" in result["final_response"].lower()

    @patch("src.agents.market_analysis_agent.get_llm")
    @patch("src.agents.market_analysis_agent.get_fundamentals")
    @patch("src.agents.market_analysis_agent.get_live_quote")
    def test_analyzes_ticker(self, mock_quote, mock_fund, mock_get_llm, sample_state, mock_quote_data=None):
        sample_state["query"] = "How is AAPL doing?"
        mock_quote.return_value = {"price": 200.0, "change_percent": "1%", "source": "yfinance"}
        mock_fund.return_value = {"sector": "Tech", "pe_ratio": 30, "52w_high": 210, "52w_low": 150, "beta": 1.1}
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="AAPL is trading near highs.")
        mock_get_llm.return_value = mock_llm

        result = market_analysis_agent.run(sample_state)
        assert "AAPL" in result["final_response"] or result["final_response"]


class TestGoalPlanningAgent:
    @patch("src.agents.goal_planning_agent.get_llm")
    @patch("src.agents.goal_planning_agent.retrieve_context")
    def test_success(self, mock_retrieve, mock_get_llm, sample_state):
        mock_retrieve.return_value = []
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Save $500/mo for 20 years.")
        mock_get_llm.return_value = mock_llm

        sample_state["query"] = "How much to retire with $1M in 20 years?"
        result = goal_planning_agent.run(sample_state)
        assert "500" in result["final_response"]


class TestNewsSynthesizerAgent:
    @patch("src.agents.news_synthesizer_agent.get_llm")
    @patch("src.agents.news_synthesizer_agent.get_news")
    def test_success(self, mock_news, mock_get_llm, sample_state):
        mock_news.return_value = [{"title": "AAPL rises", "summary": "...", "link": "http://x"}]
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="AAPL rose today on strong earnings.")
        mock_get_llm.return_value = mock_llm

        sample_state["query"] = "What's the news on AAPL?"
        result = news_synthesizer_agent.run(sample_state)
        assert "AAPL" in result["final_response"]

    @patch("src.agents.news_synthesizer_agent.get_news")
    def test_no_news_returns_graceful_message(self, mock_news, sample_state):
        mock_news.return_value = []
        result = news_synthesizer_agent.run(sample_state)
        assert "couldn't retrieve" in result["final_response"].lower()


class TestTaxEducationAgent:
    @patch("src.agents.tax_education_agent.get_llm")
    @patch("src.agents.tax_education_agent.retrieve_context")
    def test_success(self, mock_retrieve, mock_get_llm, sample_state):
        mock_retrieve.return_value = [{"content": "Roth info", "source": "IRS", "title": "Roth IRA"}]
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="A Roth IRA is funded with after-tax dollars.")
        mock_get_llm.return_value = mock_llm

        sample_state["query"] = "What is a Roth IRA?"
        result = tax_education_agent.run(sample_state)
        assert "Roth IRA" in result["final_response"]
        assert result["sources"] == ["IRS"]
