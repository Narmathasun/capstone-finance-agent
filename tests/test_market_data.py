from unittest.mock import patch, MagicMock
import pytest
from src.tools.market_data import (
    get_live_quote, get_portfolio_quotes, MarketDataError, _quote_cache
)


@pytest.fixture(autouse=True)
def clear_cache():
    _quote_cache.clear()
    yield
    _quote_cache.clear()


class TestGetLiveQuote:
    @patch("src.tools.market_data.yf.Ticker")
    def test_yfinance_success(self, mock_ticker_cls):
        mock_ticker = MagicMock()
        mock_ticker.fast_info = {
            "last_price": 200.0, "previous_close": 195.0,
            "last_volume": 1000000, "day_high": 202.0, "day_low": 198.0,
        }
        mock_ticker_cls.return_value = mock_ticker

        result = get_live_quote("AAPL")
        assert result["ticker"] == "AAPL"
        assert result["price"] == 200.0
        assert result["source"] == "yfinance"

    @patch("src.tools.market_data._alpha_vantage_quote")
    @patch("src.tools.market_data.yf.Ticker")
    def test_falls_back_to_alpha_vantage(self, mock_ticker_cls, mock_av):
        mock_ticker_cls.side_effect = Exception("yfinance down")
        mock_av.return_value = {
            "ticker": "AAPL", "price": 199.0, "change": 1.0,
            "change_percent": "0.5%", "volume": 900000, "source": "alpha_vantage",
        }
        result = get_live_quote("AAPL")
        assert result["source"] == "alpha_vantage"

    @patch("src.tools.market_data._alpha_vantage_quote")
    @patch("src.tools.market_data.yf.Ticker")
    def test_raises_when_both_sources_fail(self, mock_ticker_cls, mock_av):
        mock_ticker_cls.side_effect = Exception("yfinance down")
        mock_av.side_effect = Exception("alpha vantage down")
        with pytest.raises(MarketDataError):
            get_live_quote("AAPL")

    @patch("src.tools.market_data.yf.Ticker")
    def test_uses_cache_on_second_call(self, mock_ticker_cls):
        mock_ticker = MagicMock()
        mock_ticker.fast_info = {
            "last_price": 200.0, "previous_close": 195.0, "last_volume": 1000000,
        }
        mock_ticker_cls.return_value = mock_ticker

        get_live_quote("AAPL")
        get_live_quote("AAPL")
        assert mock_ticker_cls.call_count == 1  # second call served from cache


class TestGetPortfolioQuotes:
    @patch("src.tools.market_data.get_live_quote")
    def test_partial_failure_does_not_kill_batch(self, mock_quote):
        def side_effect(tk):
            if tk == "BADTICKER":
                raise MarketDataError("not found")
            return {"ticker": tk, "price": 100.0}
        mock_quote.side_effect = side_effect

        result = get_portfolio_quotes(["AAPL", "BADTICKER"])
        assert "AAPL" in result["quotes"]
        assert "BADTICKER" in result["errors"]
