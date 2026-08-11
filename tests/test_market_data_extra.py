import pandas as pd
from unittest.mock import patch, MagicMock
import pytest
from src.tools.market_data import (
    get_fundamentals, get_price_history, _alpha_vantage_quote,
    MarketDataError, _fundamentals_cache,
)


@pytest.fixture(autouse=True)
def clear_cache():
    _fundamentals_cache.clear()
    yield
    _fundamentals_cache.clear()


class TestAlphaVantageQuote:
    @patch("src.tools.market_data.requests.get")
    def test_parses_valid_response(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "Global Quote": {
                "05. price": "200.50", "09. change": "1.25",
                "10. change percent": "0.63%", "06. volume": "1000000",
            }
        }
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        result = _alpha_vantage_quote("AAPL")
        assert result["price"] == 200.50
        assert result["source"] == "alpha_vantage"

    @patch("src.tools.market_data.requests.get")
    def test_raises_on_rate_limit_note(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"Note": "rate limit exceeded"}
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        with pytest.raises(MarketDataError):
            _alpha_vantage_quote("AAPL")

    @patch("src.tools.market_data.requests.get")
    def test_raises_on_empty_quote(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"Global Quote": {}}
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        with pytest.raises(MarketDataError):
            _alpha_vantage_quote("BADTICKER")


class TestGetFundamentals:
    @patch("src.tools.market_data.yf.Ticker")
    def test_returns_fundamentals(self, mock_ticker_cls):
        mock_ticker = MagicMock()
        mock_ticker.info = {
            "shortName": "Apple Inc.", "sector": "Technology", "industry": "Consumer Electronics",
            "marketCap": 3000000000000, "trailingPE": 30.5, "dividendYield": 0.005,
            "fiftyTwoWeekHigh": 220.0, "fiftyTwoWeekLow": 150.0, "beta": 1.2,
        }
        mock_ticker_cls.return_value = mock_ticker

        result = get_fundamentals("AAPL")
        assert result["name"] == "Apple Inc."
        assert result["sector"] == "Technology"

    @patch("src.tools.market_data.yf.Ticker")
    def test_raises_market_data_error_on_failure(self, mock_ticker_cls):
        mock_ticker_cls.side_effect = Exception("network error")
        with pytest.raises(MarketDataError):
            get_fundamentals("AAPL")

    @patch("src.tools.market_data.yf.Ticker")
    def test_uses_cache_on_second_call(self, mock_ticker_cls):
        mock_ticker = MagicMock()
        mock_ticker.info = {"shortName": "Apple Inc.", "sector": "Technology"}
        mock_ticker_cls.return_value = mock_ticker

        get_fundamentals("AAPL")
        get_fundamentals("AAPL")
        assert mock_ticker_cls.call_count == 1


class TestGetPriceHistory:
    @patch("src.tools.market_data.yf.Ticker")
    def test_returns_dataframe(self, mock_ticker_cls):
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame({
            "Open": [100, 101], "High": [102, 103], "Low": [99, 100], "Close": [101, 102],
        })
        mock_ticker_cls.return_value = mock_ticker

        hist = get_price_history("AAPL", period="1mo")
        assert not hist.empty
        assert "Close" in hist.columns

    @patch("src.tools.market_data.yf.Ticker")
    def test_raises_on_empty_history(self, mock_ticker_cls):
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame()
        mock_ticker_cls.return_value = mock_ticker

        with pytest.raises(MarketDataError):
            get_price_history("BADTICKER")

    @patch("src.tools.market_data.yf.Ticker")
    def test_raises_on_exception(self, mock_ticker_cls):
        mock_ticker_cls.side_effect = Exception("network down")
        with pytest.raises(MarketDataError):
            get_price_history("AAPL")
