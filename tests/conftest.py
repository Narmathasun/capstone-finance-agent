import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def sample_state():
    return {
        "messages": [],
        "user_id": "test-user",
        "session_id": "test-session",
        "query": "What is diversification?",
        "route": None,
        "route_confidence": None,
        "retrieved_docs": None,
        "market_data": None,
        "portfolio": [],
        "news_items": None,
        "final_response": None,
        "sources": None,
        "error": None,
    }


@pytest.fixture
def sample_portfolio():
    return [
        {"ticker": "AAPL", "shares": 10, "cost_basis": 150.0},
        {"ticker": "MSFT", "shares": 5, "cost_basis": 300.0},
    ]


@pytest.fixture
def mock_quote():
    return {
        "ticker": "AAPL", "price": 200.0, "change": 2.5, "change_percent": "1.27%",
        "volume": 50000000, "day_high": 202.0, "day_low": 198.0, "source": "yfinance",
    }
