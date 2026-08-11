"""
MCP (Model Context Protocol) server — exposes this project's finance tools
to Claude Desktop (or any MCP-compatible client) as callable tools, and
optionally exposes the full multi-agent assistant as one tool.

Run:
    python -m src.mcp.server

Then add to Claude Desktop's claude_desktop_config.json:
{
  "mcpServers": {
    "finance-assistant": {
      "command": "python",
      "args": ["-m", "src.mcp.server"],
      "cwd": "/absolute/path/to/capstone_finance_agent"
    }
  }
}
"""
import json
import uuid
from mcp.server.fastmcp import FastMCP
from src.tools.market_data import get_live_quote, get_fundamentals, MarketDataError
from src.tools.news_tool import get_news
from src.rag.vector_store import similarity_search
from src.graph import invoke as run_assistant

mcp = FastMCP("finance-assistant")


@mcp.tool()
def get_stock_quote(ticker: str) -> str:
    """Get a live stock quote (price, change, volume) for a given ticker symbol."""
    try:
        return json.dumps(get_live_quote(ticker))
    except MarketDataError as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
def get_stock_fundamentals(ticker: str) -> str:
    """Get company fundamentals (sector, P/E, market cap, 52-week range) for a ticker."""
    try:
        return json.dumps(get_fundamentals(ticker))
    except MarketDataError as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
def get_financial_news(ticker: str = "") -> str:
    """Get recent financial news headlines, optionally filtered to a ticker symbol."""
    items = get_news(ticker=ticker or None, limit=5)
    return json.dumps(items)


@mcp.tool()
def search_finance_knowledge_base(query: str, category: str = "") -> str:
    """
    Search the curated financial-education knowledge base (RAG).
    Categories: finance_qa, tax_education, goal_planning, market_analysis,
    portfolio_analysis, news_synthesizer.
    """
    docs = similarity_search(query, k=4, category=category or None)
    return json.dumps(docs)


@mcp.tool()
def ask_financial_assistant(query: str, session_id: str = "") -> str:
    """
    Ask the full multi-agent financial assistant a question. Routes to the
    best specialist agent (Q&A, portfolio, market, goal planning, news, or
    tax) automatically and returns its response with sources.
    """
    sid = session_id or str(uuid.uuid4())
    result = run_assistant(query=query, user_id="mcp-user", session_id=sid)
    return json.dumps(result)


if __name__ == "__main__":
    mcp.run(transport="stdio")
