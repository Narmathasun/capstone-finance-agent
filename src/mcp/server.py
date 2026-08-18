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

IMPORTANT — imports are deliberately LAZY (inside each tool function, not
at module top level). Importing the full LangChain/LangGraph/Chroma/
Pinecone stack eagerly adds ~20+ seconds to process startup before FastMCP
is even ready to respond to anything. Claude Desktop's normal chat MCP
connection tolerates that, but features that spin up their own separate
copy of this server (e.g. Cowork/Code sessions) enforce a stricter startup
timeout and will report "Request timed out" / a failed connection if the
process isn't responsive quickly enough. Keeping the top-level imports
minimal (just `json`, `uuid`, and `FastMCP` itself) means the server binds
and starts listening almost immediately; the heavier imports only happen
the first time a specific tool is actually invoked, which is a one-time
cost paid on that tool's first real call rather than blocking startup.
"""
import json
import uuid
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("finance-assistant")


@mcp.tool()
def get_stock_quote(ticker: str) -> str:
    """Get a live stock quote (price, change, volume) for a given ticker symbol."""
    from src.tools.market_data import get_live_quote, MarketDataError
    try:
        return json.dumps(get_live_quote(ticker))
    except MarketDataError as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
def get_stock_fundamentals(ticker: str) -> str:
    """Get company fundamentals (sector, P/E, market cap, 52-week range) for a ticker."""
    from src.tools.market_data import get_fundamentals, MarketDataError
    try:
        return json.dumps(get_fundamentals(ticker))
    except MarketDataError as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
def get_financial_news(ticker: str = "") -> str:
    """Get recent financial news headlines, optionally filtered to a ticker symbol."""
    from src.tools.news_tool import get_news
    items = get_news(ticker=ticker or None, limit=5)
    return json.dumps(items)


@mcp.tool()
def search_finance_knowledge_base(query: str, category: str = "") -> str:
    """
    Search the curated financial-education knowledge base (RAG).
    Categories: finance_qa, tax_education, goal_planning, market_analysis,
    portfolio_analysis, news_synthesizer.
    """
    from src.rag.vector_store import similarity_search
    docs = similarity_search(query, k=4, category=category or None)
    return json.dumps(docs)


@mcp.tool()
def ask_financial_assistant(query: str, session_id: str = "") -> str:
    """
    Ask the full multi-agent financial assistant a question. Routes to the
    best specialist agent (Q&A, portfolio, market, goal planning, news, or
    tax) automatically and returns its response with sources.
    """
    from src.graph import invoke as run_assistant
    sid = session_id or str(uuid.uuid4())
    result = run_assistant(query=query, user_id="mcp-user", session_id=sid)
    return json.dumps(result)


if __name__ == "__main__":
    mcp.run(transport="stdio")
