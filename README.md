# Multi-Agent Financial Assistant — Capstone Build Guide

This is a complete, runnable scaffold for your capstone. Every file referenced
below already exists in this project — your job is to fill in secrets, expand
the knowledge base, run it, test it, and layer on polish before submission.

**Stack:** LangGraph (orchestration) · LangChain · OpenAI GPT · Chroma/Pinecone
(vector DB) · yfinance + Alpha Vantage (market data) · Streamlit (UI) · MCP
(Claude Desktop integration) · pytest (testing)

---

## 0. How the pieces fit together

```
User Query (Streamlit chat / MCP tool call)
      │
      ▼
 ┌─────────────┐
 │   Router    │  LLM classifies query → 1 of 6 routes (keyword fallback if LLM fails)
 └──────┬──────┘
        │  (LangGraph conditional edge)
        ▼
 ┌──────────────────────────────────────────────────────────────┐
 │  finance_qa | portfolio_analysis | market_analysis |          │
 │  goal_planning | news_synthesizer | tax_education             │  ← 6 agent nodes
 └──────┬────────────────────┬────────────────────┬─────────────┘
        │ RAG retrieval       │ live market data    │ news feed
        ▼                     ▼                    ▼
   Chroma/Pinecone       yfinance/AlphaVantage   Yahoo RSS
        │                     │                    │
        └──────────► LLM Processing (GPT) ◄─────────┘
                          │
                          ▼
                  final_response + sources
                          │
                          ▼
              Streamlit UI  /  MCP tool result
```

State (conversation history, session, portfolio, retrieved docs, route) flows
through a single `AgentState` TypedDict (`src/state.py`) that LangGraph
persists via a checkpointer — this is what gives you multi-turn memory and
per-user sessions with almost no extra code.

**Project layout:**
```
capstone_finance_agent/
├── app.py                    # Streamlit UI (chat, portfolio, market tabs)
├── config.py                 # env/settings loader
├── requirements.txt
├── .env.example              # copy to .env and fill in keys
├── pytest.ini
├── src/
│   ├── state.py              # shared LangGraph state schema
│   ├── router.py             # workflow router (LLM + keyword fallback)
│   ├── graph.py               # LangGraph orchestration graph
│   ├── agents/                # the 6 specialized agents
│   │   ├── base.py            # shared RAG/LLM helpers
│   │   ├── finance_qa_agent.py
│   │   ├── portfolio_agent.py
│   │   ├── market_analysis_agent.py
│   │   ├── goal_planning_agent.py
│   │   ├── news_synthesizer_agent.py
│   │   └── tax_education_agent.py
│   ├── tools/
│   │   ├── market_data.py     # yfinance + Alpha Vantage, caching, retries
│   │   └── news_tool.py       # Yahoo Finance RSS
│   ├── rag/
│   │   ├── vector_store.py    # Chroma (dev) / Pinecone (prod) abstraction
│   │   ├── ingest.py          # knowledge-base ingestion script
│   │   └── knowledge_base/    # your curated articles go here (by category)
│   └── mcp/
│       └── server.py          # MCP server for Claude Desktop
└── tests/                     # pytest suite (target: 80%+ coverage)
```

---

## 1. Environment setup

```bash
# 1. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure secrets
cp .env.example .env
```

Edit `.env` and fill in:
- `OPENAI_API_KEY` — from platform.openai.com
- `ALPHA_VANTAGE_API_KEY` — free key from alphavantage.co (yfinance itself
  needs no key; Alpha Vantage is your fallback/secondary source)
- Leave `VECTOR_BACKEND=chroma` for local dev. Switch to `pinecone` + fill
  `PINECONE_API_KEY` when you're ready for production (step 6).

**Never commit `.env`.** Add it to `.gitignore` immediately:
```bash
echo -e ".env\nchroma_db/\n__pycache__/\n*.pyc\nhtmlcov/\n.coverage" >> .gitignore
```

---

## 2. Understand the state & router (the backbone)

Read `src/state.py` and `src/router.py` first — every agent reads/writes the
same `AgentState`, and the router is what makes this a *multi-agent* system
rather than one giant prompt. The router:
1. Calls a cheap, temperature-0 GPT call with structured output (`RouteDecision` pydantic model) to pick one of six routes.
2. Falls back to keyword matching if the LLM call errors — this is your first
   "error handling and fallback mechanism" deliverable, satisfied at the
   entry point of the whole system.

This is a good pattern to explain in your capstone writeup: **LLM-first,
deterministic-fallback** routing balances flexibility with reliability.

---

## 3. Build the knowledge base (RAG) — 50-100 articles

The ingestion pipeline (`src/rag/ingest.py`) expects markdown files under
`src/rag/knowledge_base/<category>/*.md`, one category per the 6 agent
domains that need grounding: `finance_qa`, `tax_education`, `goal_planning`,
`market_analysis`, `portfolio_analysis`, `news_synthesizer`. Three sample
articles are already there so you can test end-to-end immediately.

**To reach 50-100 articles:**
1. Write/curate articles yourself (recommended for a capstone — shows
   original work) covering topics like: index funds, bonds vs stocks, risk
   tolerance, dollar-cost averaging, emergency funds, 401k vs IRA vs Roth,
   capital gains vs ordinary income, HSAs, compound interest, asset
   allocation by age, dividend investing, ETFs vs mutual funds, inflation,
   P/E ratios, market cap tiers, bull vs bear markets, etc. Aim for
   15-20 articles per category × 5-6 categories ≈ 75-100.
2. Each file needs YAML frontmatter for **source attribution**:
   ```markdown
   ---
   title: What is Dollar-Cost Averaging?
   source: Internal Financial Education Team   (or cite a real source you adapted from)
   category: finance_qa
   ---
   Article body here...
   ```
3. Run ingestion:
   ```bash
   python -m src.rag.ingest
   ```
   This chunks (800 chars, 120 overlap), embeds with
   `text-embedding-3-small`, and writes to Chroma at `./chroma_db`.

**Category-based filtering** is already wired: `similarity_search(query,
category="tax_education")` filters retrieval to that category's metadata —
this is what each agent uses so the Tax agent doesn't retrieve Portfolio
articles, etc. (see `src/agents/base.py::retrieve_context`).

**Source attribution** shows up automatically in every agent response via
`state["sources"]`, surfaced in the Streamlit UI caption under each chat
message.

---

## 4. Real-time market data integration

`src/tools/market_data.py` implements:
- **Primary source: yfinance** (`get_live_quote`, `get_fundamentals`,
  `get_price_history`) — no API key, good rate limits.
- **Secondary/fallback: Alpha Vantage** (`_alpha_vantage_quote`) — used
  automatically if yfinance fails.
- **Caching**: `cachetools.TTLCache` (5 min default via `CACHE_TTL_SECONDS`)
  avoids redundant calls and smooths over rate limits — this is your
  "implement caching strategy for performance" deliverable.
- **Retry + backoff**: `tenacity` retries Alpha Vantage calls with
  exponential backoff before giving up.
- **Graceful failure**: every function raises a clean `MarketDataError` with
  a user-friendly message; agents catch it and never crash the conversation.
- **Batch fetch with partial failure tolerance**: `get_portfolio_quotes`
  returns both successes and per-ticker errors, so one bad ticker in a
  10-stock portfolio doesn't kill the whole analysis.

Test it standalone before wiring into agents:
```bash
python -c "from src.tools.market_data import get_live_quote; print(get_live_quote('AAPL'))"
```

---

## 5. Build & run the six agents + orchestration graph

Each agent (`src/agents/*.py`) follows the same shape: pull context (RAG
docs and/or live data) → build a prompt with a strict system prompt (scope,
tone, disclaimers) → call GPT → populate `final_response` + `sources` on
state → catch exceptions into a graceful fallback message + `state["error"]`.

`src/graph.py` wires them together:
```python
graph.set_entry_point("router")
graph.add_conditional_edges("router", route_selector, {r: r for r in ROUTES})
for name in _AGENT_FN:
    graph.add_edge(name, END)
graph = graph.compile(checkpointer=MemorySaver())
```
`MemorySaver` gives you in-process conversation memory keyed by
`thread_id` = your `session_id`. **For production**, swap this for
`SqliteSaver` or `PostgresSaver` (both drop-in from
`langgraph.checkpoint.sqlite` / `.postgres`) so state survives restarts and
works across multiple app instances.

Run a quick smoke test:
```bash
python -c "
from src.graph import invoke
r = invoke('What is a Roth IRA?', user_id='u1', session_id='s1')
print(r)
"
```

---

## 6. Vector DB: dev → production

`src/rag/vector_store.py` abstracts Chroma vs Pinecone behind one
`get_retriever()` / `similarity_search()` interface.

- **Dev**: `VECTOR_BACKEND=chroma` — zero setup, persists to `./chroma_db`.
- **Production**: `VECTOR_BACKEND=pinecone` — install `langchain-pinecone`
  and `pinecone` (`pip install langchain-pinecone pinecone`), set
  `PINECONE_API_KEY`/`PINECONE_INDEX_NAME` in `.env`. The code
  auto-creates a serverless index (1536-dim, cosine) on first run.
- **FAISS** (mentioned in your preferences) is best used as a *local,
  ephemeral* cache layer (e.g., in-memory re-ranking) rather than your
  system of record — Chroma/Pinecone already give you persistence and
  metadata filtering, which FAISS alone doesn't. If your rubric specifically
  requires FAISS, you can add a third branch in `get_vectorstore()` using
  `langchain_community.vectorstores.FAISS.from_documents(...)` with
  `.save_local()` / `.load_local()`.

Re-run `python -m src.rag.ingest` any time you add articles; it's additive
(re-embeds new docs into the same collection/index).

---

## 7. Streamlit UI

```bash
streamlit run app.py
```

`app.py` gives you three tabs, all required by your deliverables:
- **Chat**: conversational interface, shows which agent handled each turn
  and its sources — good for demoing the multi-agent routing to evaluators.
- **Portfolio**: add holdings → live P/L, allocation pie chart, gain/loss bar
  chart (Plotly), then triggers the Portfolio Analysis Agent for a written
  AI analysis grounded in the actual numbers.
- **Market**: ticker lookup → live quote metrics + candlestick chart over a
  selectable period.

Session ID in the sidebar maps directly to the LangGraph `thread_id`, so
"New Session" gives a clean conversational memory slate — this demonstrates
your "user sessions" deliverable clearly to an evaluator.

---

## 8. Testing (80%+ coverage target)

```bash
pytest
```

`pytest.ini` runs with `--cov=src --cov=config --cov-fail-under=80` and
writes an HTML report to `htmlcov/index.html`. The existing suite
(`tests/test_router.py`, `tests/test_market_data.py`, `tests/test_agents.py`)
covers:
- Router: LLM-success path, LLM-failure→keyword-fallback path, all six
  keyword categories, conditional-edge selection.
- Market data: yfinance success, fallback to Alpha Vantage, both-sources-fail
  → `MarketDataError`, caching behavior, partial-failure batch fetch.
- All six agents: happy path (mocked LLM + mocked tools) and at least one
  failure/edge case each (empty portfolio, no ticker found, empty news, LLM
  exception).

**To push coverage past 80%**, add:
- `tests/test_graph.py` — build the graph with a fake checkpointer and
  assert routing end-to-end with a mocked router LLM.
- `tests/test_vector_store.py` — mock `OpenAIEmbeddings`/`Chroma` and test
  `similarity_search`'s document-shaping logic.
- `tests/test_ingest.py` — test `_parse_frontmatter` and `load_documents`
  against a temp directory of fixture `.md` files.

Everything is mocked (`unittest.mock.patch`) so your test suite never makes
real API calls — fast, deterministic, and safe to run in CI without secrets.

---

## 9. Error handling & fallback mechanisms (cross-cutting)

You already have layered resilience — call this out explicitly in your
capstone report:
1. **Router**: LLM routing → keyword-heuristic fallback.
2. **Market data**: yfinance → Alpha Vantage → clean user-facing error.
3. **RAG retrieval**: failures return `[]` instead of raising, so agents
   degrade to un-grounded (but still functional) LLM answers.
4. **Every agent**: wrapped in try/except with a friendly fallback message
   and `state["error"]` populated for logging/observability, never a raw
   stack trace shown to the user.
5. **Portfolio batch fetch**: partial failures don't block the whole
   analysis.

---

## 10. MCP server for Claude Desktop

`src/mcp/server.py` exposes five tools via `FastMCP`:
`get_stock_quote`, `get_stock_fundamentals`, `get_financial_news`,
`search_finance_knowledge_base`, and `ask_financial_assistant` (the full
multi-agent pipeline as one callable tool).

Run it standalone to verify it starts:
```bash
python -m src.mcp.server
```

**Connect to Claude Desktop** — edit (or create)
`~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or
`%APPDATA%\Claude\claude_desktop_config.json` (Windows):
```json
{
  "mcpServers": {
    "finance-assistant": {
      "command": "python",
      "args": ["-m", "src.mcp.server"],
      "cwd": "/absolute/path/to/capstone_finance_agent",
      "env": { "OPENAI_API_KEY": "sk-...", "ALPHA_VANTAGE_API_KEY": "..." }
    }
  }
}
```
Restart Claude Desktop; you should see a 🔨 tools icon confirming the server
connected, and you can ask Claude Desktop things like "get me a live quote
for NVDA" and it will call your tool.

**Document this** in your submission: protocol used (MCP over stdio
transport), tool list + descriptions (already in each `@mcp.tool()`
docstring — MCP surfaces these to the client automatically), and a
screenshot of Claude Desktop successfully calling a tool.

---

## 11. Production-readiness checklist

- [ ] Move `MemorySaver` → `SqliteSaver`/`PostgresSaver` for durable state
- [ ] Add structured logging/observability (the `get_logger` calls are
      already in place — wire to a log aggregator in production)
- [ ] Add per-user rate limiting in front of the Streamlit app / API layer
- [ ] Add input validation/guardrails on portfolio entry (ticker existence
      check before adding to `st.session_state.portfolio`)
- [ ] Set `VECTOR_BACKEND=pinecone` and load the full 50-100 article KB
- [ ] Add a `Dockerfile` + `docker-compose.yml` if containerized deployment
      is required by your rubric
- [ ] Run `pytest` in CI (GitHub Actions) on every push
- [ ] Add disclaimers (already present in every agent's system prompt) —
      keep them, this is a financial-advice-adjacent product

---

## 12. Suggested capstone write-up structure

1. Problem statement & architecture diagram (use the diagram in §0)
2. Design decisions: why LangGraph over a single agent, why LLM+fallback
   routing, why Chroma→Pinecone migration path
3. Agent specialization rationale (one paragraph per agent)
4. RAG knowledge base curation methodology + category taxonomy
5. Real-time data resilience strategy (caching, retries, fallback sources)
6. Test coverage report (`pytest --cov` output / `htmlcov`)
7. MCP integration screenshot + protocol notes
8. Known limitations & future work (e.g., multi-agent collaboration on a
   single query — e.g., Goal Planning agent calling Market Analysis agent
   as a sub-tool — is a good "future work" extension beyond this scaffold)

Good luck with the capstone — this scaffold is deliberately built so every
box in your deliverables list maps to a specific file and line you can point
an evaluator to.
