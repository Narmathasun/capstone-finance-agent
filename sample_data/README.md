# Sample Data

Standalone example data for manual testing, demos, and benchmarking —
independent of the pytest fixtures in `tests/conftest.py` (which are
scoped narrowly to what each unit test needs). These files are meant to
be inspected directly, loaded into the running app by hand, or reused by
scripts like `benchmarks/run_benchmarks.py`.

## Files

### `sample_portfolio.json`
Five illustrative holdings (a mix of individual stocks and a broad-market
ETF) for exercising the Portfolio Analysis Agent and the Streamlit
Portfolio tab's dashboard (allocation chart, P/L, sector breakdown).

**To use manually:** open the Streamlit app's Portfolio tab and enter each
holding from the `holdings` array one at a time (ticker, shares, cost
basis).

**To use in a script:**
```python
import json
with open("sample_data/sample_portfolio.json") as f:
    data = json.load(f)
portfolio = data["holdings"]  # -> [{"ticker": "AAPL", "shares": 15, "cost_basis": 145.30}, ...]
```

### `sample_queries.json`
One representative query per agent domain — the same six categories used
throughout this project's testing and demo walkthroughs. This file is the
single source of truth for "one example question per agent," reused
directly by `benchmarks/run_benchmarks.py` so the benchmark suite and any
manual demo stay in sync rather than drifting into two separate lists.

**To use manually:** paste any value into the Streamlit Chat tab or the
MCP `ask_financial_assistant` tool.

**To use in a script:**
```python
import json
with open("sample_data/sample_queries.json") as f:
    queries = json.load(f)
# queries["tax_education"] -> "What is the difference between a Roth IRA and a Traditional IRA?"
```

## Why these live outside `tests/`

Pytest fixtures in `conftest.py` are intentionally minimal — just enough
to exercise a specific unit test's logic path. These files serve a
different purpose: a realistic, human-readable example a person (or a
grader) can open directly, paste into the running app, or point a script
at, without needing to read test code to find one.
