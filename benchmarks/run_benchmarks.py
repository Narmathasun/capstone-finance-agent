"""
Performance benchmarks for the multi-agent financial assistant.

Measures wall-clock latency for the operations that matter most to a
user's perceived responsiveness: market data fetches (cold vs. cached),
RAG retrieval, router classification, and full end-to-end agent
round-trips for each of the six agents.

Usage:
    python -m benchmarks.run_benchmarks
    python -m benchmarks.run_benchmarks --iterations 10
    python -m benchmarks.run_benchmarks --skip-llm   # market data + cache only, no OpenAI calls

Requires real credentials in .env (OPENAI_API_KEY, and a live network
connection) to produce meaningful numbers — this hits your actual
configured backends (yfinance/Alpha Vantage, OpenAI, your vector store),
so results reflect your real environment, not a simulation. Results are
printed to the console and written to benchmarks/results/ as a timestamped
markdown report you can compare across runs (e.g., before/after a
prompt or model change).
"""
import argparse
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from config import settings, get_logger

logger = get_logger(__name__)

RESULTS_DIR = Path(__file__).parent / "results"

SAMPLE_TICKERS = ["AAPL", "MSFT", "GOOGL"]

# Reuses the same sample queries used for manual/demo testing, so the
# benchmark suite and the sample-data set stay in sync rather than
# drifting into two separate lists of "example questions."
SAMPLE_QUERIES_PATH = Path(__file__).parent.parent / "sample_data" / "sample_queries.json"


def compute_stats(durations_sec: list[float]) -> dict:
    """Pure function, no I/O — kept separate from the timing code above it
    specifically so it can be unit-tested without needing real API calls."""
    if not durations_sec:
        return {"count": 0}
    sorted_d = sorted(durations_sec)
    n = len(sorted_d)
    p95_index = min(n - 1, int(round(0.95 * (n - 1))))
    return {
        "count": n,
        "min_ms": round(min(sorted_d) * 1000, 1),
        "max_ms": round(max(sorted_d) * 1000, 1),
        "mean_ms": round(statistics.mean(sorted_d) * 1000, 1),
        "median_ms": round(statistics.median(sorted_d) * 1000, 1),
        "p95_ms": round(sorted_d[p95_index] * 1000, 1),
    }


def time_call(fn, *args, iterations=5, **kwargs) -> dict:
    """Times `fn(*args, **kwargs)` across `iterations` calls. Failures are
    caught per-iteration (not fatal to the whole benchmark run) and counted
    separately, matching this project's broader philosophy of partial
    failure tolerance rather than an all-or-nothing run."""
    durations, errors = [], 0
    for _ in range(iterations):
        start = time.perf_counter()
        try:
            fn(*args, **kwargs)
            durations.append(time.perf_counter() - start)
        except Exception as e:
            errors += 1
            logger.warning(f"Benchmark call failed (counted, not fatal): {e}")
    stats = compute_stats(durations)
    stats["errors"] = errors
    return stats


def load_sample_queries() -> dict:
    if SAMPLE_QUERIES_PATH.exists():
        with open(SAMPLE_QUERIES_PATH) as f:
            return json.load(f)
    # Fallback so this script is still runnable standalone even if
    # sample_data/ isn't present for some reason.
    return {
        "finance_qa": "What is dollar-cost averaging?",
        "market_analysis": "How is AAPL doing today?",
        "goal_planning": "How much should I save monthly to reach $100,000 in 10 years?",
        "tax_education": "What is the difference between a Roth and Traditional IRA?",
        "news_synthesizer": "What is the latest news on TSLA?",
        "portfolio_analysis": "Can you analyze my portfolio?",
    }


def benchmark_market_data(iterations: int) -> dict:
    from src.tools.market_data import get_live_quote, _quote_cache

    results = {}
    ticker = SAMPLE_TICKERS[0]

    _quote_cache.clear()
    results["cold_cache_single_quote"] = time_call(get_live_quote, ticker, iterations=1)

    results["warm_cache_single_quote"] = time_call(get_live_quote, ticker, iterations=iterations)

    _quote_cache.clear()
    results["multi_ticker_cold"] = time_call(
        lambda: [get_live_quote(t) for t in SAMPLE_TICKERS], iterations=1
    )
    return results


def benchmark_rag_retrieval(iterations: int) -> dict:
    from src.rag.vector_store import similarity_search

    return {
        "similarity_search_tax_education": time_call(
            similarity_search, "What is a Roth IRA?", k=4, category="tax_education",
            iterations=iterations,
        )
    }


def benchmark_router(iterations: int) -> dict:
    from src.router import route_query

    def _route():
        route_query({"query": "How is AAPL doing today?"})

    return {"router_classification": time_call(_route, iterations=iterations)}


def benchmark_full_agent_roundtrip(iterations: int) -> dict:
    from src.graph import invoke

    queries = load_sample_queries()
    results = {}
    for agent_name, query in queries.items():
        results[f"end_to_end_{agent_name}"] = time_call(
            invoke, query, user_id="benchmark", session_id=f"benchmark-{agent_name}",
            iterations=iterations,
        )
    return results


def write_report(all_results: dict, iterations: int):
    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_path = RESULTS_DIR / f"benchmark_report_{timestamp}.md"

    lines = [
        f"# Benchmark Report — {timestamp} UTC",
        "",
        f"- Iterations per operation: {iterations}",
        f"- LLM model: {settings.OPENAI_MODEL}",
        f"- Vector backend: {settings.VECTOR_BACKEND}",
        f"- Checkpointer backend: {settings.CHECKPOINTER_BACKEND}",
        "",
        "| Operation | Count | Min (ms) | Median (ms) | Mean (ms) | P95 (ms) | Max (ms) | Errors |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for section, ops in all_results.items():
        for op_name, stats in ops.items():
            if stats.get("count", 0) == 0:
                lines.append(f"| {section}.{op_name} | 0 | — | — | — | — | — | {stats.get('errors', 0)} |")
                continue
            lines.append(
                f"| {section}.{op_name} | {stats['count']} | {stats['min_ms']} | "
                f"{stats['median_ms']} | {stats['mean_ms']} | {stats['p95_ms']} | "
                f"{stats['max_ms']} | {stats['errors']} |"
            )
    report_path.write_text("\n".join(lines))
    return report_path


def main():
    parser = argparse.ArgumentParser(description="Run performance benchmarks")
    parser.add_argument("--iterations", type=int, default=5,
                         help="Iterations per operation (default: 5)")
    parser.add_argument("--skip-llm", action="store_true",
                         help="Skip RAG/router/agent benchmarks that call OpenAI; "
                              "market-data-only run")
    args = parser.parse_args()

    print(f"Running benchmarks ({args.iterations} iterations per operation)...")
    print(f"Model: {settings.OPENAI_MODEL} | Vector backend: {settings.VECTOR_BACKEND}\n")

    all_results = {}

    print("Benchmarking market data (live network calls)...")
    all_results["market_data"] = benchmark_market_data(args.iterations)

    if not args.skip_llm:
        print("Benchmarking RAG retrieval (calls OpenAI embeddings)...")
        all_results["rag"] = benchmark_rag_retrieval(args.iterations)

        print("Benchmarking router classification (calls OpenAI)...")
        all_results["router"] = benchmark_router(args.iterations)

        print("Benchmarking full agent round-trips, one query per agent (calls OpenAI)...")
        all_results["full_roundtrip"] = benchmark_full_agent_roundtrip(args.iterations)
    else:
        print("Skipping LLM-dependent benchmarks (--skip-llm)")

    print("\n=== Results ===")
    for section, ops in all_results.items():
        print(f"\n{section}:")
        for op_name, stats in ops.items():
            if stats.get("count", 0) == 0:
                print(f"  {op_name}: no successful calls (errors={stats.get('errors', 0)})")
                continue
            print(
                f"  {op_name}: median={stats['median_ms']}ms  "
                f"mean={stats['mean_ms']}ms  p95={stats['p95_ms']}ms  "
                f"(min={stats['min_ms']}, max={stats['max_ms']}, errors={stats['errors']})"
            )

    report_path = write_report(all_results, args.iterations)
    print(f"\nFull report written to: {report_path}")


if __name__ == "__main__":
    main()
