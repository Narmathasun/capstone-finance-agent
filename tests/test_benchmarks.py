from benchmarks.run_benchmarks import compute_stats, time_call


class TestComputeStats:
    def test_empty_list_returns_zero_count(self):
        result = compute_stats([])
        assert result == {"count": 0}

    def test_single_value(self):
        result = compute_stats([0.1])
        assert result["count"] == 1
        assert result["min_ms"] == 100.0
        assert result["max_ms"] == 100.0
        assert result["mean_ms"] == 100.0
        assert result["median_ms"] == 100.0

    def test_multiple_values_computes_correct_stats(self):
        # 10ms, 20ms, 30ms, 40ms, 50ms
        result = compute_stats([0.01, 0.02, 0.03, 0.04, 0.05])
        assert result["count"] == 5
        assert result["min_ms"] == 10.0
        assert result["max_ms"] == 50.0
        assert result["mean_ms"] == 30.0
        assert result["median_ms"] == 30.0

    def test_p95_is_near_the_high_end_for_larger_samples(self):
        durations = [i / 1000 for i in range(1, 101)]  # 1ms..100ms
        result = compute_stats(durations)
        assert result["p95_ms"] >= 90.0
        assert result["p95_ms"] <= 100.0


class TestTimeCall:
    def test_successful_calls_produce_stats(self):
        stats = time_call(lambda: 1 + 1, iterations=3)
        assert stats["count"] == 3
        assert stats["errors"] == 0

    def test_failing_calls_are_counted_not_fatal(self):
        def _always_fails():
            raise ValueError("simulated failure")

        stats = time_call(_always_fails, iterations=3)
        assert stats["count"] == 0
        assert stats["errors"] == 3

    def test_partial_failures_still_produce_stats_for_successes(self):
        calls = {"count": 0}

        def _fails_every_other_call():
            calls["count"] += 1
            if calls["count"] % 2 == 0:
                raise RuntimeError("simulated intermittent failure")

        stats = time_call(_fails_every_other_call, iterations=4)
        assert stats["count"] == 2
        assert stats["errors"] == 2
