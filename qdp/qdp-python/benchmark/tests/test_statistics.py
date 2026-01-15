#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for benchmark.core.statistics module."""

import pytest

from benchmark.core.statistics import BenchmarkStats, StatisticalRunner


class TestBenchmarkStats:
    """Tests for BenchmarkStats dataclass."""

    def test_from_samples_basic(self, sample_timings):
        """Test basic statistics computation."""
        stats = BenchmarkStats.from_samples(sample_timings)

        assert stats.n_samples == 10
        assert 10.5 <= stats.mean <= 11.5
        assert stats.std > 0
        assert stats.min == pytest.approx(10.5, abs=0.01)
        assert stats.max == pytest.approx(11.3, abs=0.01)

    def test_from_samples_percentiles(self, sample_timings):
        """Test percentile ordering."""
        stats = BenchmarkStats.from_samples(sample_timings)

        # Percentiles should be ordered
        assert (
            stats.p5 <= stats.p25 <= stats.median <= stats.p75 <= stats.p95 <= stats.p99
        )

    def test_outlier_removal(self, sample_timings_with_outliers):
        """Test IQR-based outlier filtering."""
        stats_with = BenchmarkStats.from_samples(
            sample_timings_with_outliers, remove_outliers=True
        )
        stats_without = BenchmarkStats.from_samples(
            sample_timings_with_outliers, remove_outliers=False
        )

        # With outlier removal, mean should be closer to ~11
        assert abs(stats_with.mean - 11.0) < abs(stats_without.mean - 11.0)
        assert stats_with.n_outliers_removed > 0

    def test_single_sample(self):
        """Test statistics with single sample."""
        stats = BenchmarkStats.from_samples([5.0])

        assert stats.mean == 5.0
        assert stats.std == 0.0
        assert stats.n_samples == 1
        assert stats.min == stats.max == stats.median == 5.0

    def test_empty_samples_raises(self):
        """Test that empty samples raises ValueError."""
        with pytest.raises(ValueError, match="empty samples"):
            BenchmarkStats.from_samples([])

    def test_immutability(self, sample_timings):
        """Test that BenchmarkStats is immutable (frozen dataclass)."""
        stats = BenchmarkStats.from_samples(sample_timings)

        with pytest.raises(Exception):  # frozen dataclass raises FrozenInstanceError
            stats.mean = 999.0

    def test_summary_format(self, sample_timings):
        """Test summary string format."""
        stats = BenchmarkStats.from_samples(sample_timings)
        summary = stats.summary()

        assert "ms" in summary
        assert "+/-" in summary
        assert "median=" in summary
        assert "p95=" in summary
        assert "n=" in summary

    def test_summary_custom_unit(self, sample_timings):
        """Test summary with custom unit."""
        stats = BenchmarkStats.from_samples(sample_timings)
        summary = stats.summary(unit="us")

        assert "us" in summary

    def test_to_dict(self, sample_timings):
        """Test dictionary conversion."""
        stats = BenchmarkStats.from_samples(sample_timings)
        d = stats.to_dict()

        assert "mean" in d
        assert "std" in d
        assert "p95" in d
        assert d["n_samples"] == 10
        # raw_samples should not be in dict
        assert "raw_samples" not in d

    def test_raw_samples_preserved(self, sample_timings):
        """Test that raw samples are preserved."""
        stats = BenchmarkStats.from_samples(sample_timings)

        assert len(stats.raw_samples) == len(sample_timings)
        assert stats.raw_samples == tuple(sample_timings)

    def test_all_outliers_fallback(self):
        """Test fallback when all samples would be filtered as outliers."""
        # Extreme case: two very different values
        samples = [1.0, 1000.0]
        stats = BenchmarkStats.from_samples(samples, remove_outliers=True)

        # Should still produce valid stats (fallback to no filtering)
        assert stats.n_samples >= 1


class TestStatisticalRunner:
    """Tests for StatisticalRunner class."""

    def test_run_with_simple_function(self):
        """Test basic runner functionality."""
        call_count = 0

        def dummy_fn():
            nonlocal call_count
            call_count += 1
            return 42

        runner = StatisticalRunner(
            warmup_runs=2,
            measurement_runs=5,
            clear_cache_between_runs=False,  # Disable cache clearing for unit test
        )
        stats, result = runner.run(dummy_fn)

        assert result == 42
        assert call_count == 7  # 2 warmup + 5 measurement
        assert stats.n_samples == 5

    def test_run_with_args(self):
        """Test runner passes arguments correctly."""

        def add(a, b, c=0):
            return a + b + c

        runner = StatisticalRunner(
            warmup_runs=1, measurement_runs=3, clear_cache_between_runs=False
        )
        stats, result = runner.run(add, 1, 2, c=3)

        assert result == 6

    def test_invalid_warmup_runs(self):
        """Test that negative warmup_runs raises ValueError."""
        with pytest.raises(ValueError, match="warmup_runs"):
            StatisticalRunner(warmup_runs=-1)

    def test_invalid_measurement_runs(self):
        """Test that zero measurement_runs raises ValueError."""
        with pytest.raises(ValueError, match="measurement_runs"):
            StatisticalRunner(measurement_runs=0)

    def test_runner_collects_timing_samples(self):
        """Test that runner collects timing data."""
        import time

        def slow_fn():
            time.sleep(0.001)  # 1ms
            return "done"

        runner = StatisticalRunner(
            warmup_runs=1, measurement_runs=5, clear_cache_between_runs=False
        )
        stats, result = runner.run(slow_fn)

        assert result == "done"
        assert stats.mean > 0  # Should have measured some time
        assert stats.n_samples == 5
