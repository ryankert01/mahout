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

"""Tests for benchmark utilities."""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add benchmark directory to path
benchmark_dir = Path(__file__).parent.parent / "benchmark"
sys.path.insert(0, str(benchmark_dir))

from benchmark_utils import statistics, timing

try:
    import torch

    HAS_TORCH = torch.cuda.is_available()
except ImportError:
    HAS_TORCH = False


class TestStatistics:
    """Test suite for statistics module."""

    def test_compute_statistics_basic(self):
        """Test basic statistics computation."""
        timings = [10.0, 10.2, 10.1, 10.3, 10.4]
        stats = statistics.compute_statistics(timings)

        assert "mean" in stats
        assert "median" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats
        assert "p25" in stats
        assert "p50" in stats
        assert "p75" in stats
        assert "p90" in stats
        assert "p95" in stats
        assert "p99" in stats
        assert "iqr" in stats
        assert "cv" in stats
        assert "n_runs" in stats

        assert stats["n_runs"] == 5
        assert stats["min"] == 10.0
        assert stats["max"] == 10.4
        assert abs(stats["mean"] - 10.2) < 0.01

    def test_compute_statistics_empty(self):
        """Test that empty timings raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            statistics.compute_statistics([])

    def test_filter_outliers_iqr(self):
        """Test IQR-based outlier filtering."""
        timings = [10.1, 10.2, 10.3, 50.0, 10.2, 10.4]  # 50.0 is an outlier
        filtered = statistics.filter_outliers(timings, method="iqr", threshold=1.5)

        assert len(filtered) < len(timings)
        assert 50.0 not in filtered

    def test_filter_outliers_zscore(self):
        """Test z-score based outlier filtering."""
        timings = [10.0] * 10 + [100.0]  # 100.0 is an outlier
        filtered = statistics.filter_outliers(timings, method="zscore", threshold=3.0)

        assert len(filtered) < len(timings)
        assert 100.0 not in filtered

    def test_filter_outliers_invalid_method(self):
        """Test that invalid method raises ValueError."""
        with pytest.raises(ValueError, match="Unknown method"):
            statistics.filter_outliers([10.0, 10.1], method="invalid")

    def test_filter_outliers_empty(self):
        """Test that empty timings raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            statistics.filter_outliers([])

    def test_compute_confidence_interval(self):
        """Test confidence interval computation."""
        # Need scipy for this test
        pytest.importorskip("scipy")

        timings = [10.0, 10.2, 10.1, 10.3, 10.4] * 10
        lower, upper = statistics.compute_confidence_interval(timings, confidence=0.95)

        mean = np.mean(timings)
        assert lower < mean < upper
        assert isinstance(lower, float)
        assert isinstance(upper, float)

    def test_compute_confidence_interval_invalid(self):
        """Test confidence interval with invalid inputs."""
        pytest.importorskip("scipy")

        with pytest.raises(ValueError, match="at least 2 measurements"):
            statistics.compute_confidence_interval([10.0])

        with pytest.raises(ValueError, match="between 0 and 1"):
            statistics.compute_confidence_interval([10.0, 10.1], confidence=1.5)

    def test_format_statistics(self):
        """Test statistics formatting."""
        stats = {
            "mean": 10.25,
            "median": 10.20,
            "std": 0.15,
            "min": 10.10,
            "max": 10.40,
            "p95": 10.38,
            "cv": 0.0146,
            "n_runs": 5,
        }

        formatted = statistics.format_statistics(stats)
        assert "10.25" in formatted
        assert "ms" in formatted
        assert "Runs:" in formatted


class TestTiming:
    """Test suite for timing module."""

    def test_warmup_cpu(self):
        """Test warmup function with CPU operation."""
        counter = {"count": 0}

        def dummy_func():
            counter["count"] += 1

        timing.warmup(dummy_func, warmup_iters=5)
        assert counter["count"] == 5

    @pytest.mark.gpu
    @pytest.mark.skipif(not HAS_TORCH, reason="CUDA not available")
    def test_warmup_gpu(self):
        """Test warmup function with GPU operation."""
        counter = {"count": 0}

        def dummy_func():
            x = torch.randn(10, 10, device="cuda")
            _ = x @ x.T
            counter["count"] += 1

        timing.warmup(dummy_func, warmup_iters=3)
        assert counter["count"] == 3

    def test_clear_all_caches(self):
        """Test cache clearing doesn't raise errors."""
        # Should not raise any exceptions
        timing.clear_all_caches()

    @pytest.mark.gpu
    @pytest.mark.skipif(not HAS_TORCH, reason="CUDA not available")
    def test_clear_l2_cache(self):
        """Test L2 cache clearing."""
        # Should not raise any exceptions
        timing.clear_l2_cache(cache_size_mb=100)

    def test_benchmark_cpu_function(self):
        """Test CPU function benchmarking."""

        def dummy_func():
            x = [i**2 for i in range(100)]
            return sum(x)

        timings = timing.benchmark_cpu_function(dummy_func, warmup_iters=2, repeat=10)

        assert len(timings) == 10
        assert all(t > 0 for t in timings)
        assert all(isinstance(t, float) for t in timings)

    @pytest.mark.gpu
    @pytest.mark.skipif(not HAS_TORCH, reason="CUDA not available")
    def test_benchmark_with_cuda_events(self):
        """Test CUDA event-based benchmarking."""

        def dummy_func():
            x = torch.randn(100, 100, device="cuda")
            return (x @ x.T).sum()

        timings = timing.benchmark_with_cuda_events(
            dummy_func, warmup_iters=2, repeat=10, clear_cache_between_runs=False
        )

        assert len(timings) == 10
        assert all(t > 0 for t in timings)
        assert all(isinstance(t, float) for t in timings)

    @pytest.mark.gpu
    @pytest.mark.skipif(not HAS_TORCH, reason="CUDA not available")
    def test_benchmark_with_cuda_events_with_cache_clear(self):
        """Test CUDA benchmarking with cache clearing between runs."""

        def dummy_func():
            x = torch.randn(50, 50, device="cuda")
            return (x @ x.T).sum()

        timings = timing.benchmark_with_cuda_events(
            dummy_func, warmup_iters=1, repeat=5, clear_cache_between_runs=True
        )

        assert len(timings) == 5
        assert all(t > 0 for t in timings)

    def test_benchmark_with_cuda_events_no_cuda(self):
        """Test that benchmarking without CUDA raises error."""
        if HAS_TORCH:
            pytest.skip("CUDA is available, cannot test no-CUDA case")

        def dummy_func():
            pass

        with pytest.raises(RuntimeError, match="PyTorch is required"):
            timing.benchmark_with_cuda_events(dummy_func)

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    def test_benchmark_with_cuda_events_invalid_params(self):
        """Test that invalid parameters raise errors."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        def dummy_func():
            pass

        with pytest.raises(ValueError, match="must be positive"):
            timing.benchmark_with_cuda_events(dummy_func, warmup_iters=0, repeat=10)

        with pytest.raises(ValueError, match="must be positive"):
            timing.benchmark_with_cuda_events(dummy_func, warmup_iters=3, repeat=-1)
