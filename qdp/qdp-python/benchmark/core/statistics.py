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

"""Statistical measurement framework for benchmarks.

This module provides tools for collecting and analyzing benchmark timing data
with proper statistical rigor, including warmup runs, multiple measurements,
and outlier detection.

Example:
    >>> from benchmark.core.statistics import StatisticalRunner, BenchmarkStats
    >>> runner = StatisticalRunner(warmup_runs=3, measurement_runs=10)
    >>> stats, result = runner.run(my_benchmark_fn, *args)
    >>> print(stats.summary())
    10.5 +/- 0.3 ms (median=10.4, p95=11.2, n=10)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Tuple, TypeVar

import numpy as np

if TYPE_CHECKING:
    from .timing import BaseTimer

T = TypeVar("T")


@dataclass(frozen=True)
class BenchmarkStats:
    """Immutable container for benchmark timing statistics.

    All timing values are in milliseconds.

    Attributes:
        mean: Arithmetic mean of measurements.
        std: Standard deviation (sample, ddof=1).
        min: Minimum observed value.
        max: Maximum observed value.
        median: Median (50th percentile).
        p5: 5th percentile.
        p25: 25th percentile (Q1).
        p75: 75th percentile (Q3).
        p95: 95th percentile.
        p99: 99th percentile.
        n_samples: Number of samples used in statistics.
        n_outliers_removed: Number of outliers filtered out.
        raw_samples: Original timing samples (tuple for immutability).
    """

    mean: float
    std: float
    min: float
    max: float
    median: float
    p5: float
    p25: float
    p75: float
    p95: float
    p99: float
    n_samples: int
    n_outliers_removed: int = 0
    raw_samples: tuple = field(default_factory=tuple, repr=False)

    @classmethod
    def from_samples(
        cls,
        samples: List[float],
        remove_outliers: bool = False,
        iqr_multiplier: float = 1.5,
    ) -> BenchmarkStats:
        """Compute statistics from raw timing samples.

        Args:
            samples: List of timing measurements in milliseconds.
            remove_outliers: If True, filter outliers using IQR method.
            iqr_multiplier: Multiplier for IQR-based outlier detection.
                Values outside [Q1 - iqr_multiplier*IQR, Q3 + iqr_multiplier*IQR]
                are considered outliers. Default is 1.5.

        Returns:
            BenchmarkStats instance with computed statistics.

        Raises:
            ValueError: If samples is empty.
        """
        if not samples:
            raise ValueError("Cannot compute statistics from empty samples")

        arr = np.array(samples, dtype=np.float64)
        n_outliers = 0

        if remove_outliers and len(arr) > 4:
            q1, q3 = np.percentile(arr, [25, 75])
            iqr = q3 - q1
            lower = q1 - iqr_multiplier * iqr
            upper = q3 + iqr_multiplier * iqr
            mask = (arr >= lower) & (arr <= upper)
            n_outliers = len(arr) - int(mask.sum())
            arr = arr[mask]

        # Handle edge case where all samples are outliers
        if len(arr) == 0:
            arr = np.array(samples, dtype=np.float64)
            n_outliers = 0

        return cls(
            mean=float(np.mean(arr)),
            std=float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
            min=float(np.min(arr)),
            max=float(np.max(arr)),
            median=float(np.median(arr)),
            p5=float(np.percentile(arr, 5)),
            p25=float(np.percentile(arr, 25)),
            p75=float(np.percentile(arr, 75)),
            p95=float(np.percentile(arr, 95)),
            p99=float(np.percentile(arr, 99)),
            n_samples=len(arr),
            n_outliers_removed=n_outliers,
            raw_samples=tuple(samples),
        )

    def summary(self, unit: str = "ms") -> str:
        """Return a human-readable summary string.

        Args:
            unit: Unit label for the output (default: "ms").

        Returns:
            Formatted string like "10.5 +/- 0.3 ms (median=10.4, p95=11.2, n=10)"
        """
        return (
            f"{self.mean:.3f} +/- {self.std:.3f} {unit} "
            f"(median={self.median:.3f}, p95={self.p95:.3f}, n={self.n_samples})"
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "mean": self.mean,
            "std": self.std,
            "min": self.min,
            "max": self.max,
            "median": self.median,
            "p5": self.p5,
            "p25": self.p25,
            "p75": self.p75,
            "p95": self.p95,
            "p99": self.p99,
            "n_samples": self.n_samples,
            "n_outliers_removed": self.n_outliers_removed,
        }


class StatisticalRunner:
    """Run a benchmark function multiple times and collect statistics.

    This class handles the warmup phase, measurement phase, cache clearing,
    and statistics computation for benchmark functions.

    Example:
        >>> runner = StatisticalRunner(warmup_runs=3, measurement_runs=10)
        >>> stats, result = runner.run(encode_batch, data, num_qubits=16)
        >>> print(f"Encoding: {stats.summary()}")
    """

    def __init__(
        self,
        warmup_runs: int = 3,
        measurement_runs: int = 10,
        remove_outliers: bool = False,
        clear_cache_between_runs: bool = True,
    ):
        """Initialize the statistical runner.

        Args:
            warmup_runs: Number of warmup iterations before measurement.
            measurement_runs: Number of timed iterations for statistics.
            remove_outliers: If True, apply IQR outlier filtering.
            clear_cache_between_runs: If True, clear GPU caches between runs.
        """
        if warmup_runs < 0:
            raise ValueError("warmup_runs must be non-negative")
        if measurement_runs < 1:
            raise ValueError("measurement_runs must be at least 1")

        self.warmup_runs = warmup_runs
        self.measurement_runs = measurement_runs
        self.remove_outliers = remove_outliers
        self.clear_cache = clear_cache_between_runs

    def run(
        self,
        fn: Callable[..., T],
        *args: Any,
        timer: Optional[BaseTimer] = None,
        **kwargs: Any,
    ) -> Tuple[BenchmarkStats, T]:
        """Execute benchmark with warmup and measurement phases.

        Args:
            fn: Benchmark function to call.
            *args: Positional arguments passed to fn.
            timer: Timer instance to use. If None, auto-selects CUDATimer
                if CUDA is available, otherwise WallClockTimer.
            **kwargs: Keyword arguments passed to fn.

        Returns:
            Tuple of (BenchmarkStats, last_result) where last_result is
            the return value from the final measurement run.
        """
        from .cuda import clear_gpu_caches, is_cuda_available
        from .timing import CUDATimer, WallClockTimer

        if timer is None:
            timer = CUDATimer() if is_cuda_available() else WallClockTimer()

        # Warmup phase
        for _ in range(self.warmup_runs):
            _ = fn(*args, **kwargs)
            if self.clear_cache:
                clear_gpu_caches()

        # Measurement phase
        samples: List[float] = []
        result: Optional[T] = None

        for _ in range(self.measurement_runs):
            if self.clear_cache:
                clear_gpu_caches()

            timer.reset()
            timer.start()
            result = fn(*args, **kwargs)
            timer.stop()

            samples.append(timer.elapsed_ms)

        stats = BenchmarkStats.from_samples(
            samples, remove_outliers=self.remove_outliers
        )
        return stats, result  # type: ignore[return-value]
