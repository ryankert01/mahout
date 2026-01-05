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

"""
Statistical computations for benchmark results.

This module provides functions to compute comprehensive statistics from
benchmark timing measurements, including mean, median, standard deviation,
percentiles, and outlier detection.
"""

from typing import Dict, List, Union

import numpy as np


def compute_statistics(timings: List[float]) -> Dict[str, float]:
    """
    Compute comprehensive statistics from benchmark timings.

    This function calculates a full suite of statistical metrics suitable
    for benchmark analysis and publication. All timing values should be
    in milliseconds.

    Args:
        timings: List of timing measurements in milliseconds

    Returns:
        Dictionary with statistical metrics:
            - mean: Arithmetic mean
            - median: 50th percentile
            - std: Standard deviation
            - min: Minimum value
            - max: Maximum value
            - p25: 25th percentile
            - p50: 50th percentile (same as median)
            - p75: 75th percentile
            - p90: 90th percentile
            - p95: 95th percentile
            - p99: 99th percentile
            - iqr: Interquartile range (p75 - p25)
            - cv: Coefficient of variation (std / mean)
            - n_runs: Number of measurements

    Raises:
        ValueError: If timings list is empty

    Example:
        >>> timings = [10.2, 10.5, 10.1, 10.3, 10.4]
        >>> stats = compute_statistics(timings)
        >>> print(f"Mean: {stats['mean']:.2f} ms ± {stats['std']:.2f} ms")
        Mean: 10.30 ms ± 0.16 ms
    """
    if not timings:
        raise ValueError("Timings list cannot be empty")

    timings_arr = np.array(timings)
    mean_val = np.mean(timings_arr)
    std_val = np.std(timings_arr)

    return {
        "mean": float(mean_val),
        "median": float(np.median(timings_arr)),
        "std": float(std_val),
        "min": float(np.min(timings_arr)),
        "max": float(np.max(timings_arr)),
        "p25": float(np.percentile(timings_arr, 25)),
        "p50": float(np.percentile(timings_arr, 50)),  # Same as median
        "p75": float(np.percentile(timings_arr, 75)),
        "p90": float(np.percentile(timings_arr, 90)),
        "p95": float(np.percentile(timings_arr, 95)),
        "p99": float(np.percentile(timings_arr, 99)),
        "iqr": float(np.percentile(timings_arr, 75) - np.percentile(timings_arr, 25)),
        "cv": float(std_val / mean_val if mean_val > 0 else 0),  # Coefficient of variation
        "n_runs": len(timings_arr),
    }


def filter_outliers(
    timings: List[float],
    method: str = "iqr",
    threshold: Union[float, int] = 1.5,
) -> np.ndarray:
    """
    Remove outliers from timing measurements.

    This function provides two methods for outlier detection and removal:
    - IQR (Interquartile Range): Uses the 25th and 75th percentiles
    - Z-score: Uses standard deviations from the mean

    Args:
        timings: List of measurements in milliseconds
        method: Outlier detection method, either 'iqr' or 'zscore' (default: 'iqr')
        threshold: Multiplier for IQR (default: 1.5) or z-score threshold (default: 3.0)
                  For IQR: values outside [Q1 - threshold*IQR, Q3 + threshold*IQR] are outliers
                  For Z-score: values with |z| > threshold are outliers

    Returns:
        Filtered numpy array of timings with outliers removed

    Raises:
        ValueError: If method is not 'iqr' or 'zscore'
        ValueError: If timings list is empty

    Example:
        >>> timings = [10.1, 10.2, 10.3, 50.0, 10.2, 10.4]  # 50.0 is an outlier
        >>> filtered = filter_outliers(timings, method='iqr')
        >>> print(filtered)
        [10.1 10.2 10.3 10.2 10.4]
    """
    if not timings:
        raise ValueError("Timings list cannot be empty")

    if method not in ["iqr", "zscore"]:
        raise ValueError(f"Unknown method '{method}'. Must be 'iqr' or 'zscore'.")

    timings_arr = np.array(timings)

    if method == "iqr":
        q1 = np.percentile(timings_arr, 25)
        q3 = np.percentile(timings_arr, 75)
        iqr = q3 - q1
        lower = q1 - threshold * iqr
        upper = q3 + threshold * iqr
        return timings_arr[(timings_arr >= lower) & (timings_arr <= upper)]

    elif method == "zscore":
        mean = np.mean(timings_arr)
        std = np.std(timings_arr)
        if std == 0:
            # All values are the same, no outliers
            return timings_arr
        z_scores = np.abs((timings_arr - mean) / std)
        return timings_arr[z_scores < threshold]


def compute_confidence_interval(
    timings: List[float],
    confidence: float = 0.95,
) -> tuple[float, float]:
    """
    Compute confidence interval for the mean.

    Uses the t-distribution for small sample sizes (n < 30) and
    normal distribution for larger samples.

    Args:
        timings: List of timing measurements in milliseconds
        confidence: Confidence level (default: 0.95 for 95% CI)

    Returns:
        Tuple of (lower_bound, upper_bound) for the confidence interval

    Raises:
        ValueError: If timings list is empty or has only one element
        ValueError: If confidence is not between 0 and 1

    Example:
        >>> timings = [10.2, 10.5, 10.1, 10.3, 10.4] * 10
        >>> lower, upper = compute_confidence_interval(timings)
        >>> print(f"95% CI: [{lower:.2f}, {upper:.2f}]")
    """
    if not timings or len(timings) < 2:
        raise ValueError("Need at least 2 measurements for confidence interval")

    if not 0 < confidence < 1:
        raise ValueError("Confidence must be between 0 and 1")

    from scipy import stats

    timings_arr = np.array(timings)
    mean = np.mean(timings_arr)
    std_err = stats.sem(timings_arr)  # Standard error of the mean
    n = len(timings_arr)

    # Use t-distribution for small samples, normal for large
    if n < 30:
        ci = stats.t.interval(confidence, n - 1, loc=mean, scale=std_err)
    else:
        ci = stats.norm.interval(confidence, loc=mean, scale=std_err)

    return (float(ci[0]), float(ci[1]))


def format_statistics(stats: Dict[str, float], unit: str = "ms") -> str:
    """
    Format statistics dictionary into a human-readable string.

    Args:
        stats: Statistics dictionary from compute_statistics()
        unit: Unit of measurement (default: "ms")

    Returns:
        Formatted string representation of statistics

    Example:
        >>> stats = compute_statistics([10.1, 10.2, 10.3])
        >>> print(format_statistics(stats))
          Mean:     10.20 ms
          Median:   10.20 ms
          Std:       0.10 ms
          Min:      10.10 ms
          Max:      10.30 ms
          P95:      10.29 ms
          CV:        0.98%
          Runs:         3
    """
    lines = [
        f"  Mean:   {stats['mean']:8.2f} {unit}",
        f"  Median: {stats['median']:8.2f} {unit}",
        f"  Std:    {stats['std']:8.2f} {unit}",
        f"  Min:    {stats['min']:8.2f} {unit}",
        f"  Max:    {stats['max']:8.2f} {unit}",
        f"  P95:    {stats['p95']:8.2f} {unit}",
        f"  CV:     {stats['cv']*100:8.2f}%",
        f"  Runs:   {stats['n_runs']:8d}",
    ]
    return "\n".join(lines)
