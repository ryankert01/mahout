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

"""Publication-quality plotting functions for benchmark results.

This module provides matplotlib-based visualization functions designed
for academic papers, blog posts, and presentations.

Example:
    >>> from benchmark.visualization.plots import plot_framework_comparison
    >>> fig = plot_framework_comparison(
    ...     results,
    ...     metric="mean",
    ...     title="Encoding Latency Comparison",
    ... )
    >>> fig.savefig("latency.pdf", dpi=300, bbox_inches="tight")
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Lazy import matplotlib to avoid startup cost
_plt = None
_mpl = None


def _get_matplotlib():
    """Lazy import matplotlib."""
    global _plt, _mpl
    if _plt is None:
        import matplotlib as mpl
        import matplotlib.pyplot as plt

        _plt = plt
        _mpl = mpl
    return _plt, _mpl


# Framework display names and colors
FRAMEWORK_COLORS = {
    "mahout": "#E63946",  # Red - primary focus
    "pennylane": "#457B9D",  # Blue
    "qiskit": "#2A9D8F",  # Teal
    "qiskit-init": "#2A9D8F",
    "qiskit-statevector": "#264653",  # Dark teal
}

FRAMEWORK_LABELS = {
    "mahout": "Mahout QDP",
    "pennylane": "PennyLane",
    "qiskit": "Qiskit",
    "qiskit-init": "Qiskit (Initialize)",
    "qiskit-statevector": "Qiskit (Statevector)",
}


def set_publication_style(
    font_size: int = 10,
    font_family: str = "serif",
    use_latex: bool = False,
) -> None:
    """Configure matplotlib for publication-quality figures.

    Args:
        font_size: Base font size for labels.
        font_family: Font family ('serif', 'sans-serif', 'monospace').
        use_latex: If True, enable LaTeX rendering (requires LaTeX installation).
    """
    plt, mpl = _get_matplotlib()

    style = {
        "font.size": font_size,
        "font.family": font_family,
        "axes.labelsize": font_size,
        "axes.titlesize": font_size + 2,
        "xtick.labelsize": font_size - 1,
        "ytick.labelsize": font_size - 1,
        "legend.fontsize": font_size - 1,
        "figure.titlesize": font_size + 2,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
    }

    if use_latex:
        style.update(
            {
                "text.usetex": True,
                "font.family": "serif",
                "font.serif": ["Computer Modern Roman"],
            }
        )

    plt.rcParams.update(style)


def plot_framework_comparison(
    results: List[Dict[str, Any]],
    metric: str = "mean",
    error_metric: str = "std",
    title: Optional[str] = None,
    ylabel: Optional[str] = None,
    figsize: Tuple[float, float] = (6, 4),
    show_values: bool = True,
    sort_by_value: bool = True,
) -> Any:
    """Create a bar chart comparing frameworks.

    Args:
        results: List of dicts with 'framework' and stats keys.
            Example: [{"framework": "mahout", "mean": 0.5, "std": 0.1}, ...]
        metric: Key for the bar heights (default: "mean").
        error_metric: Key for error bars (default: "std"). Set to None for no errors.
        title: Plot title.
        ylabel: Y-axis label.
        figsize: Figure size in inches.
        show_values: If True, display values on bars.
        sort_by_value: If True, sort bars by value (ascending).

    Returns:
        matplotlib Figure object.
    """
    plt, mpl = _get_matplotlib()

    # Extract data
    frameworks = [r.get("framework", "unknown") for r in results]
    values = [r.get(metric, 0) for r in results]
    errors = [r.get(error_metric, 0) for r in results] if error_metric else None

    # Sort if requested
    if sort_by_value:
        sorted_data = sorted(zip(values, frameworks, errors or [0] * len(values)))
        values = [v for v, _, _ in sorted_data]
        frameworks = [f for _, f, _ in sorted_data]
        if errors:
            errors = [e for _, _, e in sorted_data]

    # Get colors and labels
    colors = [FRAMEWORK_COLORS.get(f, "#888888") for f in frameworks]
    labels = [FRAMEWORK_LABELS.get(f, f) for f in frameworks]

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(frameworks))
    bars = ax.bar(x, values, color=colors, edgecolor="black", linewidth=0.5)

    if errors:
        ax.errorbar(
            x,
            values,
            yerr=errors,
            fmt="none",
            color="black",
            capsize=3,
            capthick=1,
            linewidth=1,
        )

    # Add value labels on bars
    if show_values:
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.annotate(
                f"{val:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel(ylabel or f"{metric} (ms)")
    if title:
        ax.set_title(title)

    ax.set_ylim(bottom=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    return fig


def plot_scaling(
    results: List[Dict[str, Any]],
    x_param: str,
    y_metric: str = "mean",
    error_metric: str = "std",
    group_by: str = "framework",
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    figsize: Tuple[float, float] = (8, 5),
    log_x: bool = False,
    log_y: bool = False,
) -> Any:
    """Create a line plot showing scaling behavior.

    Args:
        results: List of dicts with parameter and metric values.
        x_param: Key for x-axis values (e.g., "qubits", "batch_size").
        y_metric: Key for y-axis values (default: "mean").
        error_metric: Key for error bands (default: "std").
        group_by: Key to group lines by (default: "framework").
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        figsize: Figure size in inches.
        log_x: Use log scale for x-axis.
        log_y: Use log scale for y-axis.

    Returns:
        matplotlib Figure object.
    """
    plt, mpl = _get_matplotlib()

    # Group results
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in results:
        key = r.get(group_by, "unknown")
        if key not in groups:
            groups[key] = []
        groups[key].append(r)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    for group_name, group_results in groups.items():
        # Sort by x parameter
        group_results.sort(key=lambda r: r.get(x_param, 0))

        x_vals = [r.get(x_param, 0) for r in group_results]
        y_vals = [r.get(y_metric, 0) for r in group_results]
        y_errs = [r.get(error_metric, 0) for r in group_results] if error_metric else None

        color = FRAMEWORK_COLORS.get(group_name, None)
        label = FRAMEWORK_LABELS.get(group_name, group_name)

        ax.plot(x_vals, y_vals, marker="o", label=label, color=color, linewidth=2)

        if y_errs:
            y_vals_arr = np.array(y_vals)
            y_errs_arr = np.array(y_errs)
            ax.fill_between(
                x_vals,
                y_vals_arr - y_errs_arr,
                y_vals_arr + y_errs_arr,
                alpha=0.2,
                color=color,
            )

    if log_x:
        ax.set_xscale("log", base=2)
    if log_y:
        ax.set_yscale("log")

    ax.set_xlabel(xlabel or x_param)
    ax.set_ylabel(ylabel or f"{y_metric} (ms)")
    if title:
        ax.set_title(title)

    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    return fig


def plot_speedup(
    results: List[Dict[str, Any]],
    baseline_framework: str = "pennylane",
    metric: str = "mean",
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (6, 4),
) -> Any:
    """Create a bar chart showing speedup relative to baseline.

    Args:
        results: List of dicts with 'framework' and metric values.
        baseline_framework: Framework to use as baseline (speedup = 1x).
        metric: Metric to compute speedup from.
        title: Plot title.
        figsize: Figure size in inches.

    Returns:
        matplotlib Figure object.
    """
    plt, mpl = _get_matplotlib()

    # Find baseline value
    baseline_val = None
    for r in results:
        if r.get("framework") == baseline_framework:
            baseline_val = r.get(metric, 0)
            break

    if baseline_val is None or baseline_val == 0:
        raise ValueError(f"Baseline framework '{baseline_framework}' not found or has zero value")

    # Compute speedups
    frameworks = []
    speedups = []
    for r in results:
        fw = r.get("framework", "unknown")
        val = r.get(metric, 0)
        if val > 0:
            frameworks.append(fw)
            speedups.append(baseline_val / val)

    # Sort by speedup
    sorted_data = sorted(zip(speedups, frameworks), reverse=True)
    speedups = [s for s, _ in sorted_data]
    frameworks = [f for _, f in sorted_data]

    # Get colors and labels
    colors = [FRAMEWORK_COLORS.get(f, "#888888") for f in frameworks]
    labels = [FRAMEWORK_LABELS.get(f, f) for f in frameworks]

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(frameworks))
    bars = ax.bar(x, speedups, color=colors, edgecolor="black", linewidth=0.5)

    # Add value labels
    for bar, val in zip(bars, speedups):
        height = bar.get_height()
        ax.annotate(
            f"{val:.1f}x",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    # Add baseline reference line
    ax.axhline(y=1, color="gray", linestyle="--", linewidth=1, alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel(f"Speedup vs {FRAMEWORK_LABELS.get(baseline_framework, baseline_framework)}")
    if title:
        ax.set_title(title)

    ax.set_ylim(bottom=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    return fig
