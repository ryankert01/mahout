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
Publication-ready visualization for benchmark results.

This module provides a BenchmarkVisualizer class for creating high-quality
plots suitable for blog posts and academic papers, including bar charts,
box plots, violin plots, and comparison tables.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union

try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False


class BenchmarkVisualizer:
    """Create publication-ready benchmark visualizations."""

    def __init__(self, style: str = "seaborn-v0_8-paper"):
        """
        Initialize with publication style.

        Args:
            style: Matplotlib style name. Common options:
                   - 'seaborn-v0_8-paper' (publication quality)
                   - 'seaborn-v0_8-notebook' (presentations)
                   - 'default' (fallback if seaborn unavailable)

        Raises:
            ImportError: If matplotlib or seaborn is not installed

        Note: If specified style is unavailable, falls back to 'default'.
        """
        if not HAS_PLOTTING:
            raise ImportError(
                "matplotlib and seaborn are required for visualization. "
                "Install with: pip install matplotlib seaborn"
            )

        try:
            plt.style.use(style)
        except OSError:
            print(f"Warning: Style '{style}' not found, using 'default'")
            plt.style.use("default")

        self.colors = sns.color_palette("husl", 8)

    def plot_comparison_bars(
        self,
        results: Dict[str, Dict[str, float]],
        metric: str = "mean",
        output_path: Union[str, Path] = "benchmark_bars.png",
        title: Optional[str] = None,
        ylabel: str = "Time (ms)",
    ) -> None:
        """
        Create bar chart comparing frameworks.

        Args:
            results: Dict of {framework_name: statistics_dict}
            metric: Which metric to plot (mean, median, etc.) (default: 'mean')
            output_path: Where to save the plot
            title: Custom title (default: auto-generated)
            ylabel: Y-axis label (default: "Time (ms)")

        Example:
            >>> visualizer = BenchmarkVisualizer()
            >>> results = {
            ...     'Framework A': {'mean': 10.5, 'std': 0.2},
            ...     'Framework B': {'mean': 15.3, 'std': 0.5}
            ... }
            >>> visualizer.plot_comparison_bars(results)
        """
        import numpy as np

        frameworks = list(results.keys())
        values = [results[fw][metric] for fw in frameworks]
        errors = [results[fw]["std"] for fw in frameworks]

        fig, ax = plt.subplots(figsize=(10, 6))
        x_pos = np.arange(len(frameworks))

        bars = ax.bar(
            x_pos,
            values,
            yerr=errors,
            capsize=5,
            alpha=0.8,
            color=self.colors[: len(frameworks)],
        )

        ax.set_xlabel("Framework", fontsize=12, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
        if title is None:
            title = f"Benchmark Comparison ({metric.capitalize()})"
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(frameworks, rotation=45, ha="right")
        ax.grid(axis="y", alpha=0.3)

        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{val:.2f}",
                ha="center",
                va="bottom",
            )

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved bar chart to {output_path}")
        plt.close()

    def plot_box_comparison(
        self,
        results_raw: Dict[str, List[float]],
        output_path: Union[str, Path] = "benchmark_box.png",
        title: Optional[str] = None,
        ylabel: str = "Time (ms)",
    ) -> None:
        """
        Create box plot showing distributions.

        Args:
            results_raw: Dict of {framework_name: [list of timings]}
            output_path: Where to save the plot
            title: Custom title (default: auto-generated)
            ylabel: Y-axis label (default: "Time (ms)")

        Example:
            >>> visualizer = BenchmarkVisualizer()
            >>> results = {
            ...     'Framework A': [10.1, 10.2, 10.5, 10.3],
            ...     'Framework B': [15.1, 15.5, 15.2, 15.8]
            ... }
            >>> visualizer.plot_box_comparison(results)
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        frameworks = list(results_raw.keys())
        data = [results_raw[fw] for fw in frameworks]

        bp = ax.boxplot(
            data, labels=frameworks, patch_artist=True, showmeans=True, meanline=True
        )

        # Color the boxes
        for patch, color in zip(bp["boxes"], self.colors[: len(frameworks)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_xlabel("Framework", fontsize=12, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
        if title is None:
            title = "Benchmark Distribution Comparison (Box Plot)"
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        plt.xticks(rotation=45, ha="right")

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved box plot to {output_path}")
        plt.close()

    def plot_violin_comparison(
        self,
        results_raw: Dict[str, List[float]],
        output_path: Union[str, Path] = "benchmark_violin.png",
        title: Optional[str] = None,
        ylabel: str = "Time (ms)",
    ) -> None:
        """
        Create violin plot showing full distributions.

        Args:
            results_raw: Dict of {framework_name: [list of timings]}
            output_path: Where to save the plot
            title: Custom title (default: auto-generated)
            ylabel: Y-axis label (default: "Time (ms)")

        Example:
            >>> visualizer = BenchmarkVisualizer()
            >>> results = {
            ...     'Framework A': [10.1, 10.2, 10.5, 10.3] * 10,
            ...     'Framework B': [15.1, 15.5, 15.2, 15.8] * 10
            ... }
            >>> visualizer.plot_violin_comparison(results)
        """
        import pandas as pd

        # Prepare data for seaborn
        data_list = []
        for framework, timings in results_raw.items():
            for timing in timings:
                data_list.append({"Framework": framework, ylabel: timing})

        df = pd.DataFrame(data_list)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.violinplot(data=df, x="Framework", y=ylabel, palette=self.colors, ax=ax)

        ax.set_xlabel("Framework", fontsize=12, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
        if title is None:
            title = "Benchmark Distribution Comparison (Violin Plot)"
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        plt.xticks(rotation=45, ha="right")

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved violin plot to {output_path}")
        plt.close()

    def create_comparison_table(
        self,
        results: Dict[str, Dict[str, float]],
        output_path: Union[str, Path] = "benchmark_table.md",
        title: str = "Benchmark Results",
    ) -> None:
        """
        Create markdown table with all statistics.

        Args:
            results: Dict of {framework_name: statistics_dict}
            output_path: Where to save the table
            title: Table title

        Example:
            >>> visualizer = BenchmarkVisualizer()
            >>> results = {
            ...     'Framework A': {
            ...         'mean': 10.5, 'median': 10.4, 'std': 0.2,
            ...         'min': 10.1, 'max': 10.9, 'p95': 10.8, 'n_runs': 100
            ...     }
            ... }
            >>> visualizer.create_comparison_table(results)
        """
        frameworks = sorted(results.keys())

        # Create markdown table
        lines = []
        lines.append(f"# {title}\n")
        lines.append(
            "| Framework | Mean (ms) | Median (ms) | Std (ms) | Min (ms) | Max (ms) | P95 (ms) | Runs |"
        )
        lines.append(
            "|-----------|-----------|-------------|----------|----------|----------|----------|------|"
        )

        for fw in frameworks:
            stats = results[fw]
            lines.append(
                f"| {fw} | {stats['mean']:.2f} | {stats['median']:.2f} | "
                f"{stats['std']:.2f} | {stats['min']:.2f} | {stats['max']:.2f} | "
                f"{stats['p95']:.2f} | {stats['n_runs']} |"
            )

        table_content = "\n".join(lines)

        output_path = Path(output_path)
        output_path.write_text(table_content)

        print(f"Saved comparison table to {output_path}")
        print(table_content)

    def create_all_plots(
        self,
        results: Dict[str, Dict[str, float]],
        results_raw: Dict[str, List[float]],
        output_dir: Union[str, Path] = ".",
        prefix: str = "benchmark",
    ) -> None:
        """
        Create all visualization types in one call.

        Args:
            results: Dict of {framework_name: statistics_dict}
            results_raw: Dict of {framework_name: [list of timings]}
            output_dir: Directory to save plots (default: current directory)
            prefix: Prefix for output filenames (default: "benchmark")

        Example:
            >>> visualizer = BenchmarkVisualizer()
            >>> results = {'Framework A': {'mean': 10.5, 'std': 0.2, ...}}
            >>> results_raw = {'Framework A': [10.1, 10.2, 10.5, ...]}
            >>> visualizer.create_all_plots(results, results_raw, output_dir='./results')
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.plot_comparison_bars(results, output_path=output_dir / f"{prefix}_bars.png")
        self.plot_box_comparison(results_raw, output_path=output_dir / f"{prefix}_box.png")
        self.plot_violin_comparison(results_raw, output_path=output_dir / f"{prefix}_violin.png")
        self.create_comparison_table(results, output_path=output_dir / f"{prefix}_table.md")

        print(f"\nAll plots saved to {output_dir}/")
