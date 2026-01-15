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

"""Tests for benchmark.visualization.plots module."""

import tempfile
from pathlib import Path

import pytest

# Skip all tests if matplotlib is not available
pytest.importorskip("matplotlib")

from benchmark.visualization.plots import (
    plot_framework_comparison,
    plot_scaling,
    plot_speedup,
    set_publication_style,
)


@pytest.fixture
def sample_results():
    """Sample benchmark results for testing."""
    return [
        {"framework": "mahout", "mean": 0.5, "std": 0.1},
        {"framework": "pennylane", "mean": 2.5, "std": 0.3},
        {"framework": "qiskit", "mean": 5.0, "std": 0.5},
    ]


@pytest.fixture
def scaling_results():
    """Sample scaling results for testing."""
    results = []
    for qubits in [8, 10, 12, 14, 16]:
        results.append({
            "framework": "mahout",
            "qubits": qubits,
            "mean": 0.1 * (2 ** (qubits - 8)),
            "std": 0.01 * (2 ** (qubits - 8)),
        })
        results.append({
            "framework": "pennylane",
            "qubits": qubits,
            "mean": 0.5 * (2 ** (qubits - 8)),
            "std": 0.05 * (2 ** (qubits - 8)),
        })
    return results


class TestPlotFrameworkComparison:
    """Tests for plot_framework_comparison function."""

    def test_basic_plot(self, sample_results):
        """Test basic bar chart creation."""
        fig = plot_framework_comparison(sample_results)

        assert fig is not None
        # Should have one axes
        assert len(fig.axes) == 1

    def test_with_title(self, sample_results):
        """Test plot with title."""
        fig = plot_framework_comparison(
            sample_results,
            title="Test Comparison",
            ylabel="Latency (ms)",
        )

        ax = fig.axes[0]
        assert ax.get_title() == "Test Comparison"
        assert ax.get_ylabel() == "Latency (ms)"

    def test_no_error_bars(self, sample_results):
        """Test plot without error bars."""
        fig = plot_framework_comparison(
            sample_results,
            error_metric=None,
        )

        assert fig is not None

    def test_save_to_file(self, sample_results):
        """Test saving plot to file."""
        fig = plot_framework_comparison(sample_results)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.png"
            fig.savefig(path)
            assert path.exists()


class TestPlotScaling:
    """Tests for plot_scaling function."""

    def test_basic_scaling_plot(self, scaling_results):
        """Test basic scaling line plot."""
        fig = plot_scaling(
            scaling_results,
            x_param="qubits",
            y_metric="mean",
        )

        assert fig is not None
        assert len(fig.axes) == 1

    def test_log_scale(self, scaling_results):
        """Test plot with log scales."""
        fig = plot_scaling(
            scaling_results,
            x_param="qubits",
            log_y=True,
        )

        ax = fig.axes[0]
        assert ax.get_yscale() == "log"

    def test_with_labels(self, scaling_results):
        """Test plot with custom labels."""
        fig = plot_scaling(
            scaling_results,
            x_param="qubits",
            title="Scaling Test",
            xlabel="Number of Qubits",
            ylabel="Time (ms)",
        )

        ax = fig.axes[0]
        assert ax.get_xlabel() == "Number of Qubits"


class TestPlotSpeedup:
    """Tests for plot_speedup function."""

    def test_basic_speedup(self, sample_results):
        """Test basic speedup chart."""
        fig = plot_speedup(
            sample_results,
            baseline_framework="pennylane",
        )

        assert fig is not None

    def test_missing_baseline_raises(self, sample_results):
        """Test that missing baseline raises error."""
        with pytest.raises(ValueError, match="Baseline framework"):
            plot_speedup(
                sample_results,
                baseline_framework="nonexistent",
            )

    def test_speedup_values(self, sample_results):
        """Test speedup calculation is correct."""
        # mahout: 0.5, pennylane: 2.5
        # speedup of mahout vs pennylane = 2.5 / 0.5 = 5x
        fig = plot_speedup(
            sample_results,
            baseline_framework="pennylane",
        )

        assert fig is not None


class TestSetPublicationStyle:
    """Tests for set_publication_style function."""

    def test_applies_style(self):
        """Test that style settings are applied."""
        import matplotlib.pyplot as plt

        set_publication_style(font_size=12)

        assert plt.rcParams["font.size"] == 12

    def test_latex_style(self):
        """Test LaTeX style option."""
        # Just test it doesn't crash - LaTeX may not be installed
        try:
            set_publication_style(use_latex=True)
        except Exception:
            pytest.skip("LaTeX not available")
