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
Configuration loading and management for benchmarks.

This module provides utilities to load and validate benchmark configuration
from YAML files, enabling reproducible benchmark runs.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

try:
    import yaml

    HAS_YAML = True
except ImportError:
    HAS_YAML = False


@dataclass
class FairnessConfig:
    """Configuration for fairness settings."""

    warmup_iters: int = 3
    repeat_runs: int = 20
    clear_cache_between_runs: bool = False
    use_cuda_events: bool = True


@dataclass
class StatisticsConfig:
    """Configuration for statistical analysis."""

    collect_percentiles: List[int] = field(default_factory=lambda: [25, 50, 75, 90, 95, 99])
    outlier_detection: str = "iqr"
    outlier_threshold: float = 1.5


@dataclass
class VisualizationConfig:
    """Configuration for visualization output."""

    output_dir: str = "./benchmark_results"
    plot_formats: List[str] = field(default_factory=lambda: ["png", "pdf"])
    dpi: int = 300
    style: str = "seaborn-v0_8-paper"
    plots: List[str] = field(
        default_factory=lambda: ["bar_chart", "box_plot", "violin_plot", "comparison_table"]
    )


@dataclass
class WorkloadConfig:
    """Configuration for a specific benchmark workload."""

    qubits: int = 16
    samples: int = 200
    frameworks: List[str] = field(default_factory=lambda: ["mahout-parquet", "pennylane"])


@dataclass
class BenchmarkConfig:
    """Complete benchmark configuration."""

    fairness: FairnessConfig = field(default_factory=FairnessConfig)
    statistics: StatisticsConfig = field(default_factory=StatisticsConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    workloads: Dict[str, WorkloadConfig] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "BenchmarkConfig":
        """
        Load configuration from YAML file.

        Args:
            path: Path to YAML configuration file

        Returns:
            BenchmarkConfig instance

        Raises:
            ImportError: If PyYAML is not installed
            FileNotFoundError: If config file doesn't exist
            ValueError: If config file has invalid format

        Example:
            >>> config = BenchmarkConfig.from_yaml('benchmark_config.yaml')
            >>> print(config.fairness.warmup_iters)
            3
        """
        if not HAS_YAML:
            raise ImportError("PyYAML is required to load config from file. Install with: pip install pyyaml")

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with path.open("r") as f:
            data = yaml.safe_load(f)

        if data is None:
            raise ValueError(f"Empty config file: {path}")

        # Parse fairness config
        fairness_data = data.get("fairness", {})
        fairness = FairnessConfig(**fairness_data)

        # Parse statistics config
        stats_data = data.get("statistics", {})
        statistics = StatisticsConfig(**stats_data)

        # Parse visualization config
        viz_data = data.get("visualization", {})
        visualization = VisualizationConfig(**viz_data)

        # Parse workloads
        workloads = {}
        workloads_data = data.get("workloads", {})
        for name, wl_data in workloads_data.items():
            workloads[name] = WorkloadConfig(**wl_data)

        return cls(
            fairness=fairness,
            statistics=statistics,
            visualization=visualization,
            workloads=workloads,
        )

    def to_yaml(self, path: Union[str, Path]) -> None:
        """
        Save configuration to YAML file.

        Args:
            path: Path where to save the configuration

        Raises:
            ImportError: If PyYAML is not installed

        Example:
            >>> config = BenchmarkConfig()
            >>> config.to_yaml('my_config.yaml')
        """
        if not HAS_YAML:
            raise ImportError("PyYAML is required to save config to file. Install with: pip install pyyaml")

        path = Path(path)

        # Convert to dictionary
        data = {
            "fairness": {
                "warmup_iters": self.fairness.warmup_iters,
                "repeat_runs": self.fairness.repeat_runs,
                "clear_cache_between_runs": self.fairness.clear_cache_between_runs,
                "use_cuda_events": self.fairness.use_cuda_events,
            },
            "statistics": {
                "collect_percentiles": self.statistics.collect_percentiles,
                "outlier_detection": self.statistics.outlier_detection,
                "outlier_threshold": self.statistics.outlier_threshold,
            },
            "visualization": {
                "output_dir": self.visualization.output_dir,
                "plot_formats": self.visualization.plot_formats,
                "dpi": self.visualization.dpi,
                "style": self.visualization.style,
                "plots": self.visualization.plots,
            },
            "workloads": {
                name: {
                    "qubits": wl.qubits,
                    "samples": wl.samples,
                    "frameworks": wl.frameworks,
                }
                for name, wl in self.workloads.items()
            },
        }

        with path.open("w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

        print(f"Saved configuration to {path}")

    @classmethod
    def default(cls) -> "BenchmarkConfig":
        """
        Create default configuration with example workloads.

        Returns:
            BenchmarkConfig with default settings

        Example:
            >>> config = BenchmarkConfig.default()
            >>> config.to_yaml('default_config.yaml')
        """
        config = cls()

        # Add example workloads
        config.workloads["e2e"] = WorkloadConfig(
            qubits=16,
            samples=200,
            frameworks=["mahout-parquet", "pennylane"],
        )

        config.workloads["throughput"] = WorkloadConfig(
            qubits=16,
            samples=12800,  # 200 batches * 64 batch_size
            frameworks=["mahout", "pennylane"],
        )

        return config


def load_config(path: Optional[Union[str, Path]] = None) -> BenchmarkConfig:
    """
    Load benchmark configuration from file or return default.

    This is a convenience function that loads config from a file if provided,
    otherwise returns default configuration.

    Args:
        path: Optional path to YAML configuration file

    Returns:
        BenchmarkConfig instance

    Example:
        >>> config = load_config()  # Returns default
        >>> config = load_config('my_config.yaml')  # Loads from file
    """
    if path is None:
        return BenchmarkConfig.default()
    return BenchmarkConfig.from_yaml(path)
