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

"""Benchmark results persistence and storage.

This module provides data structures and utilities for storing, loading,
and querying benchmark results.

Example:
    >>> from benchmark.core.results import BenchmarkRun, ResultsStore
    >>> run = BenchmarkRun.create(
    ...     benchmark_name="latency",
    ...     framework="mahout",
    ...     config={"qubits": 16, "batch_size": 64},
    ...     stats=my_stats,
    ... )
    >>> store = ResultsStore("results/")
    >>> store.save(run)
    >>> all_runs = store.load()
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .statistics import BenchmarkStats


def _get_git_hash() -> str:
    """Get current git commit hash, or 'unknown' if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def _get_hardware_info() -> Dict[str, Any]:
    """Get basic hardware information."""
    info: Dict[str, Any] = {}

    try:
        import torch

        if torch.cuda.is_available():
            info["cuda_available"] = True
            info["cuda_device_count"] = torch.cuda.device_count()
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            info["cuda_total_memory_gb"] = props.total_memory / (1024**3)
        else:
            info["cuda_available"] = False
    except ImportError:
        info["cuda_available"] = False

    return info


@dataclass
class BenchmarkRun:
    """A single benchmark run with metadata and results.

    Attributes:
        benchmark_name: Name of the benchmark (e.g., "latency", "throughput").
        framework: Framework being benchmarked (e.g., "mahout", "pennylane").
        timestamp: ISO format timestamp when the run started.
        git_hash: Git commit hash at time of run.
        config: Configuration parameters used for this run.
        stats: Statistical results from the benchmark.
        hardware: Hardware information (GPU, etc.).
        metadata: Additional metadata (optional).
    """

    benchmark_name: str
    framework: str
    timestamp: str
    git_hash: str
    config: Dict[str, Any]
    stats: Dict[str, float]
    hardware: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        benchmark_name: str,
        framework: str,
        config: Dict[str, Any],
        stats: Optional[BenchmarkStats] = None,
        latency_ms: Optional[float] = None,
        throughput: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> BenchmarkRun:
        """Create a new benchmark run with auto-populated metadata.

        Args:
            benchmark_name: Name of the benchmark.
            framework: Framework being benchmarked.
            config: Configuration parameters.
            stats: BenchmarkStats object (optional).
            latency_ms: Single latency value if no stats (optional).
            throughput: Throughput value (optional).
            metadata: Additional metadata (optional).

        Returns:
            New BenchmarkRun instance.
        """
        stats_dict: Dict[str, float] = {}

        if stats is not None:
            stats_dict = stats.to_dict()
        if latency_ms is not None:
            stats_dict["latency_ms"] = latency_ms
        if throughput is not None:
            stats_dict["throughput"] = throughput

        return cls(
            benchmark_name=benchmark_name,
            framework=framework,
            timestamp=datetime.now().isoformat(),
            git_hash=_get_git_hash(),
            config=config,
            stats=stats_dict,
            hardware=_get_hardware_info(),
            metadata=metadata or {},
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> BenchmarkRun:
        """Create from dictionary."""
        return cls(**data)

    def to_flat_dict(self) -> Dict[str, Any]:
        """Flatten nested dicts for CSV/DataFrame compatibility.

        Returns dict with keys like 'config.qubits', 'stats.mean', etc.
        """
        flat: Dict[str, Any] = {
            "benchmark_name": self.benchmark_name,
            "framework": self.framework,
            "timestamp": self.timestamp,
            "git_hash": self.git_hash,
        }

        for key, value in self.config.items():
            flat[f"config.{key}"] = value

        for key, value in self.stats.items():
            flat[f"stats.{key}"] = value

        for key, value in self.hardware.items():
            flat[f"hardware.{key}"] = value

        for key, value in self.metadata.items():
            flat[f"metadata.{key}"] = value

        return flat


class ResultsStore:
    """Storage for benchmark results with JSON and CSV support.

    Results are stored as JSON files (one per run) and can be exported
    to CSV for analysis.

    Example:
        >>> store = ResultsStore("benchmark_results/")
        >>> store.save(run)
        >>> runs = store.load()
        >>> store.to_csv("results.csv")
    """

    def __init__(self, directory: str | Path):
        """Initialize results store.

        Args:
            directory: Directory to store results. Created if doesn't exist.
        """
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def save(self, run: BenchmarkRun) -> Path:
        """Save a benchmark run to JSON file.

        Args:
            run: BenchmarkRun to save.

        Returns:
            Path to the saved file.
        """
        # Create filename from timestamp and framework
        safe_timestamp = run.timestamp.replace(":", "-").replace(".", "-")
        filename = f"{run.benchmark_name}_{run.framework}_{safe_timestamp}.json"
        filepath = self.directory / filename

        with open(filepath, "w") as f:
            json.dump(run.to_dict(), f, indent=2)

        return filepath

    def load(
        self,
        benchmark_name: Optional[str] = None,
        framework: Optional[str] = None,
    ) -> List[BenchmarkRun]:
        """Load benchmark runs from storage.

        Args:
            benchmark_name: Filter by benchmark name (optional).
            framework: Filter by framework (optional).

        Returns:
            List of BenchmarkRun objects.
        """
        runs: List[BenchmarkRun] = []

        for filepath in sorted(self.directory.glob("*.json")):
            try:
                with open(filepath) as f:
                    data = json.load(f)
                run = BenchmarkRun.from_dict(data)

                # Apply filters
                if benchmark_name and run.benchmark_name != benchmark_name:
                    continue
                if framework and run.framework != framework:
                    continue

                runs.append(run)
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                print(f"Warning: Could not load {filepath}: {e}")

        return runs

    def to_csv(self, output_path: str | Path) -> Path:
        """Export all results to CSV.

        Args:
            output_path: Path for CSV output.

        Returns:
            Path to the saved CSV file.
        """
        runs = self.load()
        if not runs:
            raise ValueError("No results to export")

        output_path = Path(output_path)

        # Collect all unique keys
        all_keys: set[str] = set()
        flat_runs = []
        for run in runs:
            flat = run.to_flat_dict()
            all_keys.update(flat.keys())
            flat_runs.append(flat)

        # Sort keys for consistent ordering
        sorted_keys = sorted(all_keys)

        # Write CSV
        with open(output_path, "w") as f:
            # Header
            f.write(",".join(sorted_keys) + "\n")

            # Data rows
            for flat in flat_runs:
                row = [str(flat.get(k, "")) for k in sorted_keys]
                f.write(",".join(row) + "\n")

        return output_path

    def to_dataframe(self):
        """Convert results to pandas DataFrame.

        Returns:
            pandas DataFrame with all results.

        Raises:
            ImportError: If pandas is not installed.
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for DataFrame export")

        runs = self.load()
        if not runs:
            return pd.DataFrame()

        flat_runs = [run.to_flat_dict() for run in runs]
        return pd.DataFrame(flat_runs)

    def clear(self) -> int:
        """Remove all stored results.

        Returns:
            Number of files removed.
        """
        count = 0
        for filepath in self.directory.glob("*.json"):
            filepath.unlink()
            count += 1
        return count
