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

"""Tests for benchmark.core.results module."""

import json
import tempfile
from pathlib import Path

import pytest

from benchmark.core.results import BenchmarkRun, ResultsStore
from benchmark.core.statistics import BenchmarkStats


class TestBenchmarkRun:
    """Tests for BenchmarkRun dataclass."""

    def test_create_basic(self):
        """Test basic run creation."""
        run = BenchmarkRun.create(
            benchmark_name="latency",
            framework="mahout",
            config={"qubits": 16, "batch_size": 64},
            latency_ms=0.5,
        )

        assert run.benchmark_name == "latency"
        assert run.framework == "mahout"
        assert run.config["qubits"] == 16
        assert run.stats["latency_ms"] == 0.5
        assert run.timestamp  # Should be auto-populated
        assert run.git_hash  # Should be auto-populated

    def test_create_with_stats(self):
        """Test run creation with BenchmarkStats."""
        stats = BenchmarkStats.from_samples([10.0, 11.0, 10.5, 10.8, 11.2])
        run = BenchmarkRun.create(
            benchmark_name="throughput",
            framework="pennylane",
            config={"samples": 1000},
            stats=stats,
        )

        assert "mean" in run.stats
        assert "std" in run.stats
        assert "p95" in run.stats
        assert run.stats["n_samples"] == 5

    def test_to_dict(self):
        """Test dictionary conversion."""
        run = BenchmarkRun.create(
            benchmark_name="test",
            framework="test_fw",
            config={"key": "value"},
        )

        d = run.to_dict()
        assert d["benchmark_name"] == "test"
        assert d["framework"] == "test_fw"
        assert d["config"]["key"] == "value"

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "benchmark_name": "test",
            "framework": "fw",
            "timestamp": "2024-01-01T00:00:00",
            "git_hash": "abc123",
            "config": {"a": 1},
            "stats": {"mean": 5.0},
            "hardware": {},
            "metadata": {},
        }

        run = BenchmarkRun.from_dict(data)
        assert run.benchmark_name == "test"
        assert run.stats["mean"] == 5.0

    def test_to_flat_dict(self):
        """Test flat dictionary conversion."""
        run = BenchmarkRun.create(
            benchmark_name="test",
            framework="fw",
            config={"qubits": 16},
            latency_ms=1.5,
        )

        flat = run.to_flat_dict()
        assert flat["benchmark_name"] == "test"
        assert flat["config.qubits"] == 16
        assert flat["stats.latency_ms"] == 1.5

    def test_hardware_info_populated(self):
        """Test that hardware info is auto-populated."""
        run = BenchmarkRun.create(
            benchmark_name="test",
            framework="fw",
            config={},
        )

        # Should have at least cuda_available key
        assert "cuda_available" in run.hardware


class TestResultsStore:
    """Tests for ResultsStore class."""

    def test_save_and_load(self):
        """Test saving and loading a run."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ResultsStore(tmpdir)

            run = BenchmarkRun.create(
                benchmark_name="latency",
                framework="mahout",
                config={"qubits": 16},
                latency_ms=0.5,
            )

            # Save
            path = store.save(run)
            assert path.exists()
            assert path.suffix == ".json"

            # Load
            runs = store.load()
            assert len(runs) == 1
            assert runs[0].benchmark_name == "latency"
            assert runs[0].framework == "mahout"

    def test_load_with_filters(self):
        """Test loading with filters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ResultsStore(tmpdir)

            # Save multiple runs
            for fw in ["mahout", "pennylane", "qiskit"]:
                run = BenchmarkRun.create(
                    benchmark_name="latency",
                    framework=fw,
                    config={},
                )
                store.save(run)

            # Filter by framework
            mahout_runs = store.load(framework="mahout")
            assert len(mahout_runs) == 1
            assert mahout_runs[0].framework == "mahout"

            # All runs
            all_runs = store.load()
            assert len(all_runs) == 3

    def test_to_csv(self):
        """Test CSV export."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ResultsStore(tmpdir)

            run = BenchmarkRun.create(
                benchmark_name="test",
                framework="fw",
                config={"qubits": 16},
                latency_ms=1.0,
            )
            store.save(run)

            csv_path = Path(tmpdir) / "results.csv"
            store.to_csv(csv_path)

            assert csv_path.exists()

            # Check CSV content
            content = csv_path.read_text()
            assert "benchmark_name" in content
            assert "config.qubits" in content

    def test_to_csv_empty_raises(self):
        """Test that to_csv raises on empty store."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ResultsStore(tmpdir)

            with pytest.raises(ValueError, match="No results"):
                store.to_csv("output.csv")

    def test_clear(self):
        """Test clearing results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ResultsStore(tmpdir)

            # Save some runs
            for i in range(3):
                run = BenchmarkRun.create(
                    benchmark_name="test",
                    framework="fw",
                    config={"i": i},
                )
                store.save(run)

            assert len(store.load()) == 3

            # Clear
            count = store.clear()
            assert count == 3
            assert len(store.load()) == 0

    def test_creates_directory(self):
        """Test that store creates directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            new_dir = Path(tmpdir) / "nested" / "results"
            store = ResultsStore(new_dir)

            assert new_dir.exists()

    def test_invalid_json_skipped(self):
        """Test that invalid JSON files are skipped with warning."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ResultsStore(tmpdir)

            # Create a valid run
            run = BenchmarkRun.create(
                benchmark_name="test",
                framework="fw",
                config={},
            )
            store.save(run)

            # Create an invalid JSON file
            invalid_path = Path(tmpdir) / "invalid.json"
            invalid_path.write_text("not valid json {{{")

            # Load should skip invalid file
            runs = store.load()
            assert len(runs) == 1
