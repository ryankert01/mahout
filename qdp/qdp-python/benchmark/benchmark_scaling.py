#!/usr/bin/env python3
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
Scaling Benchmark - Generate framework comparison graphs.

This script benchmarks throughput across different configurations and
generates publication-quality comparison plots.

Usage:
    # Default: throughput vs samples for Mahout and PennyLane
    python benchmark_scaling.py

    # Custom sample counts
    python benchmark_scaling.py --samples 100 500 1000 2000 5000

    # Custom qubits
    python benchmark_scaling.py --qubits 12 --samples 100 500 1000

    # X-axis as qubits instead of samples
    python benchmark_scaling.py --x-axis qubits --qubits 8 10 12 14 16

    # Save plot to file
    python benchmark_scaling.py --output scaling_plot.png

    # Use specific GPU
    python benchmark_scaling.py --gpu 2
"""

import argparse
import gc
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

# Default output directory
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "results"


def setup_gpu(device_id: int):
    """Set up GPU environment before importing CUDA libraries."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    print(f"Set CUDA_VISIBLE_DEVICES={device_id}")


# Parse --gpu argument early, before importing torch/pennylane
def get_gpu_arg():
    """Extract --gpu argument before full argument parsing."""
    for i, arg in enumerate(sys.argv):
        if arg == "--gpu" and i + 1 < len(sys.argv):
            return int(sys.argv[i + 1])
    return 0  # default


# Set GPU before importing CUDA libraries
_GPU_ID = get_gpu_arg()
setup_gpu(_GPU_ID)

# Now import CUDA-dependent libraries
# Check for torch
try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Check for PennyLane
try:
    import pennylane as qml

    HAS_PENNYLANE = True
except ImportError:
    HAS_PENNYLANE = False

# Check for Mahout QDP
try:
    from _qdp import QdpEngine

    HAS_MAHOUT = True
except ImportError:
    HAS_MAHOUT = False

# Check for Qiskit
try:
    from qiskit import QuantumCircuit, transpile
    from qiskit_aer import AerSimulator

    HAS_QISKIT = True
except ImportError:
    HAS_QISKIT = False


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""

    framework: str
    x_value: float
    throughput: float
    latency_ms: float
    config: Dict[str, Any]


def clean_cache(device_id: int = 0):
    """Clear GPU cache and garbage collect."""
    gc.collect()
    if HAS_TORCH and torch.cuda.is_available():
        # Always use device 0 since CUDA_VISIBLE_DEVICES remaps the selected GPU
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def generate_data(n_samples: int, n_qubits: int, seed: int = 42) -> np.ndarray:
    """Generate normalized random data for benchmarking."""
    dim = 1 << n_qubits
    rng = np.random.default_rng(seed)
    data = rng.random((n_samples, dim), dtype=np.float64)
    # L2 normalize
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return data / norms


def benchmark_mahout(
    data: np.ndarray, n_qubits: int, batch_size: int = 64, warmup: int = 1, runs: int = 3,
    device_id: int = 0
) -> BenchmarkResult:
    """Benchmark Mahout QDP throughput."""
    if not HAS_MAHOUT:
        return None

    import tempfile

    import pyarrow as pa
    import pyarrow.parquet as pq

    n_samples = len(data)
    # Always use device 0 since CUDA_VISIBLE_DEVICES remaps the GPU
    engine = QdpEngine(0)

    # Write data to temporary parquet file
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        temp_path = f.name

    # Convert to parquet format
    feature_vectors = [row.tolist() for row in data]
    table = pa.table(
        {"feature_vector": pa.array(feature_vectors, type=pa.list_(pa.float64()))}
    )
    pq.write_table(table, temp_path)

    times = []
    for i in range(warmup + runs):
        clean_cache(0)  # Use device 0 since CUDA_VISIBLE_DEVICES remaps

        if HAS_TORCH:
            torch.cuda.synchronize(0)
        start = time.perf_counter()

        # Encode all data
        qtensor = engine.encode(temp_path, n_qubits)
        if HAS_TORCH:
            gpu_tensor = torch.from_dlpack(qtensor)
            torch.cuda.synchronize(0)

        elapsed = time.perf_counter() - start

        if i >= warmup:
            times.append(elapsed)

    # Cleanup
    Path(temp_path).unlink()

    mean_time = np.mean(times)
    throughput = n_samples / mean_time

    return BenchmarkResult(
        framework="Mahout",
        x_value=n_samples,
        throughput=throughput,
        latency_ms=mean_time * 1000 / n_samples,
        config={"n_qubits": n_qubits, "n_samples": n_samples},
    )


def benchmark_pennylane(
    data: np.ndarray, n_qubits: int, batch_size: int = 64, warmup: int = 1, runs: int = 3
) -> BenchmarkResult:
    """Benchmark PennyLane throughput."""
    if not HAS_PENNYLANE:
        return None

    n_samples = len(data)
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="numpy")
    def circuit(inputs):
        qml.AmplitudeEmbedding(features=inputs, wires=range(n_qubits), normalize=False)
        return qml.state()

    times = []
    for i in range(warmup + runs):
        clean_cache()

        start = time.perf_counter()

        # Process all samples
        states = []
        for vec in data:
            state = circuit(vec)
            states.append(state)

        elapsed = time.perf_counter() - start

        if i >= warmup:
            times.append(elapsed)

    mean_time = np.mean(times)
    throughput = n_samples / mean_time

    return BenchmarkResult(
        framework="PennyLane",
        x_value=n_samples,
        throughput=throughput,
        latency_ms=mean_time * 1000 / n_samples,
        config={"n_qubits": n_qubits, "n_samples": n_samples},
    )


def benchmark_qiskit(
    data: np.ndarray, n_qubits: int, batch_size: int = 64, warmup: int = 1, runs: int = 3
) -> BenchmarkResult:
    """Benchmark Qiskit throughput (warning: very slow!)."""
    if not HAS_QISKIT:
        return None

    n_samples = len(data)
    backend = AerSimulator(method="statevector")

    times = []
    for i in range(warmup + runs):
        clean_cache(0)

        start = time.perf_counter()

        # Process all samples
        states = []
        for vec in data:
            qc = QuantumCircuit(n_qubits)
            qc.initialize(vec, range(n_qubits))
            qc.save_statevector()
            t_qc = transpile(qc, backend)
            result = backend.run(t_qc).result().get_statevector().data
            states.append(result)

        elapsed = time.perf_counter() - start

        if i >= warmup:
            times.append(elapsed)

    mean_time = np.mean(times)
    throughput = n_samples / mean_time

    return BenchmarkResult(
        framework="Qiskit",
        x_value=n_samples,
        throughput=throughput,
        latency_ms=mean_time * 1000 / n_samples,
        config={"n_qubits": n_qubits, "n_samples": n_samples},
    )


def run_scaling_benchmark(
    x_axis: str,
    x_values: List[int],
    n_qubits: int,
    n_samples: int,
    frameworks: List[str],
    warmup: int,
    runs: int,
) -> List[BenchmarkResult]:
    """Run benchmark across different x-axis values."""
    results = []

    for x_val in x_values:
        # Determine actual parameters based on x-axis
        if x_axis == "samples":
            actual_samples = x_val
            actual_qubits = n_qubits
        elif x_axis == "qubits":
            actual_samples = n_samples
            actual_qubits = x_val
        elif x_axis == "batch_size":
            actual_samples = n_samples
            actual_qubits = n_qubits
        else:
            raise ValueError(f"Unknown x-axis: {x_axis}")

        print(f"\n{'='*60}")
        print(f"Benchmarking: {x_axis}={x_val}, qubits={actual_qubits}, samples={actual_samples}")
        print("=" * 60)

        # Generate data for this configuration
        data = generate_data(actual_samples, actual_qubits)

        for fw in frameworks:
            print(f"\n  Running {fw}...")

            if fw.lower() == "mahout":
                result = benchmark_mahout(
                    data, actual_qubits, warmup=warmup, runs=runs
                )
            elif fw.lower() == "pennylane":
                result = benchmark_pennylane(
                    data, actual_qubits, warmup=warmup, runs=runs
                )
            elif fw.lower() == "qiskit":
                print("    (Qiskit is slow, please wait...)")
                result = benchmark_qiskit(
                    data, actual_qubits, warmup=warmup, runs=runs
                )
            else:
                print(f"  Unknown framework: {fw}")
                continue

            if result:
                result.x_value = x_val
                results.append(result)
                print(f"    Throughput: {result.throughput:.2f} samples/sec")
                print(f"    Latency: {result.latency_ms:.4f} ms/sample")
            else:
                print(f"    {fw} not available, skipping")

    return results


def save_results_csv(
    results: List[BenchmarkResult],
    output_path: str,
):
    """Save benchmark results to CSV file."""
    import csv

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        # Header
        writer.writerow([
            "framework", "x_value", "throughput_samples_per_sec",
            "latency_ms_per_sample", "n_qubits", "n_samples"
        ])
        # Data
        for r in results:
            writer.writerow([
                r.framework,
                r.x_value,
                f"{r.throughput:.2f}",
                f"{r.latency_ms:.4f}",
                r.config.get("n_qubits", ""),
                r.config.get("n_samples", ""),
            ])

    print(f"CSV saved to: {output_path}")


def plot_results(
    results: List[BenchmarkResult],
    x_axis: str,
    y_axis: str,
    title: Optional[str] = None,
    output_path: Optional[str] = None,
    log_x: bool = False,
    log_y: bool = False,
    figsize: tuple = (10, 6),
):
    """Generate comparison plot from results."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot")
        return

    # Group results by framework
    frameworks = {}
    for r in results:
        if r.framework not in frameworks:
            frameworks[r.framework] = {"x": [], "y": []}
        frameworks[r.framework]["x"].append(r.x_value)
        if y_axis == "throughput":
            frameworks[r.framework]["y"].append(r.throughput)
        else:
            frameworks[r.framework]["y"].append(r.latency_ms)

    # Color palette
    colors = {
        "Mahout": "#E63946",  # Red
        "PennyLane": "#457B9D",  # Blue
        "Qiskit": "#2A9D8F",  # Teal
    }

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    for fw_name, data in frameworks.items():
        color = colors.get(fw_name, "#666666")
        ax.plot(
            data["x"],
            data["y"],
            marker="o",
            linewidth=2,
            markersize=8,
            label=fw_name,
            color=color,
        )

    # Labels
    x_labels = {
        "samples": "Total Samples",
        "qubits": "Number of Qubits",
        "batch_size": "Batch Size",
    }
    y_labels = {
        "throughput": "Throughput (samples/sec)",
        "latency": "Latency (ms/sample)",
    }

    ax.set_xlabel(x_labels.get(x_axis, x_axis), fontsize=12)
    ax.set_ylabel(y_labels.get(y_axis, y_axis), fontsize=12)

    if title:
        ax.set_title(title, fontsize=14, fontweight="bold")
    else:
        ax.set_title(f"Framework Comparison: {y_labels.get(y_axis, y_axis)}", fontsize=14)

    # Scale
    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")

    # Grid and legend
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc="best")

    # Tight layout
    plt.tight_layout()

    # Save or show
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"\nPlot saved to: {output_path}")
    else:
        plt.show()

    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Scaling Benchmark - Generate framework comparison graphs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # X-axis configuration
    parser.add_argument(
        "--x-axis",
        choices=["samples", "qubits", "batch_size"],
        default="samples",
        help="Parameter for X-axis (default: samples)",
    )

    # Sample counts
    parser.add_argument(
        "--samples",
        nargs="+",
        type=int,
        default=[100, 250, 500, 1000, 2000],
        help="Sample counts to benchmark (default: 100 250 500 1000 2000)",
    )

    # Qubit configuration
    parser.add_argument(
        "--qubits",
        nargs="+",
        type=int,
        default=[12],
        help="Qubit counts (default: 12). If --x-axis=qubits, these are the x-values.",
    )

    # Y-axis configuration
    parser.add_argument(
        "--y-axis",
        choices=["throughput", "latency"],
        default="throughput",
        help="Metric for Y-axis (default: throughput)",
    )

    # Framework selection
    parser.add_argument(
        "--frameworks",
        nargs="+",
        default=["mahout", "pennylane", "qiskit"],
        help="Frameworks to compare (default: mahout pennylane qiskit)",
    )

    # Benchmark settings
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Warmup runs per configuration (default: 1)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Measurement runs per configuration (default: 3)",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Output directory for results (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save files, show plot interactively instead",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Custom title for the plot",
    )

    # Scale options
    parser.add_argument("--log-x", action="store_true", help="Use log scale for X-axis")
    parser.add_argument("--log-y", action="store_true", help="Use log scale for Y-axis")

    # GPU selection
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU device ID to use (default: 0)",
    )

    args = parser.parse_args()

    # Determine x-values based on x-axis choice
    if args.x_axis == "samples":
        x_values = args.samples
        n_qubits = args.qubits[0]
        n_samples = args.samples[0]  # Not used when x-axis is samples
    elif args.x_axis == "qubits":
        x_values = args.qubits
        n_qubits = args.qubits[0]  # Not used when x-axis is qubits
        n_samples = args.samples[0]
    else:
        x_values = args.samples
        n_qubits = args.qubits[0]
        n_samples = args.samples[0]

    # Generate timestamp for output files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Setup output directory
    output_dir = Path(args.output_dir)
    if not args.no_save:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Print configuration
    print("=" * 60)
    print("SCALING BENCHMARK")
    print("=" * 60)
    print(f"X-axis:     {args.x_axis} = {x_values}")
    print(f"Y-axis:     {args.y_axis}")
    print(f"Qubits:     {n_qubits}" if args.x_axis != "qubits" else f"Qubits:     {args.qubits}")
    print(f"Frameworks: {args.frameworks}")
    print(f"GPU:        {args.gpu}")
    print(f"Warmup:     {args.warmup}, Runs: {args.runs}")
    print(f"Output:     {output_dir}" if not args.no_save else "Output:     (interactive)")
    print("=" * 60)

    # Check framework availability
    if "mahout" in [f.lower() for f in args.frameworks] and not HAS_MAHOUT:
        print("Warning: Mahout not available")
    if "pennylane" in [f.lower() for f in args.frameworks] and not HAS_PENNYLANE:
        print("Warning: PennyLane not available")
    if "qiskit" in [f.lower() for f in args.frameworks] and not HAS_QISKIT:
        print("Warning: Qiskit not available")
    if "qiskit" in [f.lower() for f in args.frameworks]:
        print("Note: Qiskit is significantly slower than other frameworks")

    # Run benchmarks (GPU already selected via CUDA_VISIBLE_DEVICES)
    results = run_scaling_benchmark(
        x_axis=args.x_axis,
        x_values=x_values,
        n_qubits=n_qubits,
        n_samples=n_samples,
        frameworks=args.frameworks,
        warmup=args.warmup,
        runs=args.runs,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Framework':<12} {'X-Value':<10} {'Throughput (s/s)':<18} {'Latency (ms)':<12}")
    print("-" * 60)
    for r in results:
        print(f"{r.framework:<12} {r.x_value:<10} {r.throughput:<18.2f} {r.latency_ms:<12.4f}")

    # Generate outputs
    if results:
        # Build filename base with timestamp
        filename_base = f"scaling_{args.x_axis}_{args.y_axis}_{timestamp}"

        if args.no_save:
            # Interactive mode - just show plot
            plot_results(
                results,
                x_axis=args.x_axis,
                y_axis=args.y_axis,
                title=args.title,
                output_path=None,
                log_x=args.log_x,
                log_y=args.log_y,
            )
        else:
            # Save CSV
            csv_path = output_dir / f"{filename_base}.csv"
            save_results_csv(results, str(csv_path))

            # Save plot
            png_path = output_dir / f"{filename_base}.png"
            plot_results(
                results,
                x_axis=args.x_axis,
                y_axis=args.y_axis,
                title=args.title,
                output_path=str(png_path),
                log_x=args.log_x,
                log_y=args.log_y,
            )

            print("\n" + "=" * 60)
            print("OUTPUT FILES")
            print("=" * 60)
            print(f"  CSV:  {csv_path}")
            print(f"  Plot: {png_path}")


if __name__ == "__main__":
    main()
