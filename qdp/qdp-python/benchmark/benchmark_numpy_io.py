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
NumPy format I/O + Encoding benchmark: Mahout vs PennyLane

Compares the performance of loading quantum state data from NumPy .npy files
and encoding them on GPU between Mahout QDP and PennyLane.

Workflow:
1. Generate NumPy arrays with quantum state vectors
2. Save to .npy file
3. Load from file and encode on GPU
4. Measure total throughput (I/O + encoding)

Run:
    python qdp/benchmark/benchmark_numpy_io.py --qubits 10 --samples 1000
"""

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from mahout_qdp import QdpEngine

BAR = "=" * 70
SEP = "-" * 70

try:
    import pennylane as qml

    HAS_PENNYLANE = True
except ImportError:
    HAS_PENNYLANE = False

# Try to import benchmark_utils for statistical mode
try:
    from benchmark_utils import (
        benchmark_with_cuda_events,
        clear_all_caches,
        compute_statistics,
        format_statistics,
        BenchmarkVisualizer,
    )
    HAS_BENCHMARK_UTILS = True
except ImportError:
    HAS_BENCHMARK_UTILS = False
    print("Note: benchmark_utils not available. Statistical mode disabled.")


def generate_test_data(
    num_samples: int, sample_size: int, seed: int = 42
) -> np.ndarray:
    """Generate deterministic test data."""
    rng = np.random.RandomState(seed)
    data = rng.randn(num_samples, sample_size).astype(np.float64)
    # Normalize each sample
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return data / norms


def run_mahout_numpy(num_qubits: int, num_samples: int, npy_path: str):
    """Benchmark Mahout with NumPy file I/O."""
    print("\n[Mahout + NumPy] Loading and encoding...")

    try:
        engine = QdpEngine(0)
    except Exception as exc:
        print(f"  Init failed: {exc}")
        return 0.0, 0.0, 0.0

    # Measure file I/O + encoding together
    torch.cuda.synchronize()
    start_total = time.perf_counter()

    try:
        # Use the NumPy reader API
        qtensor = engine.encode_from_numpy(npy_path, num_qubits, "amplitude")
        tensor = torch.utils.dlpack.from_dlpack(qtensor)

        # Small computation to ensure GPU has processed the data
        _ = tensor.abs().sum()

        torch.cuda.synchronize()
        duration_total = time.perf_counter() - start_total

        throughput = num_samples / duration_total if duration_total > 0 else 0.0

        print(f"  Total Time (I/O + Encode): {duration_total:.4f} s")
        print(f"  Throughput: {throughput:.1f} samples/sec")
        print(f"  Average per sample: {duration_total / num_samples * 1000:.2f} ms")

        return duration_total, throughput, duration_total / num_samples

    except Exception as exc:
        print(f"  Error: {exc}")
        return 0.0, 0.0, 0.0


def run_pennylane_numpy(num_qubits: int, num_samples: int, npy_path: str):
    """Benchmark PennyLane with NumPy file I/O."""
    if not HAS_PENNYLANE:
        print("\n[PennyLane + NumPy] Not installed, skipping.")
        return 0.0, 0.0, 0.0

    print("\n[PennyLane + NumPy] Loading and encoding...")

    dev = qml.device("default.qubit", wires=num_qubits)

    @qml.qnode(dev, interface="torch")
    def circuit(inputs):
        qml.AmplitudeEmbedding(
            features=inputs, wires=range(num_qubits), normalize=True, pad_with=0.0
        )
        return qml.state()

    torch.cuda.synchronize()
    start_total = time.perf_counter()

    try:
        # Load NumPy file
        data = np.load(npy_path)

        # Process each sample
        states = []
        for i in range(len(data)):
            sample = torch.tensor(data[i], dtype=torch.float64)
            state = circuit(sample)
            states.append(state)

        # Move to GPU
        states_gpu = torch.stack(states).to("cuda", dtype=torch.complex64)
        _ = states_gpu.abs().sum()

        torch.cuda.synchronize()
        duration_total = time.perf_counter() - start_total

        throughput = num_samples / duration_total if duration_total > 0 else 0.0

        print(f"  Total Time (I/O + Encode): {duration_total:.4f} s")
        print(f"  Throughput: {throughput:.1f} samples/sec")
        print(f"  Average per sample: {duration_total / num_samples * 1000:.2f} ms")

        return duration_total, throughput, duration_total / num_samples

    except Exception as exc:
        print(f"  Error: {exc}")
        return 0.0, 0.0, 0.0


def run_framework_statistical(
    framework_name: str, benchmark_func, warmup_iters: int, repeat: int
):
    """
    Run a framework benchmark in statistical mode with warmup and multiple iterations.
    
    Args:
        framework_name: Name of the framework being benchmarked
        benchmark_func: Function that runs one iteration and returns (duration, throughput, avg_per_sample)
        warmup_iters: Number of warmup iterations
        repeat: Number of measurement iterations
    
    Returns:
        Tuple of (mean_duration, mean_throughput, durations_list, throughputs_list)
    """
    print(f"\n[{framework_name}] Statistical mode (warmup={warmup_iters}, repeat={repeat})")
    
    # Warmup phase
    print(f"  Warmup ({warmup_iters} iterations)...")
    for i in range(warmup_iters):
        _ = benchmark_func()
        clear_all_caches()
    
    # Measurement phase
    print(f"  Measuring ({repeat} iterations)...")
    durations = []
    throughputs = []
    
    for i in range(repeat):
        duration, throughput, _ = benchmark_func()
        if duration > 0:
            durations.append(duration * 1000)  # Convert to ms
            throughputs.append(throughput)
        clear_all_caches()
    
    if not durations:
        print(f"  Error: No successful measurements")
        return 0.0, 0.0, [], []
    
    # Compute and display statistics
    duration_stats = compute_statistics(durations)
    throughput_stats = compute_statistics(throughputs)
    
    print(f"\n  Duration Statistics (ms):")
    print(format_statistics(duration_stats))
    
    print(f"\n  Throughput Statistics (samples/sec):")
    print(format_statistics(throughput_stats))
    
    # Return mean values for compatibility with standard mode
    mean_duration = duration_stats['mean'] / 1000  # Convert back to seconds
    mean_throughput = throughput_stats['mean']
    
    return mean_duration, mean_throughput, durations, throughputs


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark NumPy I/O + Encoding: Mahout vs PennyLane"
    )
    parser.add_argument(
        "--qubits",
        type=int,
        default=10,
        help="Number of qubits (vector length = 2^qubits)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1000,
        help="Number of samples to generate",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save .npy file (default: temp file)",
    )
    parser.add_argument(
        "--frameworks",
        type=str,
        default="all",
        help="Comma-separated list: mahout,pennylane or 'all'",
    )
    parser.add_argument(
        "--statistical",
        action="store_true",
        help="Enable statistical mode with warmup and multiple runs",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Number of warmup iterations (default: 3)",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=10,
        help="Number of measurement iterations (default: 10)",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate publication-ready plots (requires --statistical)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./benchmark_results",
        help="Directory to save visualization plots (default: ./benchmark_results)",
    )
    args = parser.parse_args()
    
    # Validate arguments
    if args.visualize and not args.statistical:
        print("Error: --visualize requires --statistical mode")
        sys.exit(1)
    
    if args.statistical and not HAS_BENCHMARK_UTILS:
        print("Error: --statistical mode requires benchmark_utils package")
        sys.exit(1)

    # Parse frameworks
    if args.frameworks.lower() == "all":
        frameworks = ["mahout", "pennylane"]
    else:
        frameworks = [f.strip().lower() for f in args.frameworks.split(",")]

    num_qubits = args.qubits
    num_samples = args.samples
    sample_size = 1 << num_qubits  # 2^qubits

    print(BAR)
    print("NUMPY I/O + ENCODING BENCHMARK")
    print(BAR)
    print(f"Qubits: {num_qubits}")
    print(f"Sample size: {sample_size} elements")
    print(f"Number of samples: {num_samples}")
    print(f"Total data: {num_samples * sample_size * 8 / (1024**2):.2f} MB")
    print(f"Frameworks: {', '.join(frameworks)}")
    if args.statistical:
        print(f"Mode: Statistical (warmup={args.warmup}, repeat={args.repeat})")
    else:
        print(f"Mode: Standard (single run)")

    # Generate test data
    print("\nGenerating test data...")
    data = generate_test_data(num_samples, sample_size)

    # Save to NumPy file
    if args.output:
        npy_path = args.output
    else:
        fd, npy_path = tempfile.mkstemp(suffix=".npy")
        os.close(fd)

    print(f"Saving to {npy_path}...")
    np.save(npy_path, data)
    file_size_mb = os.path.getsize(npy_path) / (1024**2)
    print(f"File size: {file_size_mb:.2f} MB")

    # Run benchmarks
    results = {}
    
    # For visualization: collect all timings and stats
    duration_timings_raw = {}  # Raw duration timings for each framework
    throughput_timings_raw = {}  # Raw throughput timings for each framework
    duration_stats_dict = {}   # Duration statistics for each framework
    throughput_stats_dict = {}   # Throughput statistics for each framework

    if args.statistical:
        # Statistical mode
        if "mahout" in frameworks:
            benchmark_func = lambda: run_mahout_numpy(num_qubits, num_samples, npy_path)
            t_total, throughput, durations, throughputs = run_framework_statistical(
                "Mahout", benchmark_func, args.warmup, args.repeat
            )
            if throughput > 0:
                results["Mahout"] = {
                    "time": t_total,
                    "throughput": throughput,
                    "avg_per_sample": t_total / num_samples,
                }
                duration_timings_raw["Mahout"] = durations
                throughput_timings_raw["Mahout"] = throughputs
                duration_stats_dict["Mahout"] = compute_statistics(durations)
                throughput_stats_dict["Mahout"] = compute_statistics(throughputs)

        if "pennylane" in frameworks:
            benchmark_func = lambda: run_pennylane_numpy(num_qubits, num_samples, npy_path)
            t_total, throughput, durations, throughputs = run_framework_statistical(
                "PennyLane", benchmark_func, args.warmup, args.repeat
            )
            if throughput > 0:
                results["PennyLane"] = {
                    "time": t_total,
                    "throughput": throughput,
                    "avg_per_sample": t_total / num_samples,
                }
                duration_timings_raw["PennyLane"] = durations
                throughput_timings_raw["PennyLane"] = throughputs
                duration_stats_dict["PennyLane"] = compute_statistics(durations)
                throughput_stats_dict["PennyLane"] = compute_statistics(throughputs)
    else:
        # Standard mode (original behavior)
        if "mahout" in frameworks:
            t_total, throughput, avg_per_sample = run_mahout_numpy(
                num_qubits, num_samples, npy_path
            )
            if throughput > 0:
                results["Mahout"] = {
                    "time": t_total,
                    "throughput": throughput,
                    "avg_per_sample": avg_per_sample,
                }

        if "pennylane" in frameworks:
            t_total, throughput, avg_per_sample = run_pennylane_numpy(
                num_qubits, num_samples, npy_path
            )
            if throughput > 0:
                results["PennyLane"] = {
                    "time": t_total,
                    "throughput": throughput,
                    "avg_per_sample": avg_per_sample,
                }

    # Print summary
    if results:
        print("\n" + BAR)
        print("SUMMARY")
        print(BAR)
        print(
            f"{'Framework':<15} {'Time (s)':<12} {'Throughput':<20} {'Avg/Sample':<15}"
        )
        print(SEP)

        sorted_results = sorted(
            results.items(), key=lambda x: x[1]["throughput"], reverse=True
        )

        for name, metrics in sorted_results:
            print(
                f"{name:<15} "
                f"{metrics['time']:<12.4f} "
                f"{metrics['throughput']:<20.1f} "
                f"{metrics['avg_per_sample'] * 1000:<15.2f}"
            )

        if len(results) > 1:
            print("\n" + SEP)
            print("SPEEDUP COMPARISON")
            print(SEP)

            if "Mahout" in results and "PennyLane" in results:
                speedup = (
                    results["Mahout"]["throughput"] / results["PennyLane"]["throughput"]
                )
                print(f"Mahout vs PennyLane: {speedup:.2f}x")

                time_ratio = results["PennyLane"]["time"] / results["Mahout"]["time"]
                print(f"Time reduction: {time_ratio:.2f}x faster")
    
    # Generate visualizations if requested
    if args.visualize and args.statistical and duration_timings_raw:
        print()
        print(BAR)
        print("GENERATING VISUALIZATIONS")
        print(BAR)
        
        try:
            visualizer = BenchmarkVisualizer()
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate duration plots
            print("\nGenerating duration visualizations...")
            visualizer.create_all_plots(
                results=duration_stats_dict,
                results_raw=duration_timings_raw,
                output_dir=output_dir,
                prefix=f"numpy_duration_q{num_qubits}_s{num_samples}"
            )
            
            # Generate throughput plots
            print("Generating throughput visualizations...")
            visualizer.create_all_plots(
                results=throughput_stats_dict,
                results_raw=throughput_timings_raw,
                output_dir=output_dir,
                prefix=f"numpy_throughput_q{num_qubits}_s{num_samples}"
            )
            
            print(f"\nVisualization complete! Files saved to: {output_dir}")
            print("\nDuration plots:")
            print(f"  - Bar chart: {output_dir}/numpy_duration_q{num_qubits}_s{num_samples}_bars.png")
            print(f"  - Box plot: {output_dir}/numpy_duration_q{num_qubits}_s{num_samples}_box.png")
            print(f"  - Violin plot: {output_dir}/numpy_duration_q{num_qubits}_s{num_samples}_violin.png")
            print(f"  - Statistics table: {output_dir}/numpy_duration_q{num_qubits}_s{num_samples}_table.md")
            print("\nThroughput plots:")
            print(f"  - Bar chart: {output_dir}/numpy_throughput_q{num_qubits}_s{num_samples}_bars.png")
            print(f"  - Box plot: {output_dir}/numpy_throughput_q{num_qubits}_s{num_samples}_box.png")
            print(f"  - Violin plot: {output_dir}/numpy_throughput_q{num_qubits}_s{num_samples}_violin.png")
            print(f"  - Statistics table: {output_dir}/numpy_throughput_q{num_qubits}_s{num_samples}_table.md")
            
        except Exception as e:
            print(f"\nWarning: Visualization generation failed: {e}")
            print("Benchmark results are still valid.")

    # Cleanup
    if not args.output:
        os.remove(npy_path)
        print(f"\nCleaned up temporary file: {npy_path}")

    print("\n" + BAR)
    print("BENCHMARK COMPLETE")
    print(BAR)


if __name__ == "__main__":
    main()
