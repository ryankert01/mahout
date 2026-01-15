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
FINAL END-TO-END BENCHMARK (Disk -> GPU VRAM).

Scope:
1. Disk IO: Reading Parquet file.
2. Preprocessing: L2 Normalization (CPU vs GPU).
3. Encoding: Quantum State Preparation.
4. Transfer: Moving data to GPU VRAM.
5. Consumption: 1 dummy Forward Pass to ensure data is usable.

This is the most realistic comparison for a "Cold Start" Training Epoch.
"""

import argparse
import gc
import itertools
import os
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq
import torch
import torch.nn as nn
from _qdp import QdpEngine

# Import benchmark infrastructure (optional, with fallback)
try:
    from benchmark.core import BenchmarkRun, BenchmarkStats, ResultsStore

    HAS_BENCHMARK_CORE = True
except ImportError:
    HAS_BENCHMARK_CORE = False

# Competitors
try:
    import pennylane as qml

    HAS_PENNYLANE = True
except ImportError:
    HAS_PENNYLANE = False

try:
    from qiskit import QuantumCircuit, transpile
    from qiskit_aer import AerSimulator

    HAS_QISKIT = True
except ImportError:
    HAS_QISKIT = False

# Config
DATA_FILE = "final_benchmark_data.parquet"
ARROW_FILE = "final_benchmark_data.arrow"
HIDDEN_DIM = 16
BATCH_SIZE = 64  # Small batch to stress loop overhead


def clean_cache():
    """Clear GPU cache and Python garbage collection."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


class DummyQNN(nn.Module):
    def __init__(self, n_qubits):
        super().__init__()
        self.fc = nn.Linear(1 << n_qubits, HIDDEN_DIM)

    def forward(self, x):
        return self.fc(x)


def generate_data(n_qubits, n_samples):
    for f in [DATA_FILE, ARROW_FILE]:
        if os.path.exists(f):
            os.remove(f)

    print(f"Generating {n_samples} samples of {n_qubits} qubits...")
    dim = 1 << n_qubits

    # Generate all data at once
    np.random.seed(42)
    all_data = np.random.rand(n_samples, dim).astype(np.float64)

    # Save as Parquet (List format for PennyLane/Qiskit)
    feature_vectors = [row.tolist() for row in all_data]
    table = pa.table(
        {"feature_vector": pa.array(feature_vectors, type=pa.list_(pa.float64()))}
    )
    pq.write_table(table, DATA_FILE)

    # Save as Arrow IPC (FixedSizeList format for Mahout)
    arr = pa.FixedSizeListArray.from_arrays(pa.array(all_data.flatten()), dim)
    arrow_table = pa.table({"data": arr})
    with ipc.RecordBatchFileWriter(ARROW_FILE, arrow_table.schema) as writer:
        writer.write_table(arrow_table)

    parquet_size = os.path.getsize(DATA_FILE) / (1024 * 1024)
    arrow_size = os.path.getsize(ARROW_FILE) / (1024 * 1024)
    print(f"  Generated {n_samples} samples")
    print(f"  Parquet: {parquet_size:.2f} MB, Arrow IPC: {arrow_size:.2f} MB")

    # Clean cache after data generation
    clean_cache()


# -----------------------------------------------------------
# 1. Qiskit Full Pipeline
# -----------------------------------------------------------
def run_qiskit(n_qubits, n_samples):
    if not HAS_QISKIT:
        print("\n[Qiskit] Not installed, skipping.")
        return 0.0, None

    # Clean cache before starting benchmark
    clean_cache()

    print("\n[Qiskit] Full Pipeline (Disk -> GPU)...")
    model = DummyQNN(n_qubits).cuda()
    backend = AerSimulator(method="statevector")

    torch.cuda.synchronize()
    start_time = time.perf_counter()

    # IO
    import pandas as pd

    df = pd.read_parquet(DATA_FILE)
    raw_data = np.stack(df["feature_vector"].values)
    io_time = time.perf_counter() - start_time
    print(f"  IO Time: {io_time:.4f} s")

    all_qiskit_states = []

    # Process batches
    for i in range(0, n_samples, BATCH_SIZE):
        batch = raw_data[i : i + BATCH_SIZE]

        # Normalize
        norms = np.linalg.norm(batch, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        batch = batch / norms

        # State preparation
        batch_states = []
        for vec_idx, vec in enumerate(batch):
            qc = QuantumCircuit(n_qubits)
            qc.initialize(vec, range(n_qubits))
            qc.save_statevector()
            t_qc = transpile(qc, backend)
            result = backend.run(t_qc).result().get_statevector().data
            batch_states.append(result)

            if (vec_idx + 1) % 10 == 0:
                print(f"    Processed {vec_idx + 1}/{len(batch)} vectors...", end="\r")

        # Transfer to GPU
        gpu_tensor = torch.tensor(
            np.array(batch_states), device="cuda", dtype=torch.complex64
        )
        all_qiskit_states.append(gpu_tensor)
        _ = model(gpu_tensor.abs())

    torch.cuda.synchronize()
    total_time = time.perf_counter() - start_time
    print(f"\n  Total Time: {total_time:.4f} s")

    all_qiskit_tensor = torch.cat(all_qiskit_states, dim=0)

    # Clean cache after benchmark completion
    clean_cache()

    return total_time, all_qiskit_tensor


# -----------------------------------------------------------
# 2. PennyLane Full Pipeline
# -----------------------------------------------------------
def run_pennylane(n_qubits, n_samples):
    if not HAS_PENNYLANE:
        print("\n[PennyLane] Not installed, skipping.")
        return 0.0, None

    # Clean cache before starting benchmark
    clean_cache()

    print("\n[PennyLane] Full Pipeline (Disk -> GPU)...")

    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch")
    def circuit(inputs):
        qml.AmplitudeEmbedding(
            features=inputs, wires=range(n_qubits), normalize=True, pad_with=0.0
        )
        return qml.state()

    model = DummyQNN(n_qubits).cuda()

    torch.cuda.synchronize()
    start_time = time.perf_counter()

    # IO
    import pandas as pd

    df = pd.read_parquet(DATA_FILE)
    raw_data = np.stack(df["feature_vector"].values)
    io_time = time.perf_counter() - start_time
    print(f"  IO Time: {io_time:.4f} s")

    all_pl_states = []

    # Process batches
    for i in range(0, n_samples, BATCH_SIZE):
        batch_cpu = torch.tensor(raw_data[i : i + BATCH_SIZE])

        # Execute QNode
        try:
            state_cpu = circuit(batch_cpu)
        except Exception:
            state_cpu = torch.stack([circuit(x) for x in batch_cpu])

        all_pl_states.append(state_cpu)

        # Transfer to GPU
        state_gpu = state_cpu.to("cuda", dtype=torch.float32)
        _ = model(state_gpu.abs())

    torch.cuda.synchronize()
    total_time = time.perf_counter() - start_time
    print(f"  Total Time: {total_time:.4f} s")

    # Stack all collected states
    all_pl_states_tensor = torch.cat(
        all_pl_states, dim=0
    )  # Should handle cases where last batch is smaller

    # Clean cache after benchmark completion
    clean_cache()

    return total_time, all_pl_states_tensor


# -----------------------------------------------------------
# 3. Mahout Parquet Pipeline
# -----------------------------------------------------------
def run_mahout_parquet(engine, n_qubits, n_samples):
    # Clean cache before starting benchmark
    clean_cache()

    print("\n[Mahout-Parquet] Full Pipeline (Parquet -> GPU)...")
    model = DummyQNN(n_qubits).cuda()

    torch.cuda.synchronize()
    start_time = time.perf_counter()

    # Direct Parquet to GPU pipeline
    parquet_encode_start = time.perf_counter()
    qtensor = engine.encode(DATA_FILE, n_qubits)
    parquet_encode_time = time.perf_counter() - parquet_encode_start
    print(f"  Parquet->GPU (IO+Encode): {parquet_encode_time:.4f} s")

    # Convert to torch tensor
    dlpack_start = time.perf_counter()
    gpu_batched = torch.from_dlpack(qtensor)
    dlpack_time = time.perf_counter() - dlpack_start
    print(f"  DLPack conversion: {dlpack_time:.4f} s")

    # Tensor is already 2D [n_samples, state_len]
    state_len = 1 << n_qubits
    assert gpu_batched.shape == (n_samples, state_len), (
        f"Expected shape ({n_samples}, {state_len}), got {gpu_batched.shape}"
    )

    # Convert to float for model
    reshape_start = time.perf_counter()
    gpu_all_data = gpu_batched.abs().to(torch.float32)
    reshape_time = time.perf_counter() - reshape_start
    print(f"  Convert to float32: {reshape_time:.4f} s")

    # Forward pass (data already on GPU)
    for i in range(0, n_samples, BATCH_SIZE):
        batch = gpu_all_data[i : i + BATCH_SIZE]
        _ = model(batch)

    torch.cuda.synchronize()
    total_time = time.perf_counter() - start_time
    print(f"  Total Time: {total_time:.4f} s")

    # Clean cache after benchmark completion
    clean_cache()

    return total_time, gpu_batched


# -----------------------------------------------------------
# 4. Mahout Arrow IPC Pipeline
# -----------------------------------------------------------
def run_mahout_arrow(engine, n_qubits, n_samples):
    # Clean cache before starting benchmark
    clean_cache()

    print("\n[Mahout-Arrow] Full Pipeline (Arrow IPC -> GPU)...")
    model = DummyQNN(n_qubits).cuda()

    torch.cuda.synchronize()
    start_time = time.perf_counter()

    arrow_encode_start = time.perf_counter()
    qtensor = engine.encode(ARROW_FILE, n_qubits)
    arrow_encode_time = time.perf_counter() - arrow_encode_start
    print(f"  Arrow->GPU (IO+Encode): {arrow_encode_time:.4f} s")

    dlpack_start = time.perf_counter()
    gpu_batched = torch.from_dlpack(qtensor)
    dlpack_time = time.perf_counter() - dlpack_start
    print(f"  DLPack conversion: {dlpack_time:.4f} s")

    # Tensor is already 2D [n_samples, state_len]
    state_len = 1 << n_qubits
    assert gpu_batched.shape == (n_samples, state_len), (
        f"Expected shape ({n_samples}, {state_len}), got {gpu_batched.shape}"
    )

    reshape_start = time.perf_counter()
    gpu_all_data = gpu_batched.abs().to(torch.float32)
    reshape_time = time.perf_counter() - reshape_start
    print(f"  Convert to float32: {reshape_time:.4f} s")

    for i in range(0, n_samples, BATCH_SIZE):
        batch = gpu_all_data[i : i + BATCH_SIZE]
        _ = model(batch)

    torch.cuda.synchronize()
    total_time = time.perf_counter() - start_time
    print(f"  Total Time: {total_time:.4f} s")

    # Clean cache after benchmark completion
    clean_cache()

    return total_time, gpu_batched


def compare_states(name_a, states_a, name_b, states_b):
    print("\n" + "=" * 70)
    print(f"VERIFICATION ({name_a} vs {name_b})")
    print("=" * 70)

    # Ensure both tensors are on GPU for comparison
    n_compare = min(len(states_a), len(states_b))
    tensor_a = states_a[:n_compare].cuda()
    tensor_b = states_b[:n_compare].cuda()

    # Compare Probabilities (|psi|^2)
    diff_probs = (tensor_a.abs() ** 2 - tensor_b.abs() ** 2).abs().max().item()
    print(f"Max Probability Difference: {diff_probs:.2e}")

    # Compare Raw Amplitudes
    # We compare full complex difference magnitude
    diff_amps = (tensor_a - tensor_b).abs().max().item()
    print(f"Max Amplitude Difference:   {diff_amps:.2e}")

    if diff_probs < 1e-5:
        print(">> SUCCESS: Quantum States Match!")
    else:
        print(">> FAILURE: States do not match.")


def verify_correctness(states_dict):
    # Filter out None values
    valid_states = {
        name: states for name, states in states_dict.items() if states is not None
    }

    if len(valid_states) < 2:
        return

    keys = sorted(list(valid_states.keys()))
    for name_a, name_b in itertools.combinations(keys, 2):
        compare_states(name_a, valid_states[name_a], name_b, valid_states[name_b])


def run_benchmark_with_stats(
    run_fn, name, engine, n_qubits, n_samples, warmup_runs, measurement_runs
):
    """Run a benchmark function multiple times and collect statistics."""
    all_times = []

    for run_idx in range(warmup_runs + measurement_runs):
        is_warmup = run_idx < warmup_runs
        run_type = "warmup" if is_warmup else "measure"
        print(f"  [{run_type}] Run {run_idx + 1}/{warmup_runs + measurement_runs}...")

        if engine is not None:
            elapsed, states = run_fn(engine, n_qubits, n_samples)
        else:
            elapsed, states = run_fn(n_qubits, n_samples)

        if not is_warmup:
            all_times.append(elapsed)

    # Calculate statistics
    if HAS_BENCHMARK_CORE and all_times:
        stats = BenchmarkStats.from_samples(all_times, remove_outliers=False)
        return stats, states
    elif all_times:
        # Fallback: return simple dict
        import statistics

        return {
            "mean": statistics.mean(all_times),
            "std": statistics.stdev(all_times) if len(all_times) > 1 else 0.0,
            "min": min(all_times),
            "max": max(all_times),
            "n_samples": len(all_times),
        }, states
    return None, states


def save_results(results_data, output_dir, args):
    """Save benchmark results to output directory."""
    if not HAS_BENCHMARK_CORE:
        print("Warning: benchmark.core not available, skipping result persistence")
        return

    store = ResultsStore(output_dir)

    for framework, stats_obj in results_data.items():
        if stats_obj is None:
            continue

        # Convert BenchmarkStats to dict if needed
        if hasattr(stats_obj, "to_dict"):
            stats_dict = stats_obj.to_dict()
        else:
            stats_dict = stats_obj

        run = BenchmarkRun.create(
            benchmark_name="e2e",
            framework=framework,
            config={
                "qubits": args.qubits,
                "samples": args.samples,
            },
            stats=stats_obj if hasattr(stats_obj, "to_dict") else None,
            latency_ms=stats_dict.get("mean", 0) * 1000,  # Convert to ms
            metadata={"warmup_runs": args.warmup, "measurement_runs": args.runs},
        )
        store.save(run)

    print(f"\nResults saved to: {output_dir}")


def generate_plots(results_data, output_dir, args):
    """Generate comparison plots."""
    try:
        from benchmark.visualization.plots import (
            plot_framework_comparison,
            plot_speedup,
        )
    except ImportError:
        print("Warning: benchmark.visualization not available, skipping plots")
        return

    # Prepare data for plotting
    plot_data = []
    for framework, stats_obj in results_data.items():
        if stats_obj is None:
            continue
        if hasattr(stats_obj, "mean"):
            plot_data.append(
                {"framework": framework, "mean": stats_obj.mean, "std": stats_obj.std}
            )
        elif isinstance(stats_obj, dict):
            plot_data.append(
                {
                    "framework": framework,
                    "mean": stats_obj.get("mean", 0),
                    "std": stats_obj.get("std", 0),
                }
            )

    if not plot_data:
        print("Warning: No data to plot")
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Framework comparison bar chart
    fig = plot_framework_comparison(
        plot_data,
        title=f"E2E Latency: {args.qubits}q, {args.samples} samples",
        ylabel="Total Time (s)",
    )
    comparison_path = output_path / "e2e_comparison.png"
    fig.savefig(comparison_path, dpi=150, bbox_inches="tight")
    print(f"Saved comparison plot: {comparison_path}")

    # Speedup chart (vs PennyLane if available)
    pennylane_data = [d for d in plot_data if "pennylane" in d["framework"].lower()]
    if pennylane_data:
        try:
            # Normalize framework names for speedup plot
            speedup_data = []
            for d in plot_data:
                name = d["framework"].lower().replace("-", "_")
                speedup_data.append({**d, "framework": name})
            # Find the pennylane name in normalized form
            pl_name = [
                d["framework"] for d in speedup_data if "pennylane" in d["framework"]
            ]
            if pl_name:
                fig = plot_speedup(speedup_data, baseline_framework=pl_name[0])
                speedup_path = output_path / "e2e_speedup.png"
                fig.savefig(speedup_path, dpi=150, bbox_inches="tight")
                print(f"Saved speedup plot: {speedup_path}")
        except Exception as e:
            print(f"Warning: Could not generate speedup plot: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Final End-to-End Benchmark (Disk -> GPU VRAM)"
    )
    parser.add_argument(
        "--qubits", type=int, default=16, help="Number of qubits (16 recommended)"
    )
    parser.add_argument(
        "--samples", type=int, default=200, help="Number of training samples"
    )
    parser.add_argument(
        "--frameworks",
        nargs="+",
        default=["mahout-parquet", "pennylane"],
        choices=["mahout-parquet", "mahout-arrow", "pennylane", "qiskit", "all"],
        help="Frameworks to benchmark. Use 'all' to run all available frameworks.",
    )
    # Statistical mode options
    parser.add_argument(
        "--warmup",
        type=int,
        default=0,
        help="Number of warmup runs (default: 0 for single-shot mode)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of measurement runs (default: 1 for single-shot mode)",
    )
    # Output options
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for results (enables result persistence)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate comparison plots (requires --output)",
    )
    args = parser.parse_args()

    # Expand "all" option
    if "all" in args.frameworks:
        args.frameworks = ["mahout-parquet", "mahout-arrow", "pennylane", "qiskit"]

    # Check if statistical mode is enabled
    use_stats_mode = args.warmup > 0 or args.runs > 1

    generate_data(args.qubits, args.samples)

    try:
        engine = QdpEngine(0)
    except Exception as e:
        print(f"Mahout Init Error: {e}")
        exit(1)

    # Clean cache before starting benchmarks
    clean_cache()

    print("\n" + "=" * 70)
    print(f"E2E BENCHMARK: {args.qubits} Qubits, {args.samples} Samples")
    if use_stats_mode:
        print(f"Statistical Mode: {args.warmup} warmup + {args.runs} measurement runs")
    print("=" * 70)

    # Initialize results - use stats objects when in statistical mode
    stats_results = {}  # For statistical mode
    pl_all_states = None
    mahout_parquet_all_states = None
    mahout_arrow_all_states = None
    qiskit_all_states = None

    # Run benchmarks
    if "pennylane" in args.frameworks:
        if use_stats_mode:
            stats, pl_all_states = run_benchmark_with_stats(
                run_pennylane,
                "PennyLane",
                None,
                args.qubits,
                args.samples,
                args.warmup,
                args.runs,
            )
            stats_results["pennylane"] = stats
        else:
            t_pl, pl_all_states = run_pennylane(args.qubits, args.samples)
            stats_results["pennylane"] = {"mean": t_pl, "std": 0.0}
        clean_cache()

    if "qiskit" in args.frameworks:
        if use_stats_mode:
            stats, qiskit_all_states = run_benchmark_with_stats(
                run_qiskit,
                "Qiskit",
                None,
                args.qubits,
                args.samples,
                args.warmup,
                args.runs,
            )
            stats_results["qiskit"] = stats
        else:
            t_qiskit, qiskit_all_states = run_qiskit(args.qubits, args.samples)
            stats_results["qiskit"] = {"mean": t_qiskit, "std": 0.0}
        clean_cache()

    if "mahout-parquet" in args.frameworks:
        if use_stats_mode:
            stats, mahout_parquet_all_states = run_benchmark_with_stats(
                run_mahout_parquet,
                "Mahout-Parquet",
                engine,
                args.qubits,
                args.samples,
                args.warmup,
                args.runs,
            )
            stats_results["mahout-parquet"] = stats
        else:
            t_mahout_parquet, mahout_parquet_all_states = run_mahout_parquet(
                engine, args.qubits, args.samples
            )
            stats_results["mahout-parquet"] = {"mean": t_mahout_parquet, "std": 0.0}
        clean_cache()

    if "mahout-arrow" in args.frameworks:
        if use_stats_mode:
            stats, mahout_arrow_all_states = run_benchmark_with_stats(
                run_mahout_arrow,
                "Mahout-Arrow",
                engine,
                args.qubits,
                args.samples,
                args.warmup,
                args.runs,
            )
            stats_results["mahout-arrow"] = stats
        else:
            t_mahout_arrow, mahout_arrow_all_states = run_mahout_arrow(
                engine, args.qubits, args.samples
            )
            stats_results["mahout-arrow"] = {"mean": t_mahout_arrow, "std": 0.0}
        clean_cache()

    # Print results
    print("\n" + "=" * 70)
    print("E2E LATENCY (Lower is Better)")
    print(f"Samples: {args.samples}, Qubits: {args.qubits}")
    print("=" * 70)

    # Collect results for sorting
    results = []
    for name, stats_obj in stats_results.items():
        if stats_obj is None:
            continue
        if hasattr(stats_obj, "mean"):
            mean_val = stats_obj.mean
            std_val = stats_obj.std
        else:
            mean_val = stats_obj.get("mean", 0)
            std_val = stats_obj.get("std", 0)
        if mean_val > 0:
            results.append((name, mean_val, std_val))

    results.sort(key=lambda x: x[1])

    # Print results with statistics
    if use_stats_mode:
        for name, mean_val, std_val in results:
            print(f"{name:16s} {mean_val:10.4f} +/- {std_val:.4f} s")
    else:
        for name, mean_val, _ in results:
            print(f"{name:16s} {mean_val:10.4f} s")

    print("-" * 70)

    # Calculate speedup vs competitors
    mahout_results = [r for r in results if "mahout" in r[0].lower()]
    t_mahout_best = min(r[1] for r in mahout_results) if mahout_results else 0

    if t_mahout_best > 0:
        pl_result = next((r for r in results if "pennylane" in r[0].lower()), None)
        qiskit_result = next((r for r in results if "qiskit" in r[0].lower()), None)

        if pl_result:
            print(f"Speedup vs PennyLane: {pl_result[1] / t_mahout_best:10.2f}x")
        if qiskit_result:
            print(f"Speedup vs Qiskit:    {qiskit_result[1] / t_mahout_best:10.2f}x")

    # Save results if output specified
    if args.output:
        save_results(stats_results, args.output, args)

    # Generate plots if requested
    if args.plot:
        if args.output:
            generate_plots(stats_results, args.output, args)
        else:
            print("Warning: --plot requires --output to be specified")

    # Run Verification after benchmarks
    verify_correctness(
        {
            "Mahout-Parquet": mahout_parquet_all_states,
            "Mahout-Arrow": mahout_arrow_all_states,
            "PennyLane": pl_all_states,
            "Qiskit": qiskit_all_states,
        }
    )
