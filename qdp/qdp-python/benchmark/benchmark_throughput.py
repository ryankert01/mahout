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
DataLoader throughput benchmark across Mahout (QDP), PennyLane, and Qiskit.

The workload mirrors the `qdp-core/examples/dataloader_throughput.rs` pipeline:
- Generate batches of size `BATCH_SIZE` with deterministic vectors.
- Prefetch on the CPU side to keep the GPU fed.
- Encode vectors into amplitude states on GPU and run a tiny consumer op.

Run:
    python qdp/benchmark/benchmark_throughput.py --qubits 16 --batches 200 --batch-size 64

With statistical analysis:
    python qdp/benchmark/benchmark_throughput.py --qubits 16 --batches 50 --warmup 2 --runs 5
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch

from _qdp import QdpEngine

# Import from benchmark.core if available, otherwise use local fallbacks
try:
    from benchmark.core import (
        BenchmarkRun,
        BenchmarkStats,
        ResultsStore,
        build_sample,
        clear_gpu_caches,
        normalize_batch,
        prefetched_batches,
        sync_cuda,
    )

    USE_CORE = True
except ImportError:
    import queue
    import threading
    from typing import Generator, Optional

    USE_CORE = False

    def sync_cuda(device_id: Optional[int] = None) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize(device_id)

    def clear_gpu_caches(gc_collect: bool = True) -> None:
        import gc

        if gc_collect:
            gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def build_sample(seed: int, vector_len: int) -> np.ndarray:
        mask = np.uint64(vector_len - 1)
        scale = 1.0 / vector_len
        idx = np.arange(vector_len, dtype=np.uint64)
        mixed = (idx + np.uint64(seed)) & mask
        return mixed.astype(np.float64) * scale

    def prefetched_batches(
        total_batches: int,
        batch_size: int,
        vector_len: int,
        prefetch: int = 16,
        seed_offset: int = 0,
        normalize: bool = False,
    ) -> Generator[np.ndarray, None, None]:
        q: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=prefetch)

        def producer():
            for batch_idx in range(total_batches):
                base = batch_idx * batch_size + seed_offset
                batch = [build_sample(base + i, vector_len) for i in range(batch_size)]
                arr = np.stack(batch)
                if normalize:
                    arr = normalize_batch(arr)
                q.put(arr)
            q.put(None)

        threading.Thread(target=producer, daemon=True).start()

        while True:
            batch = q.get()
            if batch is None:
                break
            yield batch

    def normalize_batch(batch: np.ndarray, epsilon: float = 0.0) -> np.ndarray:
        norms = np.linalg.norm(batch, axis=1, keepdims=True)
        norms = (
            np.maximum(norms, epsilon)
            if epsilon > 0
            else np.where(norms == 0, 1.0, norms)
        )
        return batch / norms


BAR = "=" * 70
SEP = "-" * 70
FRAMEWORK_CHOICES = ("pennylane", "qiskit", "mahout")
FRAMEWORK_LABELS = {
    "mahout": "Mahout",
    "pennylane": "PennyLane",
    "qiskit": "Qiskit",
}

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


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""

    framework: str
    duration_s: float
    throughput: float
    vectors_processed: int
    stats: Optional[object] = None  # BenchmarkStats when using statistical mode


def parse_frameworks(raw: str) -> list[str]:
    if raw.lower() == "all":
        return list(FRAMEWORK_CHOICES)

    selected: list[str] = []
    for part in raw.split(","):
        name = part.strip().lower()
        if not name:
            continue
        if name not in FRAMEWORK_CHOICES:
            raise ValueError(
                f"Unknown framework '{name}'. Choose from: "
                f"{', '.join(FRAMEWORK_CHOICES)} or 'all'."
            )
        if name not in selected:
            selected.append(name)

    return selected if selected else list(FRAMEWORK_CHOICES)


def run_mahout(num_qubits: int, total_batches: int, batch_size: int, prefetch: int):
    try:
        engine = QdpEngine(0)
    except Exception as exc:
        print(f"[Mahout] Init failed: {exc}")
        return 0.0, 0.0

    sync_cuda()
    start = time.perf_counter()

    processed = 0
    for batch in prefetched_batches(
        total_batches, batch_size, 1 << num_qubits, prefetch
    ):
        normalized = np.ascontiguousarray(normalize_batch(batch), dtype=np.float64)
        qtensor = engine.encode(normalized, num_qubits)
        tensor = torch.from_dlpack(qtensor).abs().to(torch.float32)
        _ = tensor.sum()
        processed += normalized.shape[0]

    sync_cuda()
    duration = time.perf_counter() - start
    throughput = processed / duration if duration > 0 else 0.0
    print(f"  Total Time: {duration:.4f} s ({throughput:.1f} vectors/sec)")
    return duration, throughput


def run_pennylane(num_qubits: int, total_batches: int, batch_size: int, prefetch: int):
    if not HAS_PENNYLANE:
        print("[PennyLane] Not installed, skipping.")
        return 0.0, 0.0

    dev = qml.device("default.qubit", wires=num_qubits)

    @qml.qnode(dev, interface="torch")
    def circuit(inputs):
        qml.AmplitudeEmbedding(
            features=inputs, wires=range(num_qubits), normalize=True, pad_with=0.0
        )
        return qml.state()

    sync_cuda()
    start = time.perf_counter()
    processed = 0

    for batch in prefetched_batches(
        total_batches, batch_size, 1 << num_qubits, prefetch
    ):
        batch_cpu = torch.tensor(batch, dtype=torch.float64)
        try:
            state_cpu = circuit(batch_cpu)
        except Exception:
            state_cpu = torch.stack([circuit(x) for x in batch_cpu])
        state_gpu = state_cpu.to("cuda", dtype=torch.float32)
        _ = state_gpu.abs().sum()
        processed += len(batch_cpu)

    sync_cuda()
    duration = time.perf_counter() - start
    throughput = processed / duration if duration > 0 else 0.0
    print(f"  Total Time: {duration:.4f} s ({throughput:.1f} vectors/sec)")
    return duration, throughput


def run_qiskit(num_qubits: int, total_batches: int, batch_size: int, prefetch: int):
    if not HAS_QISKIT:
        print("[Qiskit] Not installed, skipping.")
        return 0.0, 0.0

    backend = AerSimulator(method="statevector")
    sync_cuda()
    start = time.perf_counter()
    processed = 0

    for batch in prefetched_batches(
        total_batches, batch_size, 1 << num_qubits, prefetch
    ):
        normalized = normalize_batch(batch)

        batch_states = []
        for vec_idx, vec in enumerate(normalized):
            qc = QuantumCircuit(num_qubits)
            qc.initialize(vec, range(num_qubits))
            qc.save_statevector()
            t_qc = transpile(qc, backend)
            state = backend.run(t_qc).result().get_statevector().data
            batch_states.append(state)
            processed += 1

        gpu_tensor = torch.tensor(
            np.array(batch_states), device="cuda", dtype=torch.complex64
        )
        _ = gpu_tensor.abs().sum()

    sync_cuda()
    duration = time.perf_counter() - start
    throughput = processed / duration if duration > 0 else 0.0
    print(f"\n  Total Time: {duration:.4f} s ({throughput:.1f} vectors/sec)")
    return duration, throughput


def run_with_stats(
    fn: Callable, warmup_runs: int, measurement_runs: int, **kwargs
) -> tuple[float, float, Optional[object]]:
    """Run a benchmark function with statistical analysis.

    Returns:
        Tuple of (mean_duration, mean_throughput, BenchmarkStats or None)
    """
    if not USE_CORE or measurement_runs <= 1:
        # Single run mode (legacy behavior)
        duration, throughput = fn(**kwargs)
        return duration, throughput, None

    # Statistical mode: run multiple times and collect stats
    throughputs = []
    durations = []

    # Warmup runs
    for i in range(warmup_runs):
        print(f"    Warmup run {i + 1}/{warmup_runs}...", end="\r")
        clear_gpu_caches()
        fn(**kwargs)
    print(" " * 40, end="\r")  # Clear line

    # Measurement runs
    for i in range(measurement_runs):
        print(f"    Measurement run {i + 1}/{measurement_runs}...", end="\r")
        clear_gpu_caches()
        duration, throughput = fn(**kwargs)
        durations.append(duration)
        throughputs.append(throughput)
    print(" " * 40, end="\r")  # Clear line

    stats = BenchmarkStats.from_samples(throughputs)
    return np.mean(durations), stats.mean, stats


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark DataLoader throughput across frameworks."
    )
    parser.add_argument(
        "--qubits",
        type=int,
        default=16,
        help="Number of qubits (power-of-two vector length).",
    )
    parser.add_argument(
        "--batches", type=int, default=200, help="Total batches to stream."
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Vectors per batch.")
    parser.add_argument(
        "--prefetch", type=int, default=16, help="CPU-side prefetch depth."
    )
    parser.add_argument(
        "--frameworks",
        type=str,
        default="all",
        help=(
            "Comma-separated list of frameworks to run "
            "(pennylane,qiskit,mahout) or 'all'."
        ),
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=0,
        help="Number of warmup runs before measurement (enables statistical mode).",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of measurement runs for statistics (default: 1 = legacy mode).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Directory to save results (JSON). Enables result persistence.",
    )
    parser.add_argument(
        "--plot",
        type=str,
        default=None,
        help="Path to save comparison plot (PNG/PDF). Requires matplotlib.",
    )
    args = parser.parse_args()

    try:
        frameworks = parse_frameworks(args.frameworks)
    except ValueError as exc:
        parser.error(str(exc))

    total_vectors = args.batches * args.batch_size
    vector_len = 1 << args.qubits
    stats_mode = args.runs > 1

    print(f"Generating {total_vectors} samples of {args.qubits} qubits...")
    print(f"  Batch size   : {args.batch_size}")
    print(f"  Vector length: {vector_len}")
    print(f"  Batches      : {args.batches}")
    print(f"  Prefetch     : {args.prefetch}")
    print(f"  Frameworks   : {', '.join(frameworks)}")
    if stats_mode:
        print(f"  Warmup runs  : {args.warmup}")
        print(f"  Measure runs : {args.runs}")
        if USE_CORE:
            print("  Mode         : Statistical (benchmark.core enabled)")
        else:
            print("  Mode         : Statistical (benchmark.core NOT available)")
    bytes_per_vec = vector_len * 8
    print(f"  Generated {total_vectors} samples")
    print(
        f"  PennyLane/Qiskit format: {total_vectors * bytes_per_vec / (1024 * 1024):.2f} MB"
    )
    print(f"  Mahout format: {total_vectors * bytes_per_vec / (1024 * 1024):.2f} MB")
    print()

    print(BAR)
    title = f"DATALOADER THROUGHPUT BENCHMARK: {args.qubits} Qubits, {total_vectors} Samples"
    if stats_mode:
        title += f" ({args.runs} runs)"
    print(title)
    print(BAR)

    # Store results with stats
    results: dict[str, BenchmarkResult] = {}
    bench_kwargs = {
        "num_qubits": args.qubits,
        "total_batches": args.batches,
        "batch_size": args.batch_size,
        "prefetch": args.prefetch,
    }

    if "pennylane" in frameworks:
        print()
        print("[PennyLane] Full Pipeline (DataLoader -> GPU)...")
        t, th, s = run_with_stats(run_pennylane, args.warmup, args.runs, **bench_kwargs)
        results["pennylane"] = BenchmarkResult("pennylane", t, th, total_vectors, s)

    if "qiskit" in frameworks:
        print()
        print("[Qiskit] Full Pipeline (DataLoader -> GPU)...")
        t, th, s = run_with_stats(run_qiskit, args.warmup, args.runs, **bench_kwargs)
        results["qiskit"] = BenchmarkResult("qiskit", t, th, total_vectors, s)

    if "mahout" in frameworks:
        print()
        print("[Mahout] Full Pipeline (DataLoader -> GPU)...")
        t, th, s = run_with_stats(run_mahout, args.warmup, args.runs, **bench_kwargs)
        results["mahout"] = BenchmarkResult("mahout", t, th, total_vectors, s)

    print()
    print(BAR)
    print("THROUGHPUT (Higher is Better)")
    print(f"Samples: {total_vectors}, Qubits: {args.qubits}")
    if stats_mode and USE_CORE:
        print(f"Statistics: {args.runs} runs with {args.warmup} warmup")
    print(BAR)

    # Collect and sort results (higher throughput is better)
    throughput_results = []
    for key, res in results.items():
        if res.throughput > 0:
            throughput_results.append(
                (FRAMEWORK_LABELS[key], res.throughput, res.stats)
            )

    throughput_results.sort(key=lambda x: x[1], reverse=True)

    # Display results
    if stats_mode and USE_CORE:
        # Statistical output with mean +/- std
        for name, throughput, stats in throughput_results:
            if stats:
                print(f"{name:12s} {stats.mean:10.1f} +/- {stats.std:6.1f} vectors/sec")
            else:
                print(f"{name:12s} {throughput:10.1f} vectors/sec")
    else:
        # Legacy single-value output
        for name, throughput, _ in throughput_results:
            print(f"{name:12s} {throughput:10.1f} vectors/sec")

    # Speedup calculations
    th_mahout = results.get("mahout", BenchmarkResult("", 0, 0, 0)).throughput
    th_pl = results.get("pennylane", BenchmarkResult("", 0, 0, 0)).throughput
    th_qiskit = results.get("qiskit", BenchmarkResult("", 0, 0, 0)).throughput

    if th_mahout > 0:
        print(SEP)
        if th_pl > 0:
            print(f"Speedup vs PennyLane: {th_mahout / th_pl:10.2f}x")
        if th_qiskit > 0:
            print(f"Speedup vs Qiskit:    {th_mahout / th_qiskit:10.2f}x")

    # Save results if output directory specified
    if args.output and USE_CORE:
        print()
        print(SEP)
        print("SAVING RESULTS")
        print(SEP)
        store = ResultsStore(args.output)
        config = {
            "qubits": args.qubits,
            "batches": args.batches,
            "batch_size": args.batch_size,
            "prefetch": args.prefetch,
            "warmup_runs": args.warmup,
            "measurement_runs": args.runs,
        }
        for key, res in results.items():
            if res.throughput > 0:
                run = BenchmarkRun.create(
                    benchmark_name="throughput",
                    framework=key,
                    config=config,
                    stats=res.stats if res.stats else None,
                    throughput=res.throughput,
                )
                path = store.save(run)
                print(f"  Saved: {path}")

    # Generate plot if requested
    if args.plot and USE_CORE:
        try:
            from benchmark.visualization import plot_framework_comparison

            print()
            print(SEP)
            print("GENERATING PLOT")
            print(SEP)

            # Prepare data for plotting
            plot_data = []
            for key, res in results.items():
                if res.throughput > 0:
                    entry = {"framework": key, "mean": res.throughput}
                    if res.stats:
                        entry["std"] = res.stats.std
                    plot_data.append(entry)

            if plot_data:
                fig = plot_framework_comparison(
                    plot_data,
                    title=f"DataLoader Throughput ({args.qubits} qubits)",
                    ylabel="Throughput (vectors/sec)",
                    sort_by_value=False,  # Higher is better, keep original order
                )
                fig.savefig(args.plot, dpi=300, bbox_inches="tight")
                print(f"  Saved: {args.plot}")
            else:
                print("  No data to plot")
        except ImportError:
            print("  Warning: matplotlib not available, skipping plot")


if __name__ == "__main__":
    main()
