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
"""

import argparse
import queue
import threading
import time
import sys
from pathlib import Path

import numpy as np
import torch

from mahout_qdp import QdpEngine

# Add benchmark_utils to path
benchmark_dir = Path(__file__).parent
sys.path.insert(0, str(benchmark_dir))

# Import benchmark utilities (optional for statistical mode)
try:
    from benchmark_utils import (
        warmup,
        clear_all_caches,
        benchmark_with_cuda_events,
        compute_statistics,
        format_statistics,
    )
    HAS_BENCHMARK_UTILS = True
except ImportError:
    HAS_BENCHMARK_UTILS = False

BAR = "=" * 70
SEP = "-" * 70
FRAMEWORK_CHOICES = ("pennylane", "qiskit", "mahout")

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


def build_sample(seed: int, vector_len: int) -> np.ndarray:
    mask = np.uint64(vector_len - 1)
    scale = 1.0 / vector_len
    idx = np.arange(vector_len, dtype=np.uint64)
    mixed = (idx + np.uint64(seed)) & mask
    return mixed.astype(np.float64) * scale


def prefetched_batches(
    total_batches: int, batch_size: int, vector_len: int, prefetch: int
):
    q: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=prefetch)

    def producer():
        for batch_idx in range(total_batches):
            base = batch_idx * batch_size
            batch = [build_sample(base + i, vector_len) for i in range(batch_size)]
            q.put(np.stack(batch))
        q.put(None)

    threading.Thread(target=producer, daemon=True).start()

    while True:
        batch = q.get()
        if batch is None:
            break
        yield batch


def normalize_batch(batch: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(batch, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return batch / norms


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

    torch.cuda.synchronize()
    start = time.perf_counter()

    processed = 0
    for batch in prefetched_batches(
        total_batches, batch_size, 1 << num_qubits, prefetch
    ):
        normalized = normalize_batch(batch)
        for sample in normalized:
            qtensor = engine.encode(sample.tolist(), num_qubits, "amplitude")
            tensor = torch.utils.dlpack.from_dlpack(qtensor).abs().to(torch.float32)
            _ = tensor.sum()
            processed += 1

    torch.cuda.synchronize()
    duration = time.perf_counter() - start
    throughput = processed / duration if duration > 0 else 0.0
    print(f"  IO + Encode Time: {duration:.4f} s")
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

    torch.cuda.synchronize()
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

    torch.cuda.synchronize()
    duration = time.perf_counter() - start
    throughput = processed / duration if duration > 0 else 0.0
    print(f"  Total Time: {duration:.4f} s ({throughput:.1f} vectors/sec)")
    return duration, throughput


def run_qiskit(num_qubits: int, total_batches: int, batch_size: int, prefetch: int):
    if not HAS_QISKIT:
        print("[Qiskit] Not installed, skipping.")
        return 0.0, 0.0

    backend = AerSimulator(method="statevector")
    torch.cuda.synchronize()
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

    torch.cuda.synchronize()
    duration = time.perf_counter() - start
    throughput = processed / duration if duration > 0 else 0.0
    print(f"\n  Total Time: {duration:.4f} s ({throughput:.1f} vectors/sec)")
    return duration, throughput


# -----------------------------------------------------------
# Statistical Mode Wrappers (Phase 2)
# -----------------------------------------------------------
def run_framework_statistical_throughput(framework_name, framework_func, warmup_iters=2, repeat=10):
    """
    Run a throughput benchmark in statistical mode with multiple iterations.
    
    Args:
        framework_name: Name of the framework for display
        framework_func: Function to benchmark (must take no arguments and return (duration, throughput))
        warmup_iters: Number of warmup iterations
        repeat: Number of measurement iterations
    
    Returns:
        Tuple of (mean_duration, mean_throughput, all_durations, all_throughputs)
    """
    if not HAS_BENCHMARK_UTILS:
        print(f"\n[{framework_name}] Warning: benchmark_utils not available, running single iteration")
        duration, throughput = framework_func()
        return duration, throughput, [duration], [throughput]
    
    print(f"\n[{framework_name}] Statistical Mode: Warmup ({warmup_iters} iters)...")
    
    # Warmup phase
    for i in range(warmup_iters):
        clear_all_caches()
        _ = framework_func()
        if i == 0:
            print(f"  Warmup iteration {i+1}/{warmup_iters} complete")
    
    print(f"[{framework_name}] Running {repeat} measurement iterations...")
    clear_all_caches()
    
    # Measurement phase
    durations = []
    throughputs = []
    
    for i in range(repeat):
        clear_all_caches()
        
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        duration, throughput = framework_func()
        end_event.record()
        
        torch.cuda.synchronize()
        measured_duration = start_event.elapsed_time(end_event) / 1000.0  # Convert ms to seconds
        durations.append(measured_duration)
        throughputs.append(throughput)
        
        if (i + 1) % 5 == 0:
            print(f"  Completed {i+1}/{repeat} iterations")
    
    # Compute statistics
    duration_stats = compute_statistics([d * 1000 for d in durations])  # Convert to ms for stats
    throughput_stats = compute_statistics(throughputs)
    
    print(f"\n[{framework_name}] Statistical Results:")
    print("Duration Statistics:")
    print(format_statistics(duration_stats, unit="ms"))
    print("\nThroughput Statistics:")
    print(format_statistics(throughput_stats, unit="vectors/sec"))
    
    mean_duration = duration_stats['mean'] / 1000.0  # Convert back to seconds
    mean_throughput = throughput_stats['mean']
    return mean_duration, mean_throughput, durations, throughputs


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
        "--statistical",
        action="store_true",
        help="Enable statistical mode with warmup and multiple runs (requires benchmark_utils)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="Number of warmup iterations in statistical mode (default: 2 for throughput)",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=10,
        help="Number of measurement iterations in statistical mode (default: 10)",
    )
    args = parser.parse_args()
    
    # Check if statistical mode is enabled but benchmark_utils is not available
    if args.statistical and not HAS_BENCHMARK_UTILS:
        print("Error: --statistical mode requires benchmark_utils package.")
        print("Make sure benchmark_utils is installed or accessible.")
        exit(1)

    try:
        frameworks = parse_frameworks(args.frameworks)
    except ValueError as exc:
        parser.error(str(exc))

    total_vectors = args.batches * args.batch_size
    vector_len = 1 << args.qubits

    print(f"Generating {total_vectors} samples of {args.qubits} qubits...")
    print(f"  Batch size   : {args.batch_size}")
    print(f"  Vector length: {vector_len}")
    print(f"  Batches      : {args.batches}")
    print(f"  Prefetch     : {args.prefetch}")
    print(f"  Frameworks   : {', '.join(frameworks)}")
    bytes_per_vec = vector_len * 8
    print(f"  Generated {total_vectors} samples")
    print(
        f"  PennyLane/Qiskit format: {total_vectors * bytes_per_vec / (1024 * 1024):.2f} MB"
    )
    print(f"  Mahout format: {total_vectors * bytes_per_vec / (1024 * 1024):.2f} MB")
    print()

    print(BAR)
    mode_str = " (Statistical Mode)" if args.statistical else ""
    print(
        f"DATALOADER THROUGHPUT BENCHMARK: {args.qubits} Qubits, {total_vectors} Samples{mode_str}"
    )
    if args.statistical:
        print(f"Warmup: {args.warmup} iterations, Repeat: {args.repeat} measurements")
    print(BAR)

    t_pl = th_pl = t_qiskit = th_qiskit = t_mahout = th_mahout = 0.0

    if args.statistical:
        # Statistical mode: run with warmup and multiple iterations
        if "pennylane" in frameworks:
            print()
            print("[PennyLane] Full Pipeline (DataLoader -> GPU)...")
            t_pl, th_pl, _, _ = run_framework_statistical_throughput(
                "PennyLane",
                lambda: run_pennylane(args.qubits, args.batches, args.batch_size, args.prefetch),
                warmup_iters=args.warmup,
                repeat=args.repeat
            )

        if "qiskit" in frameworks:
            print()
            print("[Qiskit] Full Pipeline (DataLoader -> GPU)...")
            t_qiskit, th_qiskit, _, _ = run_framework_statistical_throughput(
                "Qiskit",
                lambda: run_qiskit(args.qubits, args.batches, args.batch_size, args.prefetch),
                warmup_iters=args.warmup,
                repeat=args.repeat
            )

        if "mahout" in frameworks:
            print()
            print("[Mahout] Full Pipeline (DataLoader -> GPU)...")
            t_mahout, th_mahout, _, _ = run_framework_statistical_throughput(
                "Mahout",
                lambda: run_mahout(args.qubits, args.batches, args.batch_size, args.prefetch),
                warmup_iters=args.warmup,
                repeat=args.repeat
            )
    else:
        # Standard mode: single run
        if "pennylane" in frameworks:
            print()
            print("[PennyLane] Full Pipeline (DataLoader -> GPU)...")
            t_pl, th_pl = run_pennylane(
                args.qubits, args.batches, args.batch_size, args.prefetch
            )

        if "qiskit" in frameworks:
            print()
            print("[Qiskit] Full Pipeline (DataLoader -> GPU)...")
            t_qiskit, th_qiskit = run_qiskit(
                args.qubits, args.batches, args.batch_size, args.prefetch
            )

        if "mahout" in frameworks:
            print()
            print("[Mahout] Full Pipeline (DataLoader -> GPU)...")
            t_mahout, th_mahout = run_mahout(
                args.qubits, args.batches, args.batch_size, args.prefetch
            )

    print()
    print(BAR)
    print("THROUGHPUT (Higher is Better)")
    print(f"Samples: {total_vectors}, Qubits: {args.qubits}")
    print(BAR)

    throughput_results = []
    if th_pl > 0:
        throughput_results.append(("PennyLane", th_pl))
    if th_qiskit > 0:
        throughput_results.append(("Qiskit", th_qiskit))
    if th_mahout > 0:
        throughput_results.append(("Mahout", th_mahout))

    throughput_results.sort(key=lambda x: x[1], reverse=True)

    for name, tput in throughput_results:
        print(f"{name:12s} {tput:10.1f} vectors/sec")

    if t_mahout > 0:
        print(SEP)
        if t_pl > 0:
            print(f"Speedup vs PennyLane: {th_mahout / th_pl:10.2f}x")
        if t_qiskit > 0:
            print(f"Speedup vs Qiskit:    {th_mahout / th_qiskit:10.2f}x")


if __name__ == "__main__":
    main()
