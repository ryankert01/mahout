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
QDP + PennyLane End-to-End Benchmark

Compares two paths for quantum ML workflows:
1. QDP Path: QDP encoding (GPU) → inject state → PennyLane circuit
2. PennyLane Path: AmplitudeEmbedding → PennyLane circuit

This is a fair apples-to-apples comparison where both paths use
PennyLane's backend for the quantum circuit execution.

Usage:
    python benchmark_qdp_pennylane.py --samples 100 500 1000 --qubits 8
    python benchmark_qdp_pennylane.py --dataset mnist_full --samples 1000 5000
"""

import argparse
import gc
import os
import sys
import time
from pathlib import Path
from typing import Tuple

import numpy as np

# Add parent directory to path
_BENCHMARK_DIR = Path(__file__).parent
if str(_BENCHMARK_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_BENCHMARK_DIR.parent))

# Setup GPU before importing CUDA libraries
def get_gpu_arg():
    for i, arg in enumerate(sys.argv):
        if arg == "--gpu" and i + 1 < len(sys.argv):
            return int(sys.argv[i + 1])
    return 0

_GPU_ID = get_gpu_arg()
os.environ["CUDA_VISIBLE_DEVICES"] = str(_GPU_ID)

import torch
import pennylane as qml

# Check for QDP
try:
    from _qdp import QdpEngine
    HAS_QDP = True
except ImportError:
    HAS_QDP = False

# Check for datasets
try:
    from benchmark.datasets import get_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False


def clean_cache():
    """Clear GPU cache and garbage collect."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def generate_data(n_samples: int, n_qubits: int, seed: int = 42) -> np.ndarray:
    """Generate normalized random data."""
    dim = 1 << n_qubits
    rng = np.random.default_rng(seed)
    data = rng.random((n_samples, dim), dtype=np.float64)
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return data / norms


def create_variational_circuit(n_qubits: int, n_layers: int = 2, use_gpu: bool = True):
    """Create a simple variational circuit for testing."""
    # Try GPU first, then CPU fallback
    if use_gpu:
        try:
            dev = qml.device("lightning.gpu", wires=n_qubits)
            print(f"  Using device: lightning.gpu")
        except Exception as e:
            print(f"  lightning.gpu not available ({e}), falling back to lightning.qubit")
            dev = qml.device("lightning.qubit", wires=n_qubits)
    else:
        try:
            dev = qml.device("lightning.qubit", wires=n_qubits)
        except:
            dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch")
    def circuit_with_state(state):
        """Circuit that takes a pre-encoded state (GPU tensor)."""
        # Inject the pre-encoded state (from QDP) - zero-copy on GPU!
        qml.StatePrep(state, wires=range(n_qubits))

        # Variational layers
        for layer in range(n_layers):
            for i in range(n_qubits):
                qml.RY(0.1 * (layer + 1), wires=i)
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])

        return qml.state()

    @qml.qnode(dev, interface="torch")
    def circuit_with_embedding(features):
        """Circuit that does its own encoding."""
        # PennyLane's amplitude embedding
        qml.AmplitudeEmbedding(features, wires=range(n_qubits), normalize=True)

        # Same variational layers
        for layer in range(n_layers):
            for i in range(n_qubits):
                qml.RY(0.1 * (layer + 1), wires=i)
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])

        return qml.state()

    return circuit_with_state, circuit_with_embedding


def benchmark_qdp_path(
    data: np.ndarray,
    n_qubits: int,
    circuit_with_state,
    warmup: int = 2,
    runs: int = 5,
) -> Tuple[float, float, float]:
    """
    Benchmark QDP encoding + PennyLane circuit.

    Returns: (total_time, encode_time, circuit_time)
    """
    if not HAS_QDP:
        return None, None, None

    n_samples = len(data)
    engine = QdpEngine(0)

    encode_times = []
    circuit_times = []
    total_times = []

    for run_idx in range(warmup + runs):
        clean_cache()

        # Save data to temp file for QDP
        temp_path = "/tmp/qdp_benchmark_data.npy"
        np.save(temp_path, data)

        torch.cuda.synchronize()
        t_start = time.perf_counter()

        # Step 1: QDP encoding (batch, GPU)
        t_encode_start = time.perf_counter()
        qtensor = engine.encode(temp_path, n_qubits)
        states_gpu = torch.from_dlpack(qtensor)  # Zero-copy on GPU
        torch.cuda.synchronize()
        t_encode_end = time.perf_counter()

        # Step 2: Run PennyLane circuit for each sample (zero-copy GPU tensor)
        t_circuit_start = time.perf_counter()
        results = []
        for i in range(n_samples):
            # Pass GPU tensor directly - no CPU transfer!
            state = states_gpu[i]
            result = circuit_with_state(state)
            results.append(result)
        torch.cuda.synchronize()
        t_circuit_end = time.perf_counter()

        t_end = time.perf_counter()

        if run_idx >= warmup:
            encode_times.append(t_encode_end - t_encode_start)
            circuit_times.append(t_circuit_end - t_circuit_start)
            total_times.append(t_end - t_start)

    return np.mean(total_times), np.mean(encode_times), np.mean(circuit_times)


def benchmark_pennylane_path(
    data: np.ndarray,
    n_qubits: int,
    circuit_with_embedding,
    warmup: int = 2,
    runs: int = 5,
) -> Tuple[float, float]:
    """
    Benchmark pure PennyLane (encoding + circuit).

    Returns: (total_time, per_sample_time)
    """
    n_samples = len(data)
    total_times = []

    # Convert to GPU tensor for fair comparison
    data_gpu = torch.tensor(data, dtype=torch.float64, device='cuda')

    for run_idx in range(warmup + runs):
        clean_cache()

        torch.cuda.synchronize()
        t_start = time.perf_counter()

        results = []
        for i in range(n_samples):
            features = data_gpu[i]
            result = circuit_with_embedding(features)
            results.append(result)

        torch.cuda.synchronize()
        t_end = time.perf_counter()

        if run_idx >= warmup:
            total_times.append(t_end - t_start)

    mean_total = np.mean(total_times)
    return mean_total, mean_total / n_samples


def benchmark_training_simulation(
    data: np.ndarray,
    n_qubits: int,
    circuit_with_state,
    circuit_with_embedding,
    n_epochs: int = 10,
    warmup: int = 1,
) -> Tuple[dict, dict]:
    """
    Simulate training loop: encode once, run circuit multiple epochs.

    This shows QDP's real advantage - batch encoding amortized over epochs.

    Returns: (qdp_results, pennylane_results)
    """
    n_samples = len(data)

    # ========== QDP Path ==========
    # Encode ONCE, reuse for all epochs
    print("\n[QDP Path] Encode once, run multiple epochs...")

    if HAS_QDP:
        engine = QdpEngine(0)
        temp_path = "/tmp/qdp_training_data.npy"
        np.save(temp_path, data)

        clean_cache()
        torch.cuda.synchronize()

        # Encoding (one-time cost)
        t_encode_start = time.perf_counter()
        qtensor = engine.encode(temp_path, n_qubits)
        states_gpu = torch.from_dlpack(qtensor)
        torch.cuda.synchronize()
        t_encode = time.perf_counter() - t_encode_start

        print(f"  Encoding: {t_encode*1000:.2f} ms (one-time)")

        # Warmup
        for _ in range(warmup):
            for i in range(min(10, n_samples)):
                _ = circuit_with_state(states_gpu[i])

        # Training epochs
        epoch_times = []
        torch.cuda.synchronize()
        t_total_start = time.perf_counter()

        for epoch in range(n_epochs):
            t_epoch_start = time.perf_counter()
            for i in range(n_samples):
                _ = circuit_with_state(states_gpu[i])
            torch.cuda.synchronize()
            epoch_time = time.perf_counter() - t_epoch_start
            epoch_times.append(epoch_time)
            print(f"  Epoch {epoch+1}/{n_epochs}: {epoch_time*1000:.2f} ms")

        t_total = time.perf_counter() - t_total_start

        qdp_results = {
            "encode_time": t_encode,
            "epoch_times": epoch_times,
            "total_time": t_encode + t_total,
            "mean_epoch": np.mean(epoch_times),
        }
    else:
        qdp_results = None

    # ========== PennyLane Path ==========
    # Must encode every epoch (no pre-encoded states)
    print("\n[PennyLane Path] Encode + run each epoch...")

    data_gpu = torch.tensor(data, dtype=torch.float64, device='cuda')

    clean_cache()

    # Warmup
    for _ in range(warmup):
        for i in range(min(10, n_samples)):
            _ = circuit_with_embedding(data_gpu[i])

    # Training epochs
    epoch_times = []
    torch.cuda.synchronize()
    t_total_start = time.perf_counter()

    for epoch in range(n_epochs):
        t_epoch_start = time.perf_counter()
        for i in range(n_samples):
            _ = circuit_with_embedding(data_gpu[i])
        torch.cuda.synchronize()
        epoch_time = time.perf_counter() - t_epoch_start
        epoch_times.append(epoch_time)
        print(f"  Epoch {epoch+1}/{n_epochs}: {epoch_time*1000:.2f} ms")

    t_total = time.perf_counter() - t_total_start

    pl_results = {
        "encode_time": 0,  # Encoding happens every epoch
        "epoch_times": epoch_times,
        "total_time": t_total,
        "mean_epoch": np.mean(epoch_times),
    }

    return qdp_results, pl_results


def main():
    parser = argparse.ArgumentParser(
        description="QDP + PennyLane End-to-End Benchmark"
    )

    parser.add_argument(
        "--samples", nargs="+", type=int, default=[100, 500, 1000],
        help="Sample counts to benchmark"
    )
    parser.add_argument(
        "--qubits", type=int, default=8,
        help="Number of qubits (default: 8)"
    )
    parser.add_argument(
        "--layers", type=int, default=2,
        help="Number of variational layers (default: 2)"
    )
    parser.add_argument(
        "--dataset", type=str, default="synthetic",
        help="Dataset to use: synthetic, mnist_full, etc."
    )
    parser.add_argument(
        "--warmup", type=int, default=2,
        help="Warmup runs (default: 2)"
    )
    parser.add_argument(
        "--runs", type=int, default=5,
        help="Measurement runs (default: 5)"
    )
    parser.add_argument(
        "--epochs", type=int, default=0,
        help="Run training simulation with N epochs (0=disabled, default: 0)"
    )
    parser.add_argument(
        "--gpu", type=int, default=0,
        help="GPU device ID (default: 0)"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("QDP + PennyLane End-to-End Benchmark")
    print("=" * 70)
    print(f"Dataset:    {args.dataset}")
    print(f"Samples:    {args.samples}")
    print(f"Qubits:     {args.qubits}")
    print(f"Layers:     {args.layers}")
    print(f"Warmup:     {args.warmup}, Runs: {args.runs}")
    print(f"GPU:        {args.gpu}")
    print(f"QDP:        {'Available' if HAS_QDP else 'Not available'}")
    print("=" * 70)

    # Create circuits
    circuit_with_state, circuit_with_embedding = create_variational_circuit(
        args.qubits, args.layers
    )

    # ========== Correctness Verification ==========
    print("\n" + "=" * 70)
    print("CORRECTNESS VERIFICATION")
    print("=" * 70)

    # Test with a few samples to verify QDP and PennyLane produce same results
    test_data = generate_data(5, args.qubits)

    if HAS_QDP:
        engine = QdpEngine(0)
        temp_path = "/tmp/qdp_verify_data.npy"
        np.save(temp_path, test_data)

        qtensor = engine.encode(temp_path, args.qubits)
        states_gpu = torch.from_dlpack(qtensor)

        print("Comparing QDP vs PennyLane outputs...")
        all_match = True
        for i in range(len(test_data)):
            # QDP path: pre-encoded state → circuit
            qdp_result = circuit_with_state(states_gpu[i])

            # PennyLane path: raw features → AmplitudeEmbedding → circuit
            features_gpu = torch.tensor(test_data[i], dtype=torch.float64, device='cuda')
            pl_result = circuit_with_embedding(features_gpu)

            # Compare (both should be on GPU)
            qdp_np = qdp_result.detach().cpu().numpy()
            pl_np = pl_result.detach().cpu().numpy()

            # Check if close (allowing for small numerical differences)
            max_diff = np.max(np.abs(qdp_np - pl_np))
            is_close = np.allclose(qdp_np, pl_np, rtol=1e-4, atol=1e-6)

            if not is_close:
                all_match = False
                print(f"  Sample {i}: MISMATCH (max diff: {max_diff:.2e})")
            else:
                print(f"  Sample {i}: OK (max diff: {max_diff:.2e})")

        if all_match:
            print("\n✓ All samples match! QDP and PennyLane produce identical results.")
        else:
            print("\n✗ WARNING: Some samples don't match!")
    else:
        print("QDP not available, skipping verification")

    # ========== Training Simulation ==========
    if args.epochs > 0:
        print("\n" + "=" * 70)
        print(f"TRAINING SIMULATION ({args.epochs} epochs)")
        print("=" * 70)

        # Use first sample count for training simulation
        n_samples = args.samples[0]
        print(f"Samples: {n_samples}, Qubits: {args.qubits}")

        if args.dataset.lower() == "synthetic" or not HAS_DATASETS:
            data = generate_data(n_samples, args.qubits)
        else:
            ds = get_dataset(args.dataset, n_samples=n_samples, seed=42)
            data, _ = ds.prepare_for_qubits(args.qubits, normalize=True)

        qdp_res, pl_res = benchmark_training_simulation(
            data, args.qubits, circuit_with_state, circuit_with_embedding,
            n_epochs=args.epochs, warmup=1
        )

        print("\n" + "-" * 70)
        print("TRAINING SIMULATION RESULTS")
        print("-" * 70)

        if qdp_res:
            print(f"QDP Path:")
            print(f"  Encoding (one-time): {qdp_res['encode_time']*1000:.2f} ms")
            print(f"  Mean epoch time:     {qdp_res['mean_epoch']*1000:.2f} ms")
            print(f"  Total time:          {qdp_res['total_time']*1000:.2f} ms")

        print(f"\nPennyLane Path:")
        print(f"  Mean epoch time:     {pl_res['mean_epoch']*1000:.2f} ms")
        print(f"  Total time:          {pl_res['total_time']*1000:.2f} ms")

        if qdp_res:
            speedup = pl_res['total_time'] / qdp_res['total_time']
            print(f"\n>>> Training Speedup: {speedup:.2f}x")

        return  # Skip regular benchmark if training simulation

    # ========== Regular Benchmark ==========
    results = []

    for n_samples in args.samples:
        print(f"\n{'='*70}")
        print(f"Benchmarking: {n_samples} samples, {args.qubits} qubits")
        print("=" * 70)

        # Load or generate data
        if args.dataset.lower() == "synthetic" or not HAS_DATASETS:
            data = generate_data(n_samples, args.qubits)
        else:
            ds = get_dataset(args.dataset, n_samples=n_samples, seed=42)
            data, _ = ds.prepare_for_qubits(args.qubits, normalize=True)

        # Benchmark QDP path
        print("\n[QDP + PennyLane Path]")
        print("  QDP encode (GPU batch) → StatePrep → Variational circuit")

        qdp_total, qdp_encode, qdp_circuit = benchmark_qdp_path(
            data, args.qubits, circuit_with_state,
            warmup=args.warmup, runs=args.runs
        )

        if qdp_total is not None:
            print(f"  Encode time:  {qdp_encode*1000:.2f} ms ({n_samples/qdp_encode:.0f} samples/sec)")
            print(f"  Circuit time: {qdp_circuit*1000:.2f} ms ({n_samples/qdp_circuit:.0f} samples/sec)")
            print(f"  Total time:   {qdp_total*1000:.2f} ms ({n_samples/qdp_total:.0f} samples/sec)")
        else:
            print("  QDP not available")

        # Benchmark PennyLane path
        print("\n[Pure PennyLane Path]")
        print("  AmplitudeEmbedding → Variational circuit")

        pl_total, pl_per_sample = benchmark_pennylane_path(
            data, args.qubits, circuit_with_embedding,
            warmup=args.warmup, runs=args.runs
        )

        print(f"  Total time:   {pl_total*1000:.2f} ms ({n_samples/pl_total:.0f} samples/sec)")
        print(f"  Per sample:   {pl_per_sample*1000:.4f} ms")

        # Calculate speedup
        if qdp_total is not None:
            speedup = pl_total / qdp_total
            print(f"\n  >>> QDP Speedup: {speedup:.2f}x")

            results.append({
                "samples": n_samples,
                "qdp_total_ms": qdp_total * 1000,
                "qdp_encode_ms": qdp_encode * 1000,
                "qdp_circuit_ms": qdp_circuit * 1000,
                "pl_total_ms": pl_total * 1000,
                "speedup": speedup,
            })

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Samples':<10} {'QDP (ms)':<12} {'PennyLane (ms)':<15} {'Speedup':<10}")
    print("-" * 70)
    for r in results:
        print(f"{r['samples']:<10} {r['qdp_total_ms']:<12.2f} {r['pl_total_ms']:<15.2f} {r['speedup']:<10.2f}x")

    print("\n" + "=" * 70)
    print("Breakdown (QDP path):")
    print("=" * 70)
    print(f"{'Samples':<10} {'Encode (ms)':<12} {'Circuit (ms)':<12} {'Encode %':<10}")
    print("-" * 70)
    for r in results:
        encode_pct = r['qdp_encode_ms'] / r['qdp_total_ms'] * 100
        print(f"{r['samples']:<10} {r['qdp_encode_ms']:<12.2f} {r['qdp_circuit_ms']:<12.2f} {encode_pct:<10.1f}%")


if __name__ == "__main__":
    main()
