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
Benchmark: PyTorch-native encodings vs Rust+CUDA encodings.

Compares performance of the new PyTorch-native QDP encodings against the
Rust+CUDA backend at various qubit counts and batch sizes.

Usage:
    python benchmark/benchmark_pytorch_vs_rust.py
    python benchmark/benchmark_pytorch_vs_rust.py --qubits 2 4 8 12 16 --batch-size 64
"""

from __future__ import annotations

import argparse
import time
from typing import Any

import numpy as np
import torch


def _has_rust_engine() -> bool:
    try:
        import _qdp

        return hasattr(_qdp, "QdpEngine")
    except ImportError:
        return False


def _has_cuda() -> bool:
    return torch.cuda.is_available()


def benchmark_pytorch_encode(
    num_qubits: int,
    batch_size: int,
    encoding_method: str,
    num_batches: int = 50,
    warmup: int = 5,
    precision: str = "float32",
) -> dict[str, Any]:
    """Benchmark PyTorch-native encoding."""
    from qumat_qdp.encodings import encode_batch

    device = torch.device("cuda" if _has_cuda() else "cpu")

    # Determine sample size
    if encoding_method == "amplitude":
        sample_size = 1 << num_qubits
    elif encoding_method == "basis":
        sample_size = 1
    elif encoding_method in ("angle", "iqp-z"):
        sample_size = num_qubits
    elif encoding_method == "iqp":
        n = num_qubits
        sample_size = n + n * (n - 1) // 2
    else:
        raise ValueError(f"Unknown encoding: {encoding_method}")

    # Warmup
    for _ in range(warmup):
        if encoding_method == "basis":
            data = torch.randint(
                0, 1 << num_qubits, (batch_size, 1), dtype=torch.float64
            )
        else:
            data = torch.randn(batch_size, sample_size)
        encode_batch(
            data, num_qubits, encoding_method, precision=precision, device=device
        )

    if _has_cuda():
        torch.cuda.synchronize()

    # Benchmark
    t0 = time.perf_counter()
    for _ in range(num_batches):
        if encoding_method == "basis":
            data = torch.randint(
                0, 1 << num_qubits, (batch_size, 1), dtype=torch.float64
            )
        else:
            data = torch.randn(batch_size, sample_size)
        encode_batch(
            data, num_qubits, encoding_method, precision=precision, device=device
        )
    if _has_cuda():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    total_vectors = num_batches * batch_size
    return {
        "backend": "pytorch",
        "device": str(device),
        "num_qubits": num_qubits,
        "batch_size": batch_size,
        "encoding": encoding_method,
        "total_vectors": total_vectors,
        "elapsed_sec": elapsed,
        "vectors_per_sec": total_vectors / elapsed,
        "ms_per_vector": (elapsed / total_vectors) * 1000,
    }


def benchmark_rust_encode(
    num_qubits: int,
    batch_size: int,
    encoding_method: str,
    num_batches: int = 50,
    warmup: int = 5,
) -> dict[str, Any] | None:
    """Benchmark Rust+CUDA encoding (if available)."""
    if not _has_rust_engine():
        return None

    import _qdp

    engine = _qdp.QdpEngine(device_id=0)
    state_len = 1 << num_qubits

    # Determine sample size
    if encoding_method == "amplitude":
        sample_size = state_len
    elif encoding_method == "basis":
        sample_size = 1
    elif encoding_method in ("angle", "iqp-z"):
        sample_size = num_qubits
    elif encoding_method == "iqp":
        n = num_qubits
        sample_size = n + n * (n - 1) // 2
    else:
        raise ValueError(f"Unknown encoding: {encoding_method}")

    def _make_data():
        if encoding_method == "basis":
            return np.random.randint(0, state_len, size=(batch_size, 1)).astype(
                np.float64
            )
        return np.random.randn(batch_size, sample_size).astype(np.float64)

    # Warmup
    for _ in range(warmup):
        data = _make_data()
        try:
            result = engine.encode(
                data, num_qubits=num_qubits, encoding_method=encoding_method
            )
            _ = torch.from_dlpack(result)
        except Exception:
            return None

    if _has_cuda():
        torch.cuda.synchronize()

    # Benchmark
    t0 = time.perf_counter()
    for _ in range(num_batches):
        data = _make_data()
        result = engine.encode(
            data, num_qubits=num_qubits, encoding_method=encoding_method
        )
        _ = torch.from_dlpack(result)
    if _has_cuda():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    total_vectors = num_batches * batch_size
    return {
        "backend": "rust+cuda",
        "device": "cuda:0",
        "num_qubits": num_qubits,
        "batch_size": batch_size,
        "encoding": encoding_method,
        "total_vectors": total_vectors,
        "elapsed_sec": elapsed,
        "vectors_per_sec": total_vectors / elapsed,
        "ms_per_vector": (elapsed / total_vectors) * 1000,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark PyTorch vs Rust+CUDA QDP encodings"
    )
    parser.add_argument(
        "--qubits",
        type=int,
        nargs="+",
        default=[2, 4, 8, 12, 16],
        help="Qubit counts to benchmark (default: 2 4 8 12 16)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size (default: 64)",
    )
    parser.add_argument(
        "--encoding",
        type=str,
        nargs="+",
        default=["amplitude", "angle", "basis", "iqp-z"],
        help="Encoding methods to benchmark",
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=50,
        help="Number of batches per benchmark run (default: 50)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Number of warmup batches (default: 5)",
    )
    args = parser.parse_args()

    has_rust = _has_rust_engine()
    has_cuda = _has_cuda()

    print("QDP Encoding Benchmark: PyTorch-native vs Rust+CUDA")
    print(f"  CUDA available: {has_cuda}")
    print(f"  Rust engine available: {has_rust}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Batches per run: {args.num_batches}")
    print()

    results: list[dict[str, Any]] = []

    for enc in args.encoding:
        print(f"--- Encoding: {enc} ---")
        print(
            f"{'Qubits':>8} {'StateLen':>10} "
            f"{'PyTorch v/s':>14} {'PyTorch ms/v':>14}"
            + (
                f" {'Rust v/s':>14} {'Rust ms/v':>14} {'Speedup':>10}"
                if has_rust
                else ""
            )
        )

        for nq in args.qubits:
            state_len = 1 << nq

            # Skip IQP for large qubit counts (ZZ terms grow quadratically)
            if enc in ("iqp", "iqp-z") and nq > 16:
                print(f"{nq:>8} {'(skipped)':>10}")
                continue

            pt_result = benchmark_pytorch_encode(
                nq,
                args.batch_size,
                enc,
                num_batches=args.num_batches,
                warmup=args.warmup,
            )
            results.append(pt_result)

            rust_result = None
            if has_rust:
                rust_result = benchmark_rust_encode(
                    nq,
                    args.batch_size,
                    enc,
                    num_batches=args.num_batches,
                    warmup=args.warmup,
                )
                if rust_result:
                    results.append(rust_result)

            line = (
                f"{nq:>8} {state_len:>10} "
                f"{pt_result['vectors_per_sec']:>14.0f} {pt_result['ms_per_vector']:>14.4f}"
            )
            if has_rust and rust_result:
                speedup = rust_result["vectors_per_sec"] / pt_result["vectors_per_sec"]
                line += (
                    f" {rust_result['vectors_per_sec']:>14.0f} "
                    f"{rust_result['ms_per_vector']:>14.4f} "
                    f"{speedup:>9.2f}x"
                )
            elif has_rust:
                line += " (rust failed)"

            print(line)

        print()

    print("Done.")


if __name__ == "__main__":
    main()
