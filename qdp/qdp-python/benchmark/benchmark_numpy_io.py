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
import tempfile
import time

import numpy as np
import torch
from _qdp import QdpEngine
from utils import normalize_batch

BAR = "=" * 70
SEP = "-" * 70

try:
    import pennylane as qml

    HAS_PENNYLANE = True
except ImportError:
    HAS_PENNYLANE = False


def generate_test_data(
    num_samples: int,
    sample_size: int,
    encoding_method: str = "amplitude",
    seed: int = 42,
) -> np.ndarray:
    """Generate deterministic test data."""
    rng = np.random.RandomState(seed)
    if encoding_method == "basis":
        # Basis encoding: single index per sample
        data = rng.randint(0, sample_size, size=(num_samples, 1)).astype(np.float64)
    elif encoding_method == "angle":
        # Angle encoding: per-qubit angles in [0, 2*pi)
        data = (rng.rand(num_samples, sample_size) * (2.0 * np.pi)).astype(np.float64)
    else:
        # Amplitude encoding: full vectors (using Gaussian distribution)
        data = rng.randn(num_samples, sample_size).astype(np.float64)
        # Normalize each sample
        data = normalize_batch(data, encoding_method)
    return data


def run_mahout_numpy(
    num_qubits: int, num_samples: int, npy_path: str, encoding_method: str = "amplitude"
):
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
        # Use the unified encode API with file path
        qtensor = engine.encode(npy_path, num_qubits, encoding_method)
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


def run_pytorch_numpy(
    num_qubits: int, num_samples: int, npy_path: str, encoding_method: str = "amplitude"
):
    """Benchmark PyTorch-native engine with NumPy file I/O."""
    print("\n[PyTorch-native + NumPy] Loading and encoding...")

    from qumat_qdp.encodings import encode_batch
    from qumat_qdp.engine import QdpEngine as PyTorchEngine

    try:
        engine = PyTorchEngine(device_id=0, precision="float32")
    except Exception as exc:
        print(f"  Init failed: {exc}")
        return 0.0, 0.0, 0.0

    device = engine.device

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_total = time.perf_counter()

    try:
        # Load NumPy file
        data = np.load(npy_path)
        data_tensor = torch.from_numpy(data)

        # Encode
        result = encode_batch(
            data_tensor,
            num_qubits,
            encoding_method,
            precision="float32",
            device=device,
        )

        # Small computation to ensure GPU has processed the data
        _ = result.abs().sum()

        if torch.cuda.is_available():
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
        help="Comma-separated list: mahout,pytorch,pennylane or 'all'",
    )
    parser.add_argument(
        "--encoding-method",
        type=str,
        default="amplitude",
        choices=["amplitude", "angle", "basis"],
        help="Encoding method to use for Mahout (amplitude, angle, or basis).",
    )
    args = parser.parse_args()

    # Parse frameworks
    if args.frameworks.lower() == "all":
        frameworks = ["mahout", "pytorch", "pennylane"]
    else:
        frameworks = [f.strip().lower() for f in args.frameworks.split(",")]

    num_qubits = args.qubits
    num_samples = args.samples
    sample_size = num_qubits if args.encoding_method == "angle" else (1 << num_qubits)

    print(BAR)
    print("NUMPY I/O + ENCODING BENCHMARK")
    print(BAR)
    print(f"Qubits: {num_qubits}")
    print(f"Sample size: {sample_size} elements")
    print(f"Number of samples: {num_samples}")
    print(f"Total data: {num_samples * sample_size * 8 / (1024**2):.2f} MB")
    print(f"Frameworks: {', '.join(frameworks)}")

    # Generate test data
    print("\nGenerating test data...")
    data = generate_test_data(num_samples, sample_size, args.encoding_method)

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

    if "mahout" in frameworks:
        t_total, throughput, avg_per_sample = run_mahout_numpy(
            num_qubits, num_samples, npy_path, args.encoding_method
        )
        if throughput > 0:
            results["Mahout"] = {
                "time": t_total,
                "throughput": throughput,
                "avg_per_sample": avg_per_sample,
            }

    if "pytorch" in frameworks:
        t_total, throughput, avg_per_sample = run_pytorch_numpy(
            num_qubits, num_samples, npy_path, args.encoding_method
        )
        if throughput > 0:
            results["PyTorch"] = {
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

    # Build display order: requested frameworks mapped to display names
    framework_labels = {
        "mahout": "Mahout",
        "pytorch": "PyTorch",
        "pennylane": "PennyLane",
    }

    # Print summary
    print("\n" + BAR)
    print("SUMMARY")
    print(BAR)
    print(
        f"{'Framework':<15} {'Time (s)':<12} {'Throughput':<20} {'Avg/Sample (ms)':<15}"
    )
    print(SEP)

    for fw in frameworks:
        label = framework_labels.get(fw, fw)
        if label in results:
            m = results[label]
            print(
                f"{label:<15} "
                f"{m['time']:<12.4f} "
                f"{m['throughput']:<20.1f} "
                f"{m['avg_per_sample'] * 1000:<15.2f}"
            )
        else:
            print(f"{label:<15} {'N/A':<12} {'N/A':<20} {'N/A':<15}")

    if len(results) >= 2:
        print("\n" + SEP)
        print("SPEEDUP COMPARISON")
        print(SEP)

        pairs = [
            ("PyTorch", "Mahout"),
            ("PyTorch", "PennyLane"),
            ("Mahout", "PennyLane"),
        ]
        for a, b in pairs:
            if a in results and b in results:
                speedup = results[a]["throughput"] / results[b]["throughput"]
                print(f"{a} vs {b}: {speedup:.2f}x")
            elif a in results or b in results:
                print(f"{a} vs {b}: N/A (missing {b if b not in results else a})")

    # Cleanup
    if not args.output:
        os.remove(npy_path)
        print(f"\nCleaned up temporary file: {npy_path}")

    print("\n" + BAR)
    print("BENCHMARK COMPLETE")
    print(BAR)


if __name__ == "__main__":
    main()
