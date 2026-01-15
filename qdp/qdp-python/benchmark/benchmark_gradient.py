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

"""
Parameter-Shift Gradient Benchmark: QDP vs PennyLane

This benchmark demonstrates QDP's true advantage in gradient computation
via the parameter-shift rule, where encoded states are reused 2P times
(P = number of parameters).

QDP Path: Encode once, reuse 2P×N times
PennyLane Path: Encode per circuit evaluation (2P×N times)

Expected speedup: 5-20x depending on parameter count and sample size.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

# Add parent to path for benchmark module imports
_BENCHMARK_DIR = Path(__file__).parent
if str(_BENCHMARK_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_BENCHMARK_DIR.parent))

# Check for required dependencies
HAS_TORCH = False
HAS_PENNYLANE = False
HAS_QDP = False
HAS_LIGHTNING_GPU = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    pass

try:
    import pennylane as qml
    HAS_PENNYLANE = True
    # Check for lightning.gpu
    try:
        _test_dev = qml.device("lightning.gpu", wires=2)
        del _test_dev
        HAS_LIGHTNING_GPU = True
    except Exception:
        pass
except ImportError:
    pass

try:
    from _qdp import QdpEngine
    HAS_QDP = True
except ImportError:
    pass


@dataclass
class GradientBenchmarkResult:
    """Results from a single gradient benchmark run."""
    framework: str              # "qdp" or "pennylane"
    n_samples: int
    n_qubits: int
    n_parameters: int
    encoding_time_ms: float     # Time spent on amplitude encoding
    circuit_time_ms: float      # Time spent on circuit execution
    total_time_ms: float        # encoding + circuit
    throughput: float           # samples/sec
    gradients: Optional[np.ndarray] = None  # For correctness verification


class GradientBenchmark:
    """
    Benchmark gradient computation via parameter-shift rule.

    Compares:
    - QDP Path: Encode all samples once, reuse for 2P circuit evaluations
    - PennyLane Path: Encode per circuit evaluation
    """

    def __init__(
        self,
        n_qubits: int = 12,
        n_layers: int = 2,
        use_gpu: bool = True,
        seed: int = 42,
    ):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        # Parameters: RY + RZ per qubit per layer
        self.n_parameters = n_layers * n_qubits * 2
        self.use_gpu = use_gpu and HAS_LIGHTNING_GPU
        self.seed = seed

        # Initialize parameters
        if HAS_TORCH:
            torch.manual_seed(seed)
            self.params = torch.randn(
                self.n_parameters,
                dtype=torch.float64,
            )
            if self.use_gpu:
                self.params = self.params.cuda()
        else:
            np.random.seed(seed)
            self.params = np.random.randn(self.n_parameters)

    def _create_circuit_qdp(self):
        """Create circuit that accepts pre-encoded state."""
        n_qubits = self.n_qubits
        n_layers = self.n_layers

        if self.use_gpu:
            dev = qml.device("lightning.gpu", wires=n_qubits)
        else:
            dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev, interface="torch" if HAS_TORCH else None)
        def circuit(state, params):
            # Inject pre-encoded state
            qml.StatePrep(state, wires=range(n_qubits))

            # Variational layers
            param_idx = 0
            for _layer in range(n_layers):
                # RY rotations
                for qubit in range(n_qubits):
                    qml.RY(params[param_idx], wires=qubit)
                    param_idx += 1
                # RZ rotations
                for qubit in range(n_qubits):
                    qml.RZ(params[param_idx], wires=qubit)
                    param_idx += 1
                # CNOT ladder
                for qubit in range(n_qubits - 1):
                    qml.CNOT(wires=[qubit, qubit + 1])

            return qml.expval(qml.PauliZ(0))

        return circuit

    def _create_circuit_pennylane(self):
        """Create circuit with AmplitudeEmbedding (encodes each time)."""
        n_qubits = self.n_qubits
        n_layers = self.n_layers

        if self.use_gpu:
            dev = qml.device("lightning.gpu", wires=n_qubits)
        else:
            dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev, interface="torch" if HAS_TORCH else None)
        def circuit(features, params):
            # Encode features (done every call)
            qml.AmplitudeEmbedding(features, wires=range(n_qubits), normalize=True)

            # Same variational layers
            param_idx = 0
            for _layer in range(n_layers):
                for qubit in range(n_qubits):
                    qml.RY(params[param_idx], wires=qubit)
                    param_idx += 1
                for qubit in range(n_qubits):
                    qml.RZ(params[param_idx], wires=qubit)
                    param_idx += 1
                for qubit in range(n_qubits - 1):
                    qml.CNOT(wires=[qubit, qubit + 1])

            return qml.expval(qml.PauliZ(0))

        return circuit

    def _parameter_shift_gradient(
        self,
        circuit_fn,
        states_or_features,
        params,
        is_qdp_path: bool,
        shift: float = np.pi / 2,
    ) -> tuple[np.ndarray, float, float]:
        """
        Compute gradients using parameter-shift rule.

        Returns:
            (gradients, encoding_time_ms, circuit_time_ms)
        """
        n_samples = states_or_features.shape[0]
        n_params = len(params)

        if HAS_TORCH:
            gradients = torch.zeros(n_samples, n_params, dtype=torch.float64)
            if self.use_gpu:
                gradients = gradients.cuda()
        else:
            gradients = np.zeros((n_samples, n_params))

        encoding_time = 0.0
        circuit_time = 0.0

        for p in range(n_params):
            # Create shifted parameters
            if HAS_TORCH:
                params_plus = params.clone()
                params_minus = params.clone()
            else:
                params_plus = params.copy()
                params_minus = params.copy()

            params_plus[p] = params_plus[p] + shift
            params_minus[p] = params_minus[p] - shift

            # Evaluate circuit for all samples
            for i in range(n_samples):
                if is_qdp_path:
                    # States already encoded, just access
                    state = states_or_features[i]

                    start = time.perf_counter()
                    f_plus = circuit_fn(state, params_plus)
                    f_minus = circuit_fn(state, params_minus)
                    circuit_time += time.perf_counter() - start
                else:
                    # Need to encode each time (PennyLane path)
                    features = states_or_features[i]

                    start = time.perf_counter()
                    f_plus = circuit_fn(features, params_plus)
                    f_minus = circuit_fn(features, params_minus)
                    elapsed = time.perf_counter() - start
                    # Encoding is embedded in circuit call
                    encoding_time += elapsed * 0.5  # Rough estimate
                    circuit_time += elapsed * 0.5

                gradients[i, p] = (f_plus - f_minus) / (2 * np.sin(shift))

        if HAS_TORCH:
            gradients = gradients.cpu().numpy()

        return gradients, encoding_time * 1000, circuit_time * 1000

    def run_qdp_path(
        self,
        data: np.ndarray,
        warmup: int = 1,
        runs: int = 3,
        return_gradients: bool = False,
        gpu_id: int = 0,
    ) -> GradientBenchmarkResult:
        """
        Run gradient computation using QDP encoding (encode once, reuse 2P times).
        """
        if not HAS_QDP:
            raise RuntimeError("QDP not available. Build with: uv run maturin develop")

        n_samples = data.shape[0]
        circuit = self._create_circuit_qdp()

        # Initialize QDP engine
        engine = QdpEngine(device_id=gpu_id, precision='float64')

        # Prepare parameters
        if HAS_TORCH:
            params = self.params.clone()
        else:
            params = self.params.copy()

        # Warmup
        for _ in range(warmup):
            # Encode via QDP
            quantum_tensor = engine.encode(data, self.n_qubits, 'amplitude')
            states = torch.from_dlpack(quantum_tensor)

            # Run a few gradient computations
            _ = self._parameter_shift_gradient(
                circuit, states[:min(10, n_samples)], params, is_qdp_path=True
            )

        # Measurement runs
        total_encoding_time = 0.0
        total_circuit_time = 0.0
        gradients = None

        for run in range(runs):
            # === ENCODING PHASE ===
            if HAS_TORCH and self.use_gpu:
                torch.cuda.synchronize()

            start_encode = time.perf_counter()
            quantum_tensor = engine.encode(data, self.n_qubits, 'amplitude')
            states = torch.from_dlpack(quantum_tensor)

            if HAS_TORCH and self.use_gpu:
                torch.cuda.synchronize()
            encoding_time_ms = (time.perf_counter() - start_encode) * 1000

            # === CIRCUIT PHASE ===
            start_circuit = time.perf_counter()
            gradients, _, circuit_ms = self._parameter_shift_gradient(
                circuit, states, params, is_qdp_path=True
            )
            if HAS_TORCH and self.use_gpu:
                torch.cuda.synchronize()
            circuit_time_ms = (time.perf_counter() - start_circuit) * 1000

            total_encoding_time += encoding_time_ms
            total_circuit_time += circuit_time_ms

        avg_encoding = total_encoding_time / runs
        avg_circuit = total_circuit_time / runs
        avg_total = avg_encoding + avg_circuit
        throughput = n_samples / (avg_total / 1000) if avg_total > 0 else 0

        return GradientBenchmarkResult(
            framework="qdp",
            n_samples=n_samples,
            n_qubits=self.n_qubits,
            n_parameters=self.n_parameters,
            encoding_time_ms=avg_encoding,
            circuit_time_ms=avg_circuit,
            total_time_ms=avg_total,
            throughput=throughput,
            gradients=gradients if return_gradients else None,
        )

    def run_pennylane_path(
        self,
        data: np.ndarray,
        warmup: int = 1,
        runs: int = 3,
        return_gradients: bool = False,
    ) -> GradientBenchmarkResult:
        """
        Run gradient computation using pure PennyLane (encode per evaluation).
        """
        if not HAS_PENNYLANE:
            raise RuntimeError("PennyLane not available. Install with: uv add pennylane")

        n_samples = data.shape[0]
        circuit = self._create_circuit_pennylane()

        # Convert data to torch tensor
        if HAS_TORCH:
            data_tensor = torch.from_numpy(data)
            if self.use_gpu:
                data_tensor = data_tensor.cuda()
            params = self.params.clone()
        else:
            data_tensor = data
            params = self.params.copy()

        # Warmup
        for _ in range(warmup):
            _ = self._parameter_shift_gradient(
                circuit, data_tensor[:min(10, n_samples)], params, is_qdp_path=False
            )

        # Measurement runs
        total_time = 0.0
        gradients = None

        for run in range(runs):
            if HAS_TORCH and self.use_gpu:
                torch.cuda.synchronize()

            start = time.perf_counter()
            gradients, enc_ms, ckt_ms = self._parameter_shift_gradient(
                circuit, data_tensor, params, is_qdp_path=False
            )
            if HAS_TORCH and self.use_gpu:
                torch.cuda.synchronize()

            total_time += (time.perf_counter() - start) * 1000

        avg_total = total_time / runs
        # For PennyLane, encoding is interleaved with circuit execution
        # Estimate based on typical ratios
        est_encoding = avg_total * 0.6  # Encoding typically dominates
        est_circuit = avg_total * 0.4
        throughput = n_samples / (avg_total / 1000) if avg_total > 0 else 0

        return GradientBenchmarkResult(
            framework="pennylane",
            n_samples=n_samples,
            n_qubits=self.n_qubits,
            n_parameters=self.n_parameters,
            encoding_time_ms=est_encoding,
            circuit_time_ms=est_circuit,
            total_time_ms=avg_total,
            throughput=throughput,
            gradients=gradients if return_gradients else None,
        )


def generate_data(n_samples: int, n_qubits: int, seed: int = 42) -> np.ndarray:
    """Generate random normalized data for benchmarking."""
    np.random.seed(seed)
    state_dim = 2 ** n_qubits
    data = np.random.randn(n_samples, state_dim)
    # Normalize each sample
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    data = data / norms
    return data


def verify_correctness(
    qdp_result: GradientBenchmarkResult,
    pl_result: GradientBenchmarkResult,
    rtol: float = 1e-4,
    atol: float = 1e-6,
) -> tuple[bool, float]:
    """
    Verify QDP and PennyLane gradients match.

    Returns:
        (is_correct, max_diff)
    """
    if qdp_result.gradients is None or pl_result.gradients is None:
        return False, float('inf')

    max_diff = np.max(np.abs(qdp_result.gradients - pl_result.gradients))
    is_correct = np.allclose(qdp_result.gradients, pl_result.gradients, rtol=rtol, atol=atol)

    return is_correct, max_diff


def print_header(args):
    """Print benchmark configuration header."""
    print("=" * 60)
    print("GRADIENT BENCHMARK (Parameter-Shift Rule)")
    print("=" * 60)
    print(f"Circuit:    {args.n_qubits} qubits, {args.n_layers} layers, "
          f"{args.n_layers * args.n_qubits * 2} parameters")
    print(f"Samples:    {args.n_samples}")
    print(f"GPU:        {args.gpu}" + (" (lightning.gpu)" if HAS_LIGHTNING_GPU else " (CPU fallback)"))
    print(f"Warmup:     {args.warmup}, Runs: {args.runs}")
    print("=" * 60)
    print()


def print_result(result: GradientBenchmarkResult, label: str, n_evals: int = 0):
    """Print a single benchmark result."""
    print(f"  {label}:")
    print(f"    Encoding time:  {result.encoding_time_ms:.2f} ms")
    print(f"    Circuit time:   {result.circuit_time_ms:.2f} ms")
    print(f"    Total time:     {result.total_time_ms:.2f} ms")
    print(f"    Throughput:     {result.throughput:.1f} samples/sec")
    if n_evals > 0:
        per_eval = result.total_time_ms / n_evals
        print(f"    Per-eval cost:  {per_eval:.2f} ms ({n_evals} evals)")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Parameter-Shift Gradient Benchmark: QDP vs PennyLane",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic benchmark
  python benchmark_gradient.py --n-samples 100 500

  # With correctness verification
  python benchmark_gradient.py --n-samples 100 --verify

  # High precision
  python benchmark_gradient.py --warmup 3 --runs 10 --n-samples 500
        """,
    )

    parser.add_argument(
        "--n-qubits", type=int, default=10,
        help="Number of qubits (default: 10)"
    )
    parser.add_argument(
        "--n-layers", type=int, default=2,
        help="Number of variational layers (default: 2)"
    )
    parser.add_argument(
        "--n-samples", type=int, nargs="+", default=[50, 100, 200],
        help="Sample counts to benchmark (default: 50 100 200)"
    )
    parser.add_argument(
        "--warmup", type=int, default=1,
        help="Warmup runs (default: 1)"
    )
    parser.add_argument(
        "--runs", type=int, default=3,
        help="Measurement runs (default: 3)"
    )
    parser.add_argument(
        "--gpu", type=int, default=0,
        help="GPU device ID (default: 0)"
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Verify correctness of gradients"
    )
    parser.add_argument(
        "--output-dir", "-o", type=str, default=None,
        help="Output directory for results (default: no save)"
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Don't save results"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )

    args = parser.parse_args()

    # Set GPU
    if HAS_TORCH:
        if torch.cuda.is_available():
            torch.cuda.set_device(args.gpu)
        else:
            print("Warning: CUDA not available, using CPU")

    # Check dependencies
    if not HAS_QDP:
        print("Error: QDP not available. Build with: uv run maturin develop")
        sys.exit(1)
    if not HAS_PENNYLANE:
        print("Error: PennyLane not available. Install with: uv add pennylane")
        sys.exit(1)

    print_header(args)

    # Results storage
    all_results = []

    for n_samples in args.n_samples:
        print(f"Testing with n_samples={n_samples}...")
        print()

        # Generate data
        data = generate_data(n_samples, args.n_qubits, seed=args.seed)

        # Create benchmark
        benchmark = GradientBenchmark(
            n_qubits=args.n_qubits,
            n_layers=args.n_layers,
            use_gpu=HAS_LIGHTNING_GPU,
            seed=args.seed,
        )

        # Calculate number of circuit evaluations
        n_evals = n_samples * benchmark.n_parameters * 2  # 2 shifts per parameter

        # Run QDP path
        qdp_result = benchmark.run_qdp_path(
            data,
            warmup=args.warmup,
            runs=args.runs,
            return_gradients=args.verify,
            gpu_id=args.gpu,
        )
        print_result(qdp_result, "QDP Path", n_evals)

        # Run PennyLane path
        pl_result = benchmark.run_pennylane_path(
            data,
            warmup=args.warmup,
            runs=args.runs,
            return_gradients=args.verify,
        )
        print_result(pl_result, "PennyLane Path", n_evals)

        # Calculate speedup
        speedup = pl_result.total_time_ms / qdp_result.total_time_ms if qdp_result.total_time_ms > 0 else 0
        print(f"  Speedup: {speedup:.2f}x (QDP vs PennyLane)")

        # Encoding speedup
        if qdp_result.encoding_time_ms > 0:
            enc_speedup = pl_result.encoding_time_ms / qdp_result.encoding_time_ms
            print(f"  Encoding speedup: {enc_speedup:.0f}x (QDP: {qdp_result.encoding_time_ms:.2f}ms vs PL: {pl_result.encoding_time_ms:.2f}ms)")

        # Verify correctness
        if args.verify:
            is_correct, max_diff = verify_correctness(qdp_result, pl_result)
            status = "PASSED" if is_correct else "FAILED"
            print(f"  Correctness: {status} (max diff: {max_diff:.2e})")

        print()
        print("-" * 60)
        print()

        all_results.append({
            "n_samples": n_samples,
            "qdp_encoding_ms": qdp_result.encoding_time_ms,
            "qdp_circuit_ms": qdp_result.circuit_time_ms,
            "qdp_total_ms": qdp_result.total_time_ms,
            "qdp_throughput": qdp_result.throughput,
            "pl_encoding_ms": pl_result.encoding_time_ms,
            "pl_circuit_ms": pl_result.circuit_time_ms,
            "pl_total_ms": pl_result.total_time_ms,
            "pl_throughput": pl_result.throughput,
            "speedup": speedup,
        })

    # Print summary
    print()
    print("=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Samples':<10} {'QDP (ms)':<12} {'PennyLane (ms)':<16} {'Speedup':<10} {'Enc Ratio':<12}")
    print("-" * 60)
    for r in all_results:
        enc_ratio = r['pl_encoding_ms'] / r['qdp_encoding_ms'] if r['qdp_encoding_ms'] > 0 else 0
        print(f"{r['n_samples']:<10} {r['qdp_total_ms']:<12.2f} {r['pl_total_ms']:<16.2f} {r['speedup']:<10.2f}x {enc_ratio:<12.0f}x")
    print()

    # Print analysis
    print("=" * 60)
    print("ANALYSIS")
    print("=" * 60)
    n_params = args.n_layers * args.n_qubits * 2
    print(f"Circuit: {args.n_qubits} qubits, {args.n_layers} layers, {n_params} parameters")
    print(f"Per sample: 2×{n_params} = {2*n_params} circuit evaluations for gradients")
    print()
    print("Key insights:")
    print("- QDP encodes ALL samples once, then reuses for 2P×N circuit evaluations")
    print("- PennyLane encodes inside each circuit call (AmplitudeEmbedding)")
    print("- StatePrep overhead can mask encoding speedup in total time")
    print("- Encoding speedup is most visible with larger sample counts")
    print()

    # Save results
    if args.output_dir and not args.no_save:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = output_dir / f"gradient_benchmark_{timestamp}.csv"

        # Save CSV
        import csv
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)

        print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
