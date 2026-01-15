# QDP Benchmark Roadmap: Demonstrating Real QDP Advantage

This document outlines actionable issues for demonstrating QDP's true advantage in quantum machine learning workflows.

---

## Executive Summary

| Issue | File | Priority | Effort | Key Metric |
|-------|------|----------|--------|------------|
| **#1 Parameter-Shift Gradient** | `benchmark_gradient.py` | **High** | Medium | **5-20x speedup** expected |
| #2 Batched Execution | Research | Medium | High | Research findings |
| #3 cuStateVec Integration | Future | Low | Very High | Zero-copy path |
| #4 State Caching | `core/state_cache.py` | Medium | Medium | Memory efficiency |
| #5 Real Dataset Scaling | `benchmark_scaling.py` | Medium | Low | Dataset support |
| #6 Benchmark Report | `docs/BENCHMARK_REPORT.md` | High | Low | Documentation |

### Quick Start: Implement Issue #1

```bash
# 1. Create the gradient benchmark (see design below)
touch benchmark/benchmark_gradient.py

# 2. Run quick test
uv run python benchmark/benchmark_gradient.py \
    --n-qubits 10 \
    --n-layers 2 \
    --n-samples 100 \
    --verify \
    --no-save

# 3. Full benchmark
uv run python benchmark/benchmark_gradient.py \
    --n-qubits 12 \
    --n-layers 2 \
    --n-samples 100 500 1000 2000 \
    --output-dir ./results
```

---

## Background

### Current Benchmark Findings

| Benchmark | QDP Speedup | Notes |
|-----------|-------------|-------|
| Raw Encoding | 4-13x | Scales with batch size |
| E2E Training (single forward) | 1.02-1.10x | Bottlenecked by StatePrep |

**Key Insight**: In single-forward-pass training loops, QDP's encoding speedup is masked by the `StatePrep` operation, which still needs to copy the encoded state into the quantum simulator's internal representation.

### Where QDP Really Shines

QDP's true advantage emerges when:
1. **Encoded state is reused multiple times** (amortizes encoding cost)
2. **Batched circuit execution** (GPU parallelism)
3. **Gradient computation via parameter-shift** (2P circuit evaluations per sample)

---

## Issue 1: Parameter-Shift Gradient Benchmark

**Priority**: High
**Complexity**: Medium
**File**: `benchmark/benchmark_gradient.py` (new)

### Problem Statement

In variational quantum circuits, gradient computation via [parameter-shift rule](https://pennylane.ai/qml/glossary/parameter_shift.html) requires:
- For P parameters: **2P circuit evaluations per sample**
- With N samples: **2P × N total evaluations**

Current training benchmark only measures forward pass, hiding QDP's reuse advantage.

### Proposed Solution

Create a benchmark that compares gradient computation:

```
QDP Path:
  1. Encode N samples once via QDP → N state vectors
  2. For each parameter p (1 to P):
     - Run circuit with θ_p + π/2 on all N states
     - Run circuit with θ_p - π/2 on all N states
  3. Compute gradients from differences

  Encoding calls: 1
  Circuit evaluations: 2P × N

PennyLane Path:
  1. For each parameter p:
     - For each sample n:
       - Encode sample n
       - Run circuit with θ_p + π/2
       - Encode sample n again
       - Run circuit with θ_p - π/2

  Encoding calls: 2P × N (or N if batched)
  Circuit evaluations: 2P × N
```

### Expected Results

For a 12-qubit circuit with 24 parameters (2 layers × 12 qubits):
- Parameter-shift requires 2 × 24 = 48 circuit evaluations per sample
- With 1000 samples: 48,000 circuit evaluations
- QDP encodes once, PennyLane encodes 48,000 times (or 1000 if batched+cached)

**Expected speedup**: 5-20x depending on caching and batching

### Design

#### Module Structure

```
benchmark/
├── benchmark_gradient.py          # Main CLI entry point
└── core/
    └── gradient_benchmark.py      # GradientBenchmark class (optional refactor)
```

#### Core Classes

```python
# benchmark/benchmark_gradient.py

from dataclasses import dataclass
from typing import Optional
import torch
import pennylane as qml

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
    gradients: Optional[torch.Tensor]  # For correctness verification


class GradientBenchmark:
    """Benchmark gradient computation via parameter-shift rule."""

    def __init__(
        self,
        n_qubits: int = 12,
        n_layers: int = 2,
        use_gpu: bool = True,
        seed: int = 42,
    ):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_parameters = n_layers * n_qubits * 2  # RY + RZ per qubit per layer
        self.use_gpu = use_gpu
        self.seed = seed
        self.device = self._create_device()
        self.params = self._init_parameters()

    def _create_device(self) -> qml.Device:
        """Create PennyLane device (lightning.gpu or default.qubit)."""
        if self.use_gpu:
            return qml.device("lightning.gpu", wires=self.n_qubits)
        return qml.device("default.qubit", wires=self.n_qubits)

    def _init_parameters(self) -> torch.Tensor:
        """Initialize variational parameters."""
        torch.manual_seed(self.seed)
        return torch.randn(
            self.n_parameters,
            dtype=torch.float64,
            device="cuda" if self.use_gpu else "cpu",
            requires_grad=True,
        )

    def run_qdp_path(
        self,
        data: torch.Tensor,
        warmup: int = 1,
        runs: int = 3,
    ) -> GradientBenchmarkResult:
        """
        Compute gradients using QDP encoding (encode once, reuse 2P times).

        Steps:
        1. Encode all samples once via QDP → cached states
        2. For each parameter p:
           - Shift θ_p by +π/2, run all samples
           - Shift θ_p by -π/2, run all samples
        3. Compute gradient: (f+ - f-) / 2
        """
        ...

    def run_pennylane_path(
        self,
        data: torch.Tensor,
        warmup: int = 1,
        runs: int = 3,
    ) -> GradientBenchmarkResult:
        """
        Compute gradients using pure PennyLane (encode per circuit evaluation).

        Steps:
        1. For each parameter p:
           - For each sample n:
             - Encode sample via AmplitudeEmbedding
             - Run circuit with θ_p + π/2
             - Encode sample again
             - Run circuit with θ_p - π/2
        2. Compute gradient: (f+ - f-) / 2
        """
        ...

    def verify_correctness(
        self,
        qdp_result: GradientBenchmarkResult,
        pl_result: GradientBenchmarkResult,
        rtol: float = 1e-5,
        atol: float = 1e-8,
    ) -> bool:
        """Verify QDP and PennyLane gradients match."""
        return torch.allclose(
            qdp_result.gradients, pl_result.gradients, rtol=rtol, atol=atol
        )
```

#### Circuit Design

```python
def create_variational_circuit(n_qubits: int, n_layers: int):
    """
    Create a variational circuit for gradient benchmarking.

    Architecture per layer:
    - RY(θ) on each qubit
    - RZ(θ) on each qubit
    - CNOT ladder (qubit i → qubit i+1)

    Total parameters: n_layers * n_qubits * 2
    """

    def circuit(state, params):
        # Inject pre-encoded state
        qml.StatePrep(state, wires=range(n_qubits))

        # Variational layers
        param_idx = 0
        for layer in range(n_layers):
            # Rotation gates
            for qubit in range(n_qubits):
                qml.RY(params[param_idx], wires=qubit)
                param_idx += 1
            for qubit in range(n_qubits):
                qml.RZ(params[param_idx], wires=qubit)
                param_idx += 1

            # Entangling gates
            for qubit in range(n_qubits - 1):
                qml.CNOT(wires=[qubit, qubit + 1])

        # Measurement: expectation of Z on first qubit
        return qml.expval(qml.PauliZ(0))

    return circuit
```

#### Parameter-Shift Implementation

```python
def compute_gradient_parameter_shift(
    circuit_fn,
    states: torch.Tensor,      # Shape: (n_samples, 2^n_qubits)
    params: torch.Tensor,      # Shape: (n_parameters,)
    shift: float = np.pi / 2,
) -> torch.Tensor:
    """
    Compute gradients using parameter-shift rule.

    For each parameter p:
        grad[p] = (f(θ_p + shift) - f(θ_p - shift)) / (2 * sin(shift))

    Returns:
        Tensor of shape (n_samples, n_parameters)
    """
    n_samples = states.shape[0]
    n_params = params.shape[0]
    gradients = torch.zeros(n_samples, n_params, device=params.device)

    for p in range(n_params):
        # Shift parameter +π/2
        params_plus = params.clone()
        params_plus[p] += shift

        # Shift parameter -π/2
        params_minus = params.clone()
        params_minus[p] -= shift

        # Evaluate circuit for all samples
        for i, state in enumerate(states):
            f_plus = circuit_fn(state, params_plus)
            f_minus = circuit_fn(state, params_minus)
            gradients[i, p] = (f_plus - f_minus) / (2 * np.sin(shift))

    return gradients
```

### Expected Usage

#### CLI Interface

```bash
# Basic usage - compare QDP vs PennyLane gradient computation
uv run python benchmark/benchmark_gradient.py

# Specify parameters
uv run python benchmark/benchmark_gradient.py \
    --n-qubits 12 \
    --n-layers 2 \
    --n-samples 100 500 1000

# With correctness verification
uv run python benchmark/benchmark_gradient.py \
    --n-samples 100 \
    --verify

# High precision benchmark
uv run python benchmark/benchmark_gradient.py \
    --warmup 3 \
    --runs 10 \
    --n-samples 500 1000 2000

# Use specific GPU
uv run python benchmark/benchmark_gradient.py \
    --gpu 1 \
    --n-samples 1000

# Save results
uv run python benchmark/benchmark_gradient.py \
    --output-dir ./results \
    --n-samples 100 500 1000 2000 5000

# Use real dataset
uv run python benchmark/benchmark_gradient.py \
    --dataset mnist_full \
    --n-samples 1000 \
    --n-qubits 10
```

#### Expected Console Output

```
============================================================
GRADIENT BENCHMARK (Parameter-Shift Rule)
============================================================
Circuit:    12 qubits, 2 layers, 48 parameters
Samples:    [100, 500, 1000]
GPU:        0 (NVIDIA A100)
Warmup:     1, Runs: 3
============================================================

Testing with n_samples=100...

  QDP Path:
    Encoding time:  2.34 ms (encode once)
    Circuit time:   156.78 ms (48 params × 2 shifts × 100 samples = 9600 evals)
    Total time:     159.12 ms
    Throughput:     628.5 samples/sec

  PennyLane Path:
    Encoding time:  234.56 ms (encode per eval: 9600 times)
    Circuit time:   156.89 ms (same circuit evaluations)
    Total time:     391.45 ms
    Throughput:     255.5 samples/sec

  Speedup: 2.46x (QDP vs PennyLane)

Testing with n_samples=500...
  ...

Testing with n_samples=1000...
  ...

============================================================
RESULTS SUMMARY
============================================================
Samples   QDP (ms)    PennyLane (ms)   Speedup   Breakdown (enc/ckt)
--------------------------------------------------------------------
100       159.12      391.45           2.46x     2ms/157ms vs 235ms/157ms
500       795.67      1957.23          2.46x     12ms/784ms vs 1173ms/784ms
1000      1591.34     3914.46          2.46x     23ms/1568ms vs 2346ms/1568ms

============================================================
ANALYSIS
============================================================
- Encoding accounts for 60% of PennyLane's total time
- QDP encodes once; PennyLane encodes 2P×N times
- Circuit execution time is identical (as expected)
- QDP speedup scales with parameter count

Saved: results/gradient_benchmark_20260115_143022.csv
Saved: results/gradient_benchmark_20260115_143022.png
```

#### Expected Plot

The benchmark generates a comparison plot showing:
- X-axis: Number of samples
- Y-axis: Total time (ms) or Throughput (samples/sec)
- Two lines: QDP Path vs PennyLane Path
- Annotations: Speedup at each point

### Implementation Tasks

- [ ] Create `benchmark/benchmark_gradient.py`
- [ ] Implement `GradientBenchmark` class with `run_qdp_path()` and `run_pennylane_path()`
- [ ] Implement parameter-shift gradient computation
- [ ] Add time breakdown: encoding vs circuit execution
- [ ] Add CLI with argparse (mirrors `benchmark_scaling.py` style)
- [ ] Add correctness verification (`--verify` flag)
- [ ] Add CSV and PNG output
- [ ] Add dataset integration (`--dataset` option)
- [ ] Add tests in `benchmark/tests/test_gradient_benchmark.py`

### Verification Command

```bash
# Quick verification test
uv run python benchmark/benchmark_gradient.py \
    --n-qubits 8 \
    --n-layers 1 \
    --n-samples 50 \
    --verify \
    --no-save

# Full benchmark with verification
uv run python benchmark/benchmark_gradient.py \
    --n-qubits 12 \
    --n-layers 2 \
    --n-samples 100 500 1000 \
    --verify \
    --warmup 2 \
    --runs 5
```

---

## Issue 2: Batched Circuit Execution Exploration

**Priority**: Medium
**Complexity**: High
**Status**: Research

### Problem Statement

Current PennyLane execution is sample-by-sample. Even with QDP's batch encoding, circuit execution is sequential, creating a bottleneck.

### Research Questions

1. Can we execute multiple circuits in parallel on GPU?
2. Does `lightning.gpu` support batched execution?
3. What is the overhead of `StatePrep` vs direct statevector injection?

### Investigation Tasks

- [ ] Profile `lightning.gpu` execution to identify bottlenecks
- [ ] Research cuStateVec batched simulation APIs
- [ ] Test if passing batch of states to single circuit call is faster
- [ ] Explore `qml.batch_transform` for parallel execution
- [ ] Document findings in `benchmark/docs/batched_execution.md`

### Potential Approaches

1. **PennyLane batch_transform**: Transform circuit to accept batch input
2. **Manual batching**: Loop unrolling with multiple devices
3. **Custom simulator**: Direct cuStateVec integration

---

## Issue 3: Direct cuStateVec Integration

**Priority**: Low
**Complexity**: Very High
**Status**: Future Work

### Problem Statement

`StatePrep` operation in PennyLane copies the state vector into the simulator. For true zero-copy, we need direct integration with cuStateVec.

### Proposed Architecture

```
Current:
  QDP (GPU Tensor) → .cpu().numpy() → StatePrep → cuStateVec internal copy

Proposed:
  QDP (GPU Tensor) → Direct DLPack → cuStateVec internal buffer
```

### Investigation Tasks

- [ ] Study cuStateVec API for external state buffer injection
- [ ] Check if `custatevecInitializeStateVector` supports external memory
- [ ] Prototype direct CUDA interop between QDP and cuStateVec
- [ ] Measure memory transfer overhead in current path

### References

- [NVIDIA cuStateVec Documentation](https://docs.nvidia.com/cuda/cuquantum/latest/custatevec/index.html)
- [PennyLane Lightning GPU Source](https://github.com/PennyLaneAI/pennylane-lightning)
- [DLPack Specification](https://dmlc.github.io/dlpack/latest/)

---

## Issue 4: Multi-Sample State Vector Caching

**Priority**: Medium
**Complexity**: Medium
**File**: `benchmark/core/state_cache.py` (new)

### Problem Statement

When computing gradients, the same sample's state vector is used 2P times. Currently, each use may re-inject the state. An explicit caching layer can:
1. Pre-allocate GPU memory for all encoded states
2. Enable efficient indexed access during gradient computation
3. Support memory-mapped caching for datasets larger than GPU memory

### Design

```python
# benchmark/core/state_cache.py

import torch
from typing import Optional
import _qdp  # Mahout QDP bindings

class StateCache:
    """
    GPU-resident cache for pre-encoded quantum states.

    Supports:
    - Batch encoding via QDP
    - Indexed access for gradient computation
    - Memory tracking and management
    """

    def __init__(
        self,
        n_samples: int,
        n_qubits: int,
        device: str = "cuda",
        dtype: torch.dtype = torch.complex128,
    ):
        self.n_samples = n_samples
        self.n_qubits = n_qubits
        self.state_dim = 2 ** n_qubits
        self.device = device
        self.dtype = dtype

        # Pre-allocate cache
        self._cache: Optional[torch.Tensor] = None
        self._encoded = False

    @property
    def memory_bytes(self) -> int:
        """Memory usage in bytes."""
        # complex128 = 16 bytes per element
        bytes_per_element = 16 if self.dtype == torch.complex128 else 8
        return self.n_samples * self.state_dim * bytes_per_element

    @property
    def memory_mb(self) -> float:
        """Memory usage in MB."""
        return self.memory_bytes / (1024 * 1024)

    def encode_all(self, data: torch.Tensor) -> None:
        """
        Encode all samples and store in cache.

        Args:
            data: Input features, shape (n_samples, state_dim)
        """
        if data.shape[0] != self.n_samples:
            raise ValueError(f"Expected {self.n_samples} samples, got {data.shape[0]}")

        # Encode via QDP (batch operation)
        self._cache = _qdp.amplitude_encode(
            data.to(self.device),
            normalize=True,
        )
        self._encoded = True

    def get_state(self, idx: int) -> torch.Tensor:
        """Get single encoded state by index."""
        if not self._encoded:
            raise RuntimeError("Cache not populated. Call encode_all() first.")
        return self._cache[idx]

    def get_states(self, indices: torch.Tensor) -> torch.Tensor:
        """Get multiple encoded states by indices (for batched access)."""
        if not self._encoded:
            raise RuntimeError("Cache not populated. Call encode_all() first.")
        return self._cache[indices]

    def get_all(self) -> torch.Tensor:
        """Get all encoded states."""
        if not self._encoded:
            raise RuntimeError("Cache not populated. Call encode_all() first.")
        return self._cache

    def clear(self) -> None:
        """Free GPU memory."""
        self._cache = None
        self._encoded = False
        torch.cuda.empty_cache()
```

### Usage Example

```python
# In gradient benchmark

def run_qdp_path(self, data: torch.Tensor) -> GradientBenchmarkResult:
    n_samples = data.shape[0]

    # === ENCODING PHASE (timed separately) ===
    cache = StateCache(n_samples, self.n_qubits)

    start_encode = torch.cuda.Event(enable_timing=True)
    end_encode = torch.cuda.Event(enable_timing=True)

    start_encode.record()
    cache.encode_all(data)
    end_encode.record()
    torch.cuda.synchronize()

    encoding_time_ms = start_encode.elapsed_time(end_encode)

    # === CIRCUIT PHASE (timed separately) ===
    start_circuit = torch.cuda.Event(enable_timing=True)
    end_circuit = torch.cuda.Event(enable_timing=True)

    start_circuit.record()

    # All states already encoded, just access from cache
    all_states = cache.get_all()  # No re-encoding!

    for p in range(self.n_parameters):
        # Shift parameter
        params_plus = self.params.clone()
        params_plus[p] += np.pi / 2
        params_minus = self.params.clone()
        params_minus[p] -= np.pi / 2

        # Evaluate on all cached states
        for i in range(n_samples):
            state = all_states[i]  # O(1) access, no encoding
            f_plus = self.circuit(state, params_plus)
            f_minus = self.circuit(state, params_minus)
            gradients[i, p] = (f_plus - f_minus) / 2

    end_circuit.record()
    torch.cuda.synchronize()

    circuit_time_ms = start_circuit.elapsed_time(end_circuit)

    return GradientBenchmarkResult(
        framework="qdp",
        encoding_time_ms=encoding_time_ms,
        circuit_time_ms=circuit_time_ms,
        total_time_ms=encoding_time_ms + circuit_time_ms,
        ...
    )
```

### Memory Considerations

| Qubits | State Dim | Samples | Memory (MB) |
|--------|-----------|---------|-------------|
| 10 | 1,024 | 1,000 | 15.6 |
| 12 | 4,096 | 1,000 | 62.5 |
| 14 | 16,384 | 1,000 | 250.0 |
| 16 | 65,536 | 1,000 | 1,000.0 |
| 12 | 4,096 | 10,000 | 625.0 |

For large datasets with many qubits, consider:
- Chunked processing (encode/process in batches)
- Mixed precision (complex64 instead of complex128)
- CPU-GPU streaming for datasets > GPU memory

### Tasks

- [ ] Implement `StateCache` class in `benchmark/core/state_cache.py`
- [ ] Add memory tracking and reporting
- [ ] Integrate with gradient benchmark
- [ ] Add chunked processing for large datasets
- [ ] Add unit tests for cache operations
- [ ] Document memory requirements in README

---

## Issue 5: Real Dataset Scaling Benchmark

**Priority**: Medium
**Complexity**: Low
**File**: `benchmark/benchmark_scaling.py` (modify)

### Current Status

✅ `--dataset` CLI option added
✅ Full MNIST (70k samples) integrated
✅ Basic scaling benchmark works with real data

### Remaining Tasks

- [ ] Add `--dataset-samples` to limit samples from large datasets
- [ ] Add memory usage reporting
- [ ] Test with larger datasets (CIFAR-10, Fashion-MNIST)
- [ ] Document dataset preparation requirements

---

## Issue 6: Comprehensive Benchmark Report

**Priority**: High
**Complexity**: Low
**File**: `benchmark/docs/BENCHMARK_REPORT.md` (new)

### Purpose

Create a publication-ready benchmark report summarizing QDP performance characteristics.

### Content Outline

1. **Executive Summary**: Key findings and recommendations
2. **Methodology**: Hardware, software versions, measurement approach
3. **Raw Encoding Performance**: QDP vs PennyLane vs Qiskit
4. **End-to-End Training**: With and without QDP
5. **Gradient Computation**: Parameter-shift comparison
6. **Dataset Impact**: Synthetic vs real data
7. **Scalability**: Qubits, samples, batch size
8. **Recommendations**: When to use QDP

### Report Template

```markdown
# QDP Performance Benchmark Report

**Date**: YYYY-MM-DD
**Hardware**: NVIDIA [GPU Model], [RAM] GB
**Software**: QDP v[X.Y.Z], PennyLane v[X.Y.Z], PyTorch v[X.Y.Z]

## Executive Summary

QDP provides **Nx speedup** over pure PennyLane encoding in:
- Raw encoding throughput: [X]x at [N] samples
- Parameter-shift gradients: [Y]x at [P] parameters

**Recommendation**: Use QDP when:
- Batch size > 1000 samples
- Computing gradients via parameter-shift
- Working with large datasets (MNIST, etc.)

## 1. Raw Encoding Performance

### 1.1 Throughput vs Sample Count (12 qubits)

![Throughput Plot](./results/scaling_samples_throughput.png)

| Samples | QDP (s/s) | PennyLane (s/s) | Speedup |
|---------|-----------|-----------------|---------|
| 500     | [X]       | [Y]             | [Z]x    |
| 1,000   | [X]       | [Y]             | [Z]x    |
| ...     | ...       | ...             | ...     |

### 1.2 Throughput vs Qubit Count (1000 samples)

![Qubits Plot](./results/scaling_qubits_throughput.png)

...

## 2. Parameter-Shift Gradient Performance

### 2.1 Configuration

- Circuit: 12 qubits, 2 layers, 48 parameters
- Gradient method: Parameter-shift rule (2P evaluations per sample)

### 2.2 Results

| Samples | QDP (ms) | PennyLane (ms) | Speedup |
|---------|----------|----------------|---------|
| 100     | [X]      | [Y]            | [Z]x    |
| 500     | [X]      | [Y]            | [Z]x    |
| ...     | ...      | ...            | ...     |

### 2.3 Time Breakdown

![Breakdown Plot](./results/gradient_breakdown.png)

- Encoding time: QDP [X] ms vs PennyLane [Y] ms
- Circuit time: [Z] ms (identical)

## 3. Dataset Impact

### 3.1 Synthetic vs MNIST

...

## 4. Scalability Analysis

### 4.1 Memory Usage

...

### 4.2 GPU Utilization

...

## 5. Methodology

### 5.1 Hardware

- GPU: NVIDIA [Model]
- CPU: [Model]
- RAM: [X] GB
- CUDA: [Version]

### 5.2 Software Versions

```
qumat-qdp==X.Y.Z
pennylane==X.Y.Z
pennylane-lightning-gpu==X.Y.Z
torch==X.Y.Z
numpy==X.Y.Z
```

### 5.3 Measurement Protocol

- Warmup runs: 3
- Measurement runs: 10
- Timing: CUDA events (for GPU code)
- Synchronization: `torch.cuda.synchronize()` before/after

## 6. Reproducibility

### 6.1 Commands

```bash
# Install dependencies
cd qdp/qdp-python
uv sync --group benchmark

# Build QDP
uv run maturin develop --release

# Run benchmarks
uv run python benchmark/benchmark_scaling.py \
    --samples 500 1000 2000 5000 10000 \
    --warmup 3 --runs 10

uv run python benchmark/benchmark_gradient.py \
    --n-samples 100 500 1000 2000 \
    --warmup 3 --runs 10
```

### 6.2 Data Availability

All raw benchmark data is available in `benchmark/results/`.

## 7. Conclusions

...
```

### Tasks

- [ ] Run comprehensive benchmark suite with production settings
- [ ] Generate all plots (throughput, latency, gradients, breakdown)
- [ ] Fill in template with actual results
- [ ] Add GPU utilization metrics (nvidia-smi logging)
- [ ] Document exact reproduction steps
- [ ] Review for publication quality

---

## Priority Matrix

| Issue | Priority | Effort | Impact | Dependencies |
|-------|----------|--------|--------|--------------|
| #1 Parameter-Shift Gradient | High | Medium | High | None |
| #2 Batched Execution | Medium | High | High | Research |
| #3 cuStateVec Integration | Low | Very High | Very High | #2 |
| #4 State Caching | Medium | Medium | Medium | #1 |
| #5 Real Dataset Scaling | Medium | Low | Medium | None |
| #6 Benchmark Report | High | Low | High | #1, #5 |

---

## Suggested Implementation Order

1. **Phase 1** (Immediate):
   - Issue #1: Parameter-Shift Gradient Benchmark
   - Issue #5: Real Dataset Scaling (remaining tasks)

2. **Phase 2** (After validation):
   - Issue #4: State Caching
   - Issue #6: Comprehensive Report

3. **Phase 3** (Research):
   - Issue #2: Batched Execution Exploration
   - Issue #3: cuStateVec Integration (if #2 is promising)

---

## Success Criteria

- [ ] Parameter-shift benchmark shows **>5x speedup** for QDP path
- [ ] Benchmark report ready for publication/blog post
- [ ] Clear documentation of when to use QDP vs alternatives
- [ ] Reproducible benchmarks with CI integration

---

## Related Files

| File | Purpose |
|------|---------|
| `benchmark/benchmark_scaling.py` | Encoding throughput comparison |
| `benchmark/benchmark_qdp_pennylane.py` | E2E training comparison |
| `benchmark/benchmark_gradient.py` | Parameter-shift gradients (new) |
| `benchmark/datasets/` | Dataset abstractions |
| `benchmark/core/` | Benchmark infrastructure |

---

## References

1. [PennyLane Parameter-Shift Rule](https://pennylane.ai/qml/glossary/parameter_shift.html)
2. [NVIDIA cuQuantum SDK](https://developer.nvidia.com/cuquantum-sdk)
3. [PennyLane Lightning GPU](https://docs.pennylane.ai/projects/lightning/en/stable/lightning_gpu/device.html)
4. [DLPack Zero-Copy Tensor Exchange](https://dmlc.github.io/dlpack/latest/)
