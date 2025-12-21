# Enhanced Benchmark E2E with Component Timing

## Overview

The `benchmark_e2e.py` script has been enhanced to provide detailed component-level timing evaluation. This allows you to see exactly how much time each component (IO, normalization, encoding, transfer, forward pass) consumes across different frameworks.

## Features

### Component Timing Breakdown

Each framework now reports detailed timing for its components:

**Mahout (Parquet/Arrow):**
- IO + Encoding: Combined disk read and quantum encoding
- DLPack Conversion: Converting to PyTorch tensor
- Reshape & Convert: Reshaping and type conversion
- Forward Pass: Neural network forward pass

**PennyLane:**
- IO (Disk Read): Reading data from disk
- Encoding (with Norm): Quantum encoding with normalization
- GPU Transfer: Moving data to GPU
- Forward Pass: Neural network forward pass

**Qiskit:**
- IO (Disk Read): Reading data from disk
- Normalization: L2 normalization of input vectors
- Encoding (State Prep): Quantum state preparation
- GPU Transfer: Moving data to GPU
- Forward Pass: Neural network forward pass

### Comparison Table

At the end of the benchmark, a comprehensive comparison table shows the time spent on each component across all frameworks, making it easy to identify bottlenecks.

## Usage

### Basic Usage

```bash
# Run with default parameters (16 qubits, 200 samples)
python benchmark_e2e.py

# Run with custom parameters
python benchmark_e2e.py --qubits 18 --samples 500

# Run on specific GPU
CUDA_VISIBLE_DEVICES=1 python benchmark_e2e.py --qubits 18 --samples 500
```

### Framework Selection

```bash
# Run only Mahout benchmarks
python benchmark_e2e.py --frameworks mahout-parquet mahout-arrow

# Run Mahout and PennyLane comparison
python benchmark_e2e.py --frameworks mahout-parquet pennylane

# Run all available frameworks
python benchmark_e2e.py --frameworks all
```

## Example Output

```
======================================================================
E2E BENCHMARK: 18 Qubits, 500 Samples
======================================================================

[Mahout-Parquet] Full Pipeline (Parquet -> GPU)...
  Parquet->GPU (IO+Encode): 1.2345 s
  DLPack conversion: 0.0123 s
  Reshape & convert: 0.0234 s
  Total Time: 1.3456 s

  === Mahout-Parquet Component Breakdown ===
  1. IO + Encoding              1.2345 s ( 91.7%)
  2. DLPack Conversion          0.0123 s (  0.9%)
  3. Reshape & Convert          0.0234 s (  1.7%)
  4. Forward Pass               0.0754 s (  5.6%)
  Total                         1.3456 s (100.0%)

======================================================================
E2E LATENCY (Lower is Better)
Samples: 500, Qubits: 18
======================================================================
Mahout-Parquet    1.3456 s
Mahout-Arrow      1.2987 s
PennyLane         5.6789 s
Qiskit           12.3456 s
----------------------------------------------------------------------
Speedup vs PennyLane:    4.37x
Speedup vs Qiskit:       9.51x

======================================================================
COMPONENT TIMING COMPARISON
Samples: 500, Qubits: 18
======================================================================
Component                       Mahout-Parquet       PennyLane          Qiskit
----------------------------------------------------------------------
1. IO (Disk Read)                            -        0.3456s        0.3456s
1. IO + Encoding                      1.2345s               -               -
2. DLPack Conversion                  0.0123s               -               -
2. Encoding (with Norm)                      -        4.8901s               -
2. Normalization                             -               -        0.4567s
3. Encoding (State Prep)                     -               -       10.1234s
3. GPU Transfer                              -        0.2345s               -
3. Reshape & Convert                  0.0234s               -               -
4. Forward Pass                       0.0754s        0.2087s               -
4. GPU Transfer                              -               -        0.9876s
5. Forward Pass                              -               -        0.4323s
----------------------------------------------------------------------
TOTAL                                 1.3456s        5.6789s       12.3456s
```

## Interpretation

The component breakdown helps identify:

1. **I/O Bottlenecks**: If IO time is high, consider faster storage or data formats
2. **Encoding Efficiency**: Compare encoding times across frameworks
3. **Transfer Overhead**: GPU transfer time can be significant for CPU-based frameworks
4. **Overall Pipeline**: See which parts of the pipeline dominate execution time

The Mahout framework's advantage comes from:
- Combined IO+Encoding that happens directly on GPU
- Minimal data transfer overhead (DLPack zero-copy when possible)
- Efficient quantum state preparation kernels

## Understanding GPU Usage

**Which frameworks use GPU for quantum encoding?**

- **Mahout (Parquet/Arrow)**: ✅ GPU-accelerated quantum encoding via CUDA kernels
  - Encoding happens on GPU (this is the main advantage)
  - Only minimal CPU usage for coordination

- **PennyLane**: ❌ CPU-based quantum encoding
  - Quantum state preparation happens on CPU
  - Data is transferred to GPU only for the neural network forward pass
  - Expect high CPU usage during encoding phase

- **Qiskit**: ❌ CPU-based quantum encoding
  - Quantum state preparation happens on CPU via AerSimulator
  - Data is transferred to GPU only for the neural network forward pass
  - Expect high CPU usage during encoding phase

**To maximize GPU usage:**
```bash
# Run ONLY Mahout frameworks (exclude PennyLane and Qiskit)
CUDA_VISIBLE_DEVICES=1 ./benchmark_e2e.py --qubits 18 --samples 500 --frameworks mahout-parquet

# Or both Mahout variants for comparison
CUDA_VISIBLE_DEVICES=1 ./benchmark_e2e.py --qubits 18 --samples 500 --frameworks mahout-parquet mahout-arrow
```

**Verify GPU usage:**
- Check the component timing breakdown
- Mahout's "IO + Encoding" should be 5-10x faster than PennyLane's "Encoding (with Norm)"
- If timings are similar, there may be an installation issue

**Troubleshooting:**
```bash
# Verify Mahout QDP engine can access GPU
python -c "from mahout_qdp import QdpEngine; engine = QdpEngine(0); print('GPU initialized successfully')"

# Check CUDA is available to PyTorch
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

## Requirements

- NVIDIA GPU with CUDA support
- Python 3.10+
- All dependencies from `requirements.txt`
- Mahout QDP package installed (`maturin develop`)

See [DEVELOPMENT.md](../DEVELOPMENT.md) for setup instructions.
