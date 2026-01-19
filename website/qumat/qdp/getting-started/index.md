---
layout: page
title: Getting Started with QDP
---

# Getting Started with QDP

QDP (Quantum Data Plane) is a high-performance GPU-accelerated library for encoding classical data into quantum states. It provides zero-copy tensor transfer via DLPack for seamless integration with PyTorch, NumPy, and TensorFlow.

## Prerequisites

- Linux machine
- NVIDIA GPU with CUDA driver and toolkit installed
- Python 3.10 or higher
- Rust & Cargo

You can verify your CUDA installation with:

```bash
nvcc --version
```

## Installation

### Option 1: Install from Source

```bash
git clone https://github.com/apache/mahout.git
cd mahout
pip install uv
uv sync --group qdp  # Install QDP with dependencies
```

### Option 2: Build the Python Package

Navigate to the `qdp/` directory:

```bash
cd qdp
make install
```

Or manually:

```bash
cd qdp/qdp-python
uv venv -p python3.11
source .venv/bin/activate
uv sync --group dev
uv run maturin develop
```

## Quick Example

```python
from _qdp import QdpEngine

# Initialize on GPU 0 (defaults to float32 output)
engine = QdpEngine(0)

# Encode data from Python list
data = [0.5, 0.5, 0.5, 0.5]
qtensor = engine.encode(data, num_qubits=2, encoding_method="amplitude")
```

### Using with Qumat

```python
import qumat.qdp as qdp

engine = qdp.QdpEngine(device_id=0)
qtensor = engine.encode([1.0, 2.0, 3.0, 4.0], num_qubits=2, encoding_method="amplitude")
```

## Encoding Methods

QDP supports three encoding methods:

- **`amplitude`** - Amplitude encoding: Encodes classical data into quantum state amplitudes
- **`angle`** - Angle encoding: Encodes data as rotation angles
- **`basis`** - Basis encoding: Encodes data in computational basis states

## Supported File Formats

QDP can read data directly from files:

```python
# Parquet files
tensor_parquet = engine.encode("data.parquet", 10, "amplitude")

# Arrow IPC files
tensor_arrow = engine.encode("data.arrow", 10, "amplitude")

# NumPy files
tensor_npy = engine.encode("data.npy", 10, "amplitude")
```

## Precision Options

By default, QDP uses float32 output. For higher precision:

```python
engine = QdpEngine(0, precision="float64")
```

## Next Steps

- Learn about [QDP Concepts](/qumat/qdp/concepts)
- Explore the [API Reference](/qumat/qdp/api)
- Check out [Examples](/qumat/qdp/examples)
- Read the [Development Guide](https://github.com/apache/mahout/blob/trunk/qdp/DEVELOPMENT.md) for building and testing
