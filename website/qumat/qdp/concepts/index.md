---
layout: page
title: Core Concepts - QDP
---

# Core Concepts

Understanding these key concepts will help you use QDP (Quantum Data Plane) effectively.

## What is QDP?

QDP (Quantum Data Plane) is a high-performance GPU-accelerated library for encoding classical data into quantum states. It bridges the gap between classical machine learning frameworks and quantum computing by providing:

- **GPU Acceleration**: CUDA kernels for fast data encoding
- **Zero-Copy Transfer**: DLPack protocol for seamless tensor movement
- **Multiple Encodings**: Support for amplitude, angle, and basis encoding
- **Framework Integration**: Works with PyTorch, NumPy, and TensorFlow

## Encoding Methods

QDP supports three primary encoding methods:

### Amplitude Encoding

Encodes classical data into quantum state amplitudes. For n qubits, you can encode 2^n classical values.

```python
# Encode 4 values into 2 qubits
data = [0.5, 0.5, 0.5, 0.5]
qtensor = engine.encode(data, num_qubits=2, encoding_method="amplitude")
```

**Use case**: Encoding feature vectors for quantum machine learning

### Angle Encoding

Encodes data as rotation angles applied to qubits.

```python
qtensor = engine.encode(data, num_qubits=2, encoding_method="angle")
```

**Use case**: Variational quantum algorithms

### Basis Encoding

Encodes data in computational basis states (|0⟩ and |1⟩).

```python
qtensor = engine.encode(data, num_qubits=2, encoding_method="basis")
```

**Use case**: Binary classification problems

## GPU Acceleration

QDP leverages CUDA to perform data encoding on GPU, providing significant speedup over CPU-based approaches:

- **Parallel Processing**: CUDA kernels execute encoding in parallel
- **Memory Management**: Efficient GPU memory allocation and transfer
- **Device Selection**: Choose which GPU to use via device_id

```python
# Use GPU 0
engine = QdpEngine(device_id=0)

# Or GPU 1 if you have multiple GPUs
engine = QdpEngine(device_id=1)
```

## Zero-Copy Data Transfer

QDP uses the DLPack protocol for zero-copy tensor transfer between frameworks:

```python
import torch
import numpy as np

# From PyTorch (zero-copy)
torch_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0])
qtensor = engine.encode(torch_tensor, num_qubits=2, encoding_method="amplitude")

# From NumPy (zero-copy)
numpy_array = np.array([1.0, 2.0, 3.0, 4.0])
qtensor = engine.encode(numpy_array, num_qubits=2, encoding_method="amplitude")
```

## File Format Support

QDP can read data directly from various file formats:

- **Parquet**: Columnar storage format
- **Arrow IPC**: In-memory columnar format
- **NumPy**: `.npy` binary format

```python
# Read from file
qtensor = engine.encode("data.parquet", num_qubits=10, encoding_method="amplitude")
```

## Precision Options

QDP supports both float32 and float64 precision:

```python
# Default: float32 (faster, less memory)
engine = QdpEngine(0)

# High precision: float64 (slower, more memory)
engine = QdpEngine(0, precision="float64")
```

**Trade-off**: float32 is faster and uses less memory, while float64 provides higher numerical accuracy.

## Data Flow

The typical QDP workflow:

1. **Initialize Engine**: Create QdpEngine with device_id
2. **Prepare Data**: Load or create classical data
3. **Encode**: Convert classical data to quantum tensor
4. **Use in Quantum Circuit**: Feed encoded data to quantum algorithms

```python
# 1. Initialize
engine = QdpEngine(device_id=0)

# 2. Prepare
data = [1.0, 2.0, 3.0, 4.0]

# 3. Encode
qtensor = engine.encode(data, num_qubits=2, encoding_method="amplitude")

# 4. Use (in your quantum circuit/algorithm)
```

## Performance Considerations

- **GPU Memory**: Ensure sufficient GPU memory for your data size
- **Batch Processing**: Process data in batches for large datasets
- **Data Locality**: Keep data on GPU to avoid transfer overhead
- **Precision**: Use float32 unless you need float64 accuracy

## Integration with Qumat

QDP integrates seamlessly with Qumat Core:

```python
import qumat.qdp as qdp

engine = qdp.QdpEngine(device_id=0)
qtensor = engine.encode(data, num_qubits=2, encoding_method="amplitude")
```

## Next Steps

- Explore the [API Reference](/qumat/qdp/api) for detailed method documentation
- Try the [Examples](/qumat/qdp/examples) to see QDP in action
- Read the [Development Guide](https://github.com/apache/mahout/blob/trunk/qdp/DEVELOPMENT.md) for building from source
