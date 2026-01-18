---
layout: page
title: API Reference - QDP
---

# API Reference

Complete API reference for QDP (Quantum Data Plane).

## QdpEngine Class

The main class for encoding classical data into quantum states using GPU acceleration.

### Constructor

```python
QdpEngine(device_id: int, precision: str = "float32")
```

Creates a QdpEngine instance for GPU-accelerated quantum data encoding.

**Parameters:**
- `device_id` (int): GPU device ID to use (e.g., 0 for first GPU, 1 for second GPU)
- `precision` (str, optional): Output precision, either "float32" (default) or "float64"

**Example:**
```python
from _qdp import QdpEngine

# Use GPU 0 with default float32 precision
engine = QdpEngine(0)

# Use GPU 1 with float64 precision
engine_high_precision = QdpEngine(1, precision="float64")
```

### Methods

#### encode

```python
encode(data, num_qubits: int, encoding_method: str)
```

Encodes classical data into a quantum tensor using the specified encoding method.

**Parameters:**
- `data`: Input data to encode. Can be:
  - Python list of floats
  - NumPy array
  - PyTorch tensor
  - File path (str) to `.parquet`, `.arrow`, `.feather`, or `.npy` file
- `num_qubits` (int): Number of qubits to use for encoding
- `encoding_method` (str): Encoding method to use:
  - `"amplitude"`: Amplitude encoding
  - `"angle"`: Angle encoding
  - `"basis"`: Basis encoding

**Returns:**
- Quantum tensor representing the encoded data

**Raises:**
- `ValueError`: If data format is unsupported or parameters are invalid
- `RuntimeError`: If GPU operation fails

**Example:**
```python
# From Python list
data = [0.5, 0.5, 0.5, 0.5]
qtensor = engine.encode(data, num_qubits=2, encoding_method="amplitude")

# From file
qtensor = engine.encode("data.parquet", num_qubits=10, encoding_method="amplitude")

# From NumPy
import numpy as np
numpy_data = np.array([1.0, 2.0, 3.0, 4.0])
qtensor = engine.encode(numpy_data, num_qubits=2, encoding_method="angle")
```

## Encoding Methods

### Amplitude Encoding

Encodes classical data into quantum state amplitudes. For n qubits, encodes 2^n values.

**Formula:** |ψ⟩ = Σᵢ xᵢ|i⟩ where xᵢ are normalized data values

**Constraints:**
- Data length must equal 2^num_qubits
- Data will be normalized to unit vector

**Use case:** Feature vectors for quantum machine learning

### Angle Encoding

Encodes data as rotation angles applied to qubits.

**Formula:** |ψ⟩ = ⨂ᵢ (cos(θᵢ)|0⟩ + sin(θᵢ)|1⟩)

**Constraints:**
- Data length should match num_qubits

**Use case:** Variational quantum algorithms

### Basis Encoding

Encodes data in computational basis states.

**Formula:** Classical bits → |b₁b₂...bₙ⟩

**Constraints:**
- Data interpreted as binary values

**Use case:** Binary classification, discrete optimization

## File Format Support

QDP supports reading data from multiple file formats:

### Parquet Files

```python
qtensor = engine.encode("dataset.parquet", num_qubits=10, encoding_method="amplitude")
```

- Efficient columnar storage
- Compressed data
- Standard for big data

### Arrow IPC Files

```python
qtensor = engine.encode("dataset.arrow", num_qubits=10, encoding_method="amplitude")
qtensor = engine.encode("dataset.feather", num_qubits=10, encoding_method="amplitude")
```

- In-memory columnar format
- Zero-copy reads
- Cross-platform compatibility

### NumPy Files

```python
qtensor = engine.encode("dataset.npy", num_qubits=10, encoding_method="amplitude")
```

- Binary NumPy array format
- Fast loading
- Python ecosystem standard

## DLPack Integration

QDP uses the DLPack protocol for zero-copy tensor transfer between frameworks:

```python
import torch
import numpy as np

# PyTorch tensor (zero-copy)
torch_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0])
qtensor = engine.encode(torch_tensor, num_qubits=2, encoding_method="amplitude")

# NumPy array (zero-copy)
numpy_array = np.array([1.0, 2.0, 3.0, 4.0])
qtensor = engine.encode(numpy_array, num_qubits=2, encoding_method="amplitude")
```

## Error Handling

QDP provides clear error messages for common issues:

```python
# Invalid device ID
engine = QdpEngine(999)  # Raises error if GPU 999 doesn't exist

# Invalid encoding method
qtensor = engine.encode(data, 2, "invalid_method")  # Raises ValueError

# Invalid data size
data = [1.0, 2.0, 3.0]  # Not a power of 2
qtensor = engine.encode(data, 2, "amplitude")  # Raises ValueError
```

## Performance Tips

1. **Choose appropriate precision**: Use float32 unless you need float64 accuracy
2. **Batch processing**: Process multiple samples to amortize overhead
3. **Keep data on GPU**: Minimize CPU-GPU transfers
4. **File formats**: Parquet and Arrow are optimized for large datasets
5. **Device selection**: Distribute work across multiple GPUs if available

## Integration with Qumat

```python
import qumat.qdp as qdp

# Use the qdp module from qumat
engine = qdp.QdpEngine(device_id=0)
qtensor = engine.encode(data, num_qubits=2, encoding_method="amplitude")
```

## See Also

- [Getting Started Guide](/qumat/qdp/getting-started)
- [Core Concepts](/qumat/qdp/concepts)
- [Examples](/qumat/qdp/examples)
- [Development Guide](https://github.com/apache/mahout/blob/trunk/qdp/DEVELOPMENT.md)
- [Source Code on GitHub](https://github.com/apache/mahout/tree/trunk/qdp)
