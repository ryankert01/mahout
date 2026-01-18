---
layout: page
title: Examples - QDP
---

# Examples

Practical examples showing how to use QDP for encoding classical data into quantum states.

## Basic Encoding from Python List

The simplest way to encode data:

```python
from _qdp import QdpEngine

# Initialize on GPU 0
engine = QdpEngine(0)

# Encode 4 values into 2 qubits using amplitude encoding
data = [0.5, 0.5, 0.5, 0.5]
qtensor = engine.encode(data, num_qubits=2, encoding_method="amplitude")

print(f"Encoded tensor: {qtensor}")
```

## Using Different Encoding Methods

Compare the three encoding methods:

```python
from _qdp import QdpEngine

engine = QdpEngine(0)
data = [1.0, 2.0, 3.0, 4.0]

# Amplitude encoding
amp_tensor = engine.encode(data, num_qubits=2, encoding_method="amplitude")
print(f"Amplitude encoding: {amp_tensor}")

# Angle encoding
angle_tensor = engine.encode(data, num_qubits=2, encoding_method="angle")
print(f"Angle encoding: {angle_tensor}")

# Basis encoding
basis_tensor = engine.encode(data, num_qubits=2, encoding_method="basis")
print(f"Basis encoding: {basis_tensor}")
```

## Encoding from NumPy Arrays

Zero-copy encoding from NumPy:

```python
import numpy as np
from _qdp import QdpEngine

engine = QdpEngine(0)

# Create NumPy array
numpy_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])

# Encode with zero-copy transfer
qtensor = engine.encode(numpy_data, num_qubits=3, encoding_method="amplitude")
print(f"Encoded {len(numpy_data)} values into {3} qubits")
```

## Encoding from PyTorch Tensors

Integrate with PyTorch:

```python
import torch
from _qdp import QdpEngine

engine = QdpEngine(0)

# Create PyTorch tensor
torch_data = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)

# Zero-copy encoding
qtensor = engine.encode(torch_data, num_qubits=2, encoding_method="amplitude")
print(f"Encoded PyTorch tensor: {qtensor}")
```

## Reading from Files

Encode data directly from files:

```python
from _qdp import QdpEngine

engine = QdpEngine(0)

# From Parquet file
qtensor_parquet = engine.encode("data.parquet", num_qubits=10, encoding_method="amplitude")

# From Arrow IPC file
qtensor_arrow = engine.encode("data.arrow", num_qubits=10, encoding_method="amplitude")

# From NumPy file
qtensor_npy = engine.encode("data.npy", num_qubits=10, encoding_method="amplitude")

print("Successfully encoded data from files")
```

## High Precision Encoding

Use float64 for higher precision:

```python
from _qdp import QdpEngine

# Initialize with float64 precision
engine = QdpEngine(0, precision="float64")

data = [0.123456789012345, 0.987654321098765, 0.111111111111111, 0.999999999999999]
qtensor = engine.encode(data, num_qubits=2, encoding_method="amplitude")

print(f"High-precision encoding: {qtensor}")
```

## Multi-GPU Setup

Use different GPUs:

```python
from _qdp import QdpEngine

# Use GPU 0
engine_0 = QdpEngine(device_id=0)
qtensor_0 = engine_0.encode([1.0, 2.0, 3.0, 4.0], num_qubits=2, encoding_method="amplitude")

# Use GPU 1 (if available)
engine_1 = QdpEngine(device_id=1)
qtensor_1 = engine_1.encode([5.0, 6.0, 7.0, 8.0], num_qubits=2, encoding_method="amplitude")

print("Encoded on multiple GPUs")
```

## Batch Processing

Process multiple data samples:

```python
from _qdp import QdpEngine
import numpy as np

engine = QdpEngine(0)

# Prepare batch of data
batch_size = 10
data_size = 16  # For 4 qubits

for i in range(batch_size):
    # Generate or load data
    data = np.random.rand(data_size)
    
    # Encode
    qtensor = engine.encode(data, num_qubits=4, encoding_method="amplitude")
    
    # Use qtensor in your quantum algorithm
    print(f"Processed batch {i+1}/{batch_size}")
```

## Integration with Qumat Core

Combine QDP with Qumat for full quantum ML pipeline:

```python
import qumat.qdp as qdp
from qumat import QuMat

# Encode classical data
engine = qdp.QdpEngine(device_id=0)
data = [1.0, 2.0, 3.0, 4.0]
qtensor = engine.encode(data, num_qubits=2, encoding_method="amplitude")

# Use in quantum circuit
backend_config = {
    "backend_name": "qiskit",
    "backend_options": {
        "simulator_type": "aer_simulator",
        "shots": 1024,
    },
}
qumat = QuMat(backend_config)
qumat.create_empty_circuit(num_qubits=2)

# Apply quantum operations
qumat.apply_hadamard_gate(0)
qumat.apply_cnot_gate(0, 1)

# Execute
results = qumat.execute_circuit()
print(f"Results: {results}")
```

## More Examples

For more examples and benchmarks, check out:
- [Benchmark E2E](https://github.com/apache/mahout/blob/trunk/qdp/qdp-python/benchmark/benchmark_e2e.py)
- [Latency Benchmark](https://github.com/apache/mahout/blob/trunk/qdp/qdp-python/benchmark/benchmark_latency.py)
- [Throughput Benchmark](https://github.com/apache/mahout/blob/trunk/qdp/qdp-python/benchmark/benchmark_throughput.py)
