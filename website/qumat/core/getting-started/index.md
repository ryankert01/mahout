---
layout: page
title: Getting Started with Qumat Core
---

# Getting Started with Qumat Core

Qumat Core provides a unified API for building quantum circuits across multiple backends (Qiskit, Cirq, Amazon Braket). This guide will help you get started with creating and executing quantum circuits.

## Installation

```bash
git clone https://github.com/apache/mahout.git
cd mahout
pip install uv
uv sync  # Install Qumat Core dependencies
```

## Quick Example

Here's a simple example that creates a 2-qubit circuit with a Hadamard gate and CNOT gate:

```python
from qumat import QuMat

# Configure your backend
backend_config = {
    "backend_name": "qiskit",
    "backend_options": {
        "simulator_type": "aer_simulator",
        "shots": 1024,
    },
}

# Create QuMat instance
qumat = QuMat(backend_config)

# Create a quantum circuit with 2 qubits
qumat.create_empty_circuit(num_qubits=2)

# Apply quantum gates
qumat.apply_hadamard_gate(qubit_index=0)
qumat.apply_cnot_gate(control_qubit_index=0, target_qubit_index=1)

# Execute the circuit
results = qumat.execute_circuit()
print(results)
```

## Supported Backends

Qumat supports three major quantum computing backends:

### Qiskit

```python
backend_config = {
    "backend_name": "qiskit",
    "backend_options": {
        "simulator_type": "aer_simulator",
        "shots": 1024,
    },
}
```

### Cirq

```python
backend_config = {
    "backend_name": "cirq",
    "backend_options": {
        "simulator_type": "simulator",
        "shots": 1024,
    },
}
```

### Amazon Braket

```python
backend_config = {
    "backend_name": "amazon_braket",
    "backend_options": {
        "simulator_type": "local_simulator",
        "shots": 1024,
    },
}
```

## Basic Workflow

1. **Initialize QuMat**: Create a QuMat instance with your chosen backend configuration
2. **Create Circuit**: Call `create_empty_circuit()` with the number of qubits
3. **Apply Gates**: Use methods like `apply_hadamard_gate()`, `apply_cnot_gate()`, etc.
4. **Execute**: Call `execute_circuit()` to run the circuit and get results

## Next Steps

- Learn about [Core Concepts](/qumat/core/concepts)
- Explore the [API Reference](/qumat/core/api)
- Check out more [Examples](/qumat/core/examples)
