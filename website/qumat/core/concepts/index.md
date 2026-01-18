---
layout: page
title: Core Concepts - Qumat Core
---

# Core Concepts

Understanding these key concepts will help you use Qumat Core effectively.

## Backend Abstraction

Qumat Core provides a unified interface to multiple quantum computing backends. This means you can write your quantum circuit once and execute it on different platforms without changing your code.

### Supported Backends

- **Qiskit**: IBM's quantum computing framework
- **Cirq**: Google's quantum computing framework
- **Amazon Braket**: AWS quantum computing service

### Backend Configuration

Each backend is configured using a dictionary:

```python
backend_config = {
    "backend_name": "qiskit",  # or "cirq", "amazon_braket"
    "backend_options": {
        "simulator_type": "aer_simulator",
        "shots": 1024,
    },
}
```

## Quantum Circuits

A quantum circuit is a sequence of quantum gates applied to qubits. In Qumat, you create circuits using:

1. **Initialize**: Create a QuMat instance with backend configuration
2. **Create Circuit**: Call `create_empty_circuit(num_qubits)` 
3. **Apply Gates**: Use gate methods to build your circuit
4. **Execute**: Run `execute_circuit()` to get results

### Circuit Lifecycle

```python
qumat = QuMat(backend_config)              # 1. Initialize
qumat.create_empty_circuit(num_qubits=2)   # 2. Create
qumat.apply_hadamard_gate(0)               # 3. Apply gates
results = qumat.execute_circuit()          # 4. Execute
```

## Quantum Gates

Qumat supports standard quantum gates that work across all backends:

### Single-Qubit Gates

- **Hadamard (H)**: Creates superposition
- **Pauli-X**: Quantum NOT gate
- **Pauli-Y**: Rotation around Y-axis
- **Pauli-Z**: Phase flip

### Two-Qubit Gates

- **CNOT**: Controlled-NOT gate for entanglement
- **CZ**: Controlled-Z gate
- **SWAP**: Swaps two qubits

### Example

```python
qumat.apply_hadamard_gate(qubit_index=0)
qumat.apply_pauli_x_gate(qubit_index=1)
qumat.apply_cnot_gate(control_qubit_index=0, target_qubit_index=1)
```

## Measurements

After executing a circuit, you receive measurement results showing the probability distribution of quantum states:

```python
results = qumat.execute_circuit()
# Results show counts of each measurement outcome
# Example: {'00': 512, '11': 512} for a Bell state with 1024 shots
```

## Write Once, Run Anywhere

The key benefit of Qumat Core is portability. The same quantum circuit code runs on any supported backend:

```python
# Same code, different backend
for backend in ["qiskit", "cirq", "amazon_braket"]:
    config = {"backend_name": backend, "backend_options": {...}}
    qumat = QuMat(config)
    qumat.create_empty_circuit(2)
    qumat.apply_hadamard_gate(0)
    qumat.apply_cnot_gate(0, 1)
    results = qumat.execute_circuit()
```

## Error Handling

Qumat validates operations and provides clear error messages:

- Circuit must be created before applying gates
- Qubit indices must be within circuit bounds
- Backend configuration must be valid

```python
# This will raise RuntimeError
qumat = QuMat(backend_config)
qumat.apply_hadamard_gate(0)  # Error: circuit not initialized

# Correct approach
qumat.create_empty_circuit(2)
qumat.apply_hadamard_gate(0)  # OK
```

## Next Steps

- Explore the [API Reference](/qumat/core/api) for detailed method documentation
- Try the [Examples](/qumat/core/examples) to see concepts in action
- Read about [Basic Gates](https://github.com/apache/mahout/blob/trunk/docs/basic_gates.md)
