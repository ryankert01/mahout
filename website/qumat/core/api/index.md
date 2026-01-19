---
layout: page
title: API Reference - Qumat Core
---

# API Reference

Complete API reference for Qumat Core.

## QuMat Class

The main class for creating and executing quantum circuits.

### Constructor

```python
QuMat(backend_config: dict)
```

Creates a QuMat instance with the specified backend configuration.

**Parameters:**
- `backend_config` (dict): Configuration dictionary with:
  - `backend_name` (str): Name of backend ("qiskit", "cirq", "amazon_braket")
  - `backend_options` (dict): Backend-specific options including `simulator_type` and `shots`

**Raises:**
- `ValueError`: If backend_config is not a dictionary
- `KeyError`: If required configuration keys are missing
- `ImportError`: If the backend module cannot be imported

**Example:**
```python
backend_config = {
    "backend_name": "qiskit",
    "backend_options": {
        "simulator_type": "aer_simulator",
        "shots": 1024,
    },
}
qumat = QuMat(backend_config)
```

### Methods

#### create_empty_circuit

```python
create_empty_circuit(num_qubits: int | None = None)
```

Creates an empty quantum circuit with the specified number of qubits. Must be called before applying gates.

**Parameters:**
- `num_qubits` (int | None): Number of qubits in the circuit. If None, creates a circuit without pre-allocated qubits.

**Example:**
```python
qumat.create_empty_circuit(num_qubits=2)
```

#### apply_hadamard_gate

```python
apply_hadamard_gate(qubit_index: int)
```

Applies a Hadamard gate to the specified qubit. Creates superposition.

**Parameters:**
- `qubit_index` (int): Index of the qubit to apply the gate to

**Raises:**
- `RuntimeError`: If circuit is not initialized
- `ValueError`: If qubit_index is out of bounds

#### apply_pauli_x_gate

```python
apply_pauli_x_gate(qubit_index: int)
```

Applies a Pauli-X (NOT) gate to the specified qubit.

**Parameters:**
- `qubit_index` (int): Index of the qubit

#### apply_pauli_y_gate

```python
apply_pauli_y_gate(qubit_index: int)
```

Applies a Pauli-Y gate to the specified qubit.

**Parameters:**
- `qubit_index` (int): Index of the qubit

#### apply_pauli_z_gate

```python
apply_pauli_z_gate(qubit_index: int)
```

Applies a Pauli-Z gate to the specified qubit. Applies phase flip.

**Parameters:**
- `qubit_index` (int): Index of the qubit

#### apply_cnot_gate

```python
apply_cnot_gate(control_qubit_index: int, target_qubit_index: int)
```

Applies a CNOT (Controlled-NOT) gate. Flips target qubit if control is |1⟩.

**Parameters:**
- `control_qubit_index` (int): Index of the control qubit
- `target_qubit_index` (int): Index of the target qubit

**Raises:**
- `RuntimeError`: If circuit is not initialized
- `ValueError`: If qubit indices are out of bounds or equal

**Example:**
```python
qumat.apply_cnot_gate(control_qubit_index=0, target_qubit_index=1)
```

#### execute_circuit

```python
execute_circuit() -> dict
```

Executes the quantum circuit and returns measurement results.

**Returns:**
- dict: Measurement results showing counts for each quantum state

**Example:**
```python
results = qumat.execute_circuit()
print(results)  # e.g., {'00': 512, '11': 512}
```

## Backend-Specific Options

### Qiskit Backend

```python
backend_options = {
    "simulator_type": "aer_simulator",  # or other Qiskit simulators
    "shots": 1024,
}
```

### Cirq Backend

```python
backend_options = {
    "simulator_type": "simulator",  # Cirq simulator
    "shots": 1024,
}
```

### Amazon Braket Backend

```python
backend_options = {
    "simulator_type": "local_simulator",  # or Braket cloud simulators
    "shots": 1024,
}
```

## See Also

- [Getting Started Guide](/qumat/core/getting-started)
- [Core Concepts](/qumat/core/concepts)
- [Examples](/qumat/core/examples)
- [Source Code on GitHub](https://github.com/apache/mahout/blob/trunk/qumat/qumat.py)
