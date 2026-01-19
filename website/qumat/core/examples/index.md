---
layout: page
title: Examples - Qumat Core
---

# Examples

Learn by example! Here are some practical examples showing how to use Qumat Core for various quantum computing tasks.

## Basic Bell State

Create an entangled Bell state using Hadamard and CNOT gates:

```python
from qumat import QuMat

backend_config = {
    "backend_name": "qiskit",
    "backend_options": {
        "simulator_type": "aer_simulator",
        "shots": 1024,
    },
}

qumat = QuMat(backend_config)
qumat.create_empty_circuit(num_qubits=2)

# Create Bell state: |Φ+⟩ = (|00⟩ + |11⟩) / √2
qumat.apply_hadamard_gate(qubit_index=0)
qumat.apply_cnot_gate(control_qubit_index=0, target_qubit_index=1)

results = qumat.execute_circuit()
print(results)
# Expected: approximately 50% |00⟩ and 50% |11⟩
```

## Quantum Teleportation

Implement the quantum teleportation protocol:

```python
from qumat import QuMat

backend_config = {
    "backend_name": "qiskit",
    "backend_options": {
        "simulator_type": "aer_simulator",
        "shots": 1,
    },
}

quantum_computer = QuMat(backend_config)
quantum_computer.create_empty_circuit(3)

# Step 1: Create entanglement between qubits 1 and 2
quantum_computer.apply_hadamard_gate(1)
quantum_computer.apply_cnot_gate(1, 2)

# Step 2: Prepare the state to be teleported on qubit 0
quantum_computer.apply_hadamard_gate(0)
quantum_computer.apply_pauli_z_gate(0)

# Step 3: Perform Bell measurement on qubits 0 and 1
quantum_computer.apply_cnot_gate(0, 1)
quantum_computer.apply_hadamard_gate(0)

# Measure qubits 0 and 1
quantum_computer.circuit.measure([0, 1], [0, 1])

# Execute
sender_results = quantum_computer.execute_circuit()
print("Measurement results:", sender_results)
```

## Multi-Backend Example

Run the same circuit on different backends:

```python
from qumat import QuMat

def run_on_backend(backend_name, simulator_type):
    backend_config = {
        "backend_name": backend_name,
        "backend_options": {
            "simulator_type": simulator_type,
            "shots": 1024,
        },
    }
    
    qumat = QuMat(backend_config)
    qumat.create_empty_circuit(num_qubits=2)
    qumat.apply_hadamard_gate(0)
    qumat.apply_cnot_gate(0, 1)
    qumat.apply_pauli_x_gate(0)
    
    return qumat.execute_circuit()

# Run on Qiskit
qiskit_results = run_on_backend("qiskit", "aer_simulator")
print(f"Qiskit results: {qiskit_results}")

# Run on Cirq
cirq_results = run_on_backend("cirq", "simulator")
print(f"Cirq results: {cirq_results}")

# Run on Amazon Braket
braket_results = run_on_backend("amazon_braket", "local_simulator")
print(f"Braket results: {braket_results}")
```

## Applying Multiple Gates

Example showing various gate applications:

```python
from qumat import QuMat

backend_config = {
    "backend_name": "qiskit",
    "backend_options": {
        "simulator_type": "aer_simulator",
        "shots": 1024,
    },
}

qumat = QuMat(backend_config)
qumat.create_empty_circuit(num_qubits=3)

# Apply different gates
qumat.apply_hadamard_gate(0)
qumat.apply_pauli_x_gate(1)
qumat.apply_pauli_y_gate(1)
qumat.apply_pauli_z_gate(2)
qumat.apply_cnot_gate(0, 1)
qumat.apply_cnot_gate(1, 2)

results = qumat.execute_circuit()
print(results)
```

## More Examples

For more examples, check out:
- [Simple Example](https://github.com/apache/mahout/blob/trunk/examples/simple_example.py)
- [Quantum Teleportation](https://github.com/apache/mahout/blob/trunk/examples/quantum_teleportation.py)
- [Jupyter Notebooks](https://github.com/apache/mahout/tree/trunk/examples)
