#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Pure PyTorch quantum state encoding functions.

Replaces the Rust+CUDA kernel stack with native PyTorch operations.
All functions return complex-valued PyTorch tensors on the specified device.

Supported encodings:
  - amplitude: Normalize input by L2 norm, pad to 2^n, cast to complex.
  - basis: One-hot state vector at the given index.
  - angle: Product state from per-qubit rotation angles.
  - iqp / iqp-z: Instantaneous Quantum Polynomial encoding with FWT.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

MAX_QUBITS = 30

_USE_COMPILE = True  # Set to False to disable torch.compile (debugging)


def _maybe_compile(fn):
    """Apply torch.compile(dynamic=True) if _USE_COMPILE is True."""
    if _USE_COMPILE:
        return torch.compile(fn, dynamic=True)
    return fn


def _complex_dtype(precision: str) -> torch.dtype:
    """Map precision string to complex dtype."""
    if precision == "float32":
        return torch.complex64
    elif precision == "float64":
        return torch.complex128
    raise ValueError(f"precision must be 'float32' or 'float64', got {precision!r}")


def _real_dtype(precision: str) -> torch.dtype:
    """Map precision string to real dtype."""
    if precision == "float32":
        return torch.float32
    elif precision == "float64":
        return torch.float64
    raise ValueError(f"precision must be 'float32' or 'float64', got {precision!r}")


def _validate_num_qubits(num_qubits: int) -> None:
    if not isinstance(num_qubits, int) or num_qubits < 1 or num_qubits > MAX_QUBITS:
        raise ValueError(
            f"num_qubits must be an integer in [1, {MAX_QUBITS}], got {num_qubits!r}"
        )


def _validate_finite(data: torch.Tensor) -> None:
    if not torch.isfinite(data).all():
        raise ValueError("Input data contains non-finite values (NaN or Inf)")


# ---------------------------------------------------------------------------
# Amplitude Encoding
# ---------------------------------------------------------------------------


def amplitude_encode(
    data: torch.Tensor,
    num_qubits: int,
    *,
    precision: str = "float32",
    device: torch.device | None = None,
) -> torch.Tensor:
    """Encode a 1-D real vector into a quantum state via amplitude encoding.

    The input is normalized by its L2 norm and zero-padded to length 2^num_qubits.

    Args:
        data: 1-D real tensor of length <= 2^num_qubits.
        num_qubits: Number of qubits (state vector length = 2^num_qubits).
        precision: "float32" or "float64".
        device: Target device. If None, uses data's device or CUDA if available.

    Returns:
        Complex tensor of shape (2^num_qubits,).
    """
    _validate_num_qubits(num_qubits)
    state_len = 1 << num_qubits
    cdtype = _complex_dtype(precision)

    if data.dim() != 1:
        raise ValueError(f"Expected 1-D tensor, got {data.dim()}-D")
    if data.shape[0] > state_len:
        raise ValueError(
            f"Input length {data.shape[0]} exceeds state vector length {state_len}"
        )
    if data.shape[0] == 0:
        raise ValueError("Input data must not be empty")

    if device is None:
        device = data.device
    data = data.to(device=device, dtype=_real_dtype(precision))
    _validate_finite(data)

    norm = torch.linalg.norm(data)
    if norm == 0:
        raise ValueError("Input data has zero L2 norm; cannot normalize")
    normalized = data / norm

    state = torch.zeros(state_len, dtype=cdtype, device=device)
    state[: data.shape[0]] = normalized.to(cdtype)
    return state


def _amplitude_batch_kernel(
    batch: torch.Tensor, state_len: int, rdtype: torch.dtype
) -> torch.Tensor:
    B, D = batch.shape
    norms = torch.linalg.norm(batch, dim=1, keepdim=True)
    normalized = batch / norms
    # F.pad: add 1 zero on last dim (imag), pad second-to-last to state_len
    return F.pad(normalized.unsqueeze(-1), (0, 1, 0, state_len - D))


_amplitude_batch_kernel_compiled = _maybe_compile(_amplitude_batch_kernel)


def amplitude_encode_batch(
    batch: torch.Tensor,
    num_qubits: int,
    *,
    precision: str = "float32",
    device: torch.device | None = None,
) -> torch.Tensor:
    """Batch amplitude encoding: (B, D) -> (B, 2^num_qubits) complex tensor.

    Each row is independently normalized by its L2 norm and zero-padded.

    Args:
        batch: 2-D real tensor of shape (B, D) where D <= 2^num_qubits.
        num_qubits: Number of qubits.
        precision: "float32" or "float64".
        device: Target device.

    Returns:
        Complex tensor of shape (B, 2^num_qubits).
    """
    _validate_num_qubits(num_qubits)
    state_len = 1 << num_qubits
    rdtype = _real_dtype(precision)

    if batch.dim() != 2:
        raise ValueError(f"Expected 2-D tensor, got {batch.dim()}-D")
    B, D = batch.shape
    if D > state_len:
        raise ValueError(f"Input dimension {D} exceeds state vector length {state_len}")
    if D == 0:
        raise ValueError("Input data must not be empty")

    if device is None:
        device = batch.device
    batch = batch.to(device=device, dtype=rdtype)
    _validate_finite(batch)

    norms = torch.linalg.norm(batch, dim=1, keepdim=True)
    if (norms == 0).any():
        raise ValueError("One or more rows have zero L2 norm; cannot normalize")

    return torch.view_as_complex(
        _amplitude_batch_kernel_compiled(batch, state_len, rdtype)
    )


# ---------------------------------------------------------------------------
# Basis Encoding
# ---------------------------------------------------------------------------


def basis_encode(
    index: int,
    num_qubits: int,
    *,
    precision: str = "float32",
    device: torch.device | None = None,
) -> torch.Tensor:
    """Encode a computational basis state |index>.

    Args:
        index: Basis state index, 0 <= index < 2^num_qubits.
        num_qubits: Number of qubits.
        precision: "float32" or "float64".
        device: Target device.

    Returns:
        Complex tensor of shape (2^num_qubits,) with state[index] = 1+0j.
    """
    _validate_num_qubits(num_qubits)
    state_len = 1 << num_qubits
    cdtype = _complex_dtype(precision)

    if not isinstance(index, int) or index < 0 or index >= state_len:
        raise ValueError(f"Basis index must be in [0, {state_len - 1}], got {index!r}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state = torch.zeros(state_len, dtype=cdtype, device=device)
    state[index] = 1.0
    return state


def _basis_batch_kernel(
    indices: torch.Tensor, state_len: int, rdtype: torch.dtype
) -> torch.Tensor:
    oh = F.one_hot(indices, state_len).to(rdtype)
    return F.pad(oh.unsqueeze(-1), (0, 1))


_basis_batch_kernel_compiled = _maybe_compile(_basis_batch_kernel)


def basis_encode_batch(
    indices: torch.Tensor,
    num_qubits: int,
    *,
    precision: str = "float32",
    device: torch.device | None = None,
) -> torch.Tensor:
    """Batch basis encoding: each index becomes a one-hot complex state vector.

    Args:
        indices: 1-D integer tensor of shape (B,), each in [0, 2^num_qubits).
        num_qubits: Number of qubits.
        precision: "float32" or "float64".
        device: Target device.

    Returns:
        Complex tensor of shape (B, 2^num_qubits).
    """
    _validate_num_qubits(num_qubits)
    state_len = 1 << num_qubits

    if indices.dim() != 1:
        raise ValueError(f"Expected 1-D index tensor, got {indices.dim()}-D")

    if device is None:
        device = indices.device
    indices = indices.to(device=device, dtype=torch.long)

    if (indices < 0).any() or (indices >= state_len).any():
        raise ValueError(f"All indices must be in [0, {state_len - 1}]")

    rdtype = _real_dtype(precision)
    return torch.view_as_complex(
        _basis_batch_kernel_compiled(indices, state_len, rdtype)
    )


# ---------------------------------------------------------------------------
# Angle Encoding
# ---------------------------------------------------------------------------


def angle_encode(
    angles: torch.Tensor,
    num_qubits: int,
    *,
    precision: str = "float32",
    device: torch.device | None = None,
) -> torch.Tensor:
    """Encode angles into a product state: |psi> = tensor_product(cos(t)|0> + sin(t)|1>).

    For each basis state |z>, the amplitude is the product of cos(t_k) or sin(t_k)
    depending on whether bit k of z is 0 or 1.

    Args:
        angles: 1-D tensor of length num_qubits (one angle per qubit).
        num_qubits: Number of qubits.
        precision: "float32" or "float64".
        device: Target device.

    Returns:
        Complex tensor of shape (2^num_qubits,).
    """
    _validate_num_qubits(num_qubits)
    state_len = 1 << num_qubits

    if angles.dim() != 1:
        raise ValueError(f"Expected 1-D tensor, got {angles.dim()}-D")
    if angles.shape[0] != num_qubits:
        raise ValueError(
            f"Angle encoding requires exactly {num_qubits} angles, got {angles.shape[0]}"
        )

    rdtype = _real_dtype(precision)

    if device is None:
        device = angles.device
    angles = angles.to(device=device, dtype=rdtype)
    _validate_finite(angles)

    # bits[z, k] = (z >> k) & 1 for all z in [0, 2^n)
    z = torch.arange(state_len, device=device, dtype=torch.long)
    bits = ((z.unsqueeze(1) >> torch.arange(num_qubits, device=device)) & 1).bool()

    cos_a = torch.cos(angles)
    sin_a = torch.sin(angles)
    factors = torch.where(bits, sin_a, cos_a)
    amplitudes = factors.prod(dim=1).to(rdtype)

    return torch.complex(amplitudes, torch.zeros_like(amplitudes))


def _angle_batch_kernel(
    angles_batch: torch.Tensor,
    num_qubits: int,
    state_len: int,
    rdtype: torch.dtype,
) -> torch.Tensor:
    z = torch.arange(state_len, device=angles_batch.device, dtype=torch.long)
    bits = (
        (z.unsqueeze(1) >> torch.arange(num_qubits, device=angles_batch.device)) & 1
    ).bool()
    cos_a = torch.cos(angles_batch)
    sin_a = torch.sin(angles_batch)
    factors = torch.where(bits.unsqueeze(0), sin_a.unsqueeze(1), cos_a.unsqueeze(1))
    amplitudes = factors.prod(dim=2).to(rdtype)
    return F.pad(amplitudes.unsqueeze(-1), (0, 1))


_angle_batch_kernel_compiled = _maybe_compile(_angle_batch_kernel)


def angle_encode_batch(
    angles_batch: torch.Tensor,
    num_qubits: int,
    *,
    precision: str = "float32",
    device: torch.device | None = None,
) -> torch.Tensor:
    """Batch angle encoding: (B, num_qubits) -> (B, 2^num_qubits) complex tensor.

    Args:
        angles_batch: 2-D tensor of shape (B, num_qubits).
        num_qubits: Number of qubits.
        precision: "float32" or "float64".
        device: Target device.

    Returns:
        Complex tensor of shape (B, 2^num_qubits).
    """
    _validate_num_qubits(num_qubits)
    state_len = 1 << num_qubits

    if angles_batch.dim() != 2:
        raise ValueError(f"Expected 2-D tensor, got {angles_batch.dim()}-D")
    _, D = angles_batch.shape
    if D != num_qubits:
        raise ValueError(
            f"Angle encoding requires {num_qubits} angles per sample, got {D}"
        )

    rdtype = _real_dtype(precision)

    if device is None:
        device = angles_batch.device
    angles_batch = angles_batch.to(device=device, dtype=rdtype)
    _validate_finite(angles_batch)

    return torch.view_as_complex(
        _angle_batch_kernel_compiled(angles_batch, num_qubits, state_len, rdtype)
    )


# ---------------------------------------------------------------------------
# IQP Encoding
# ---------------------------------------------------------------------------


def _fwht_inplace(state: torch.Tensor) -> torch.Tensor:
    """Fast Walsh-Hadamard Transform (in-place, iterative butterfly).

    Operates on the last dimension of state. Supports batched input.

    Args:
        state: Real or complex tensor with last dimension being a power of 2.

    Returns:
        The transformed tensor (same object, modified in-place).
    """
    n = state.shape[-1]
    num_qubits = int(math.log2(n))
    assert 1 << num_qubits == n, f"Last dimension must be a power of 2, got {n}"

    for stage in range(num_qubits):
        stride = 1 << stage
        # For each butterfly pair at this stage
        # Reshape to isolate butterfly pairs
        # Group elements into pairs separated by stride
        shape = list(state.shape[:-1]) + [n // (2 * stride), 2, stride]
        view = state.view(*shape)
        a = view[..., 0, :].clone()
        b = view[..., 1, :].clone()
        view[..., 0, :] = a + b
        view[..., 1, :] = a - b

    return state


def iqp_encode(
    data: torch.Tensor,
    num_qubits: int,
    *,
    enable_zz: bool = True,
    precision: str = "float32",
    device: torch.device | None = None,
) -> torch.Tensor:
    """IQP (Instantaneous Quantum Polynomial) encoding.

    Circuit: H^n . U_phase(x) . H^n |0>^n

    For full IQP (enable_zz=True), data has n + n(n-1)/2 parameters (Z + ZZ terms).
    For Z-only (enable_zz=False), data has n parameters.

    Args:
        data: 1-D tensor of encoding parameters.
        num_qubits: Number of qubits.
        enable_zz: Whether to include ZZ interaction terms.
        precision: "float32" or "float64".
        device: Target device.

    Returns:
        Complex tensor of shape (2^num_qubits,).
    """
    _validate_num_qubits(num_qubits)
    state_len = 1 << num_qubits
    n = num_qubits

    if enable_zz:
        expected_len = n + n * (n - 1) // 2
    else:
        expected_len = n

    if data.dim() != 1:
        raise ValueError(f"Expected 1-D tensor, got {data.dim()}-D")
    if data.shape[0] != expected_len:
        raise ValueError(
            f"IQP encoding ({'full' if enable_zz else 'z-only'}) requires "
            f"{expected_len} parameters for {n} qubits, got {data.shape[0]}"
        )

    rdtype = _real_dtype(precision)

    if device is None:
        device = data.device
    data = data.to(device=device, dtype=rdtype)
    _validate_finite(data)

    z = torch.arange(state_len, device=device, dtype=torch.long)
    bits = ((z.unsqueeze(1) >> torch.arange(n, device=device)) & 1).to(rdtype)
    # bits: (state_len, n)

    # Z terms: sum_i bits[z,i] * data[i]
    phase = bits @ data[:n]

    # ZZ terms: sum_{i<j} bits[z,i] * bits[z,j] * data[n + pair_idx]
    if enable_zz:
        pair_idx = n
        for i in range(n):
            for j in range(i + 1, n):
                phase = phase + bits[:, i] * bits[:, j] * data[pair_idx]
                pair_idx += 1

    # exp(i * phase) -- diagonal unitary
    state = torch.polar(torch.ones(state_len, dtype=rdtype, device=device), phase)

    # Apply H^n via Fast Walsh-Hadamard Transform
    state = _fwht_inplace(state)

    # Normalize by 1/2^n (Hadamard normalization)
    state = state / state_len

    return state


def _fwht_real_out_of_place(state: torch.Tensor, num_qubits: int) -> torch.Tensor:
    """Out-of-place FWHT on real tensors, suitable for torch.compile."""
    for stage in range(num_qubits):
        stride = 1 << stage
        n = state.shape[-1]
        shape = list(state.shape[:-1]) + [n // (2 * stride), 2, stride]
        view = state.view(*shape)
        a = view[..., 0, :]
        b = view[..., 1, :]
        new_view = torch.stack([a + b, a - b], dim=-2)
        state = new_view.view(*state.shape)
    return state


def _iqp_batch_kernel(
    data_batch: torch.Tensor,
    num_qubits: int,
    state_len: int,
    enable_zz: bool,
    rdtype: torch.dtype,
) -> torch.Tensor:
    n = num_qubits
    z = torch.arange(state_len, device=data_batch.device, dtype=torch.long)
    bits = ((z.unsqueeze(1) >> torch.arange(n, device=data_batch.device)) & 1).to(
        rdtype
    )

    phase = (bits @ data_batch[:, :n].T).T  # (B, state_len)

    if enable_zz:
        pair_idx = n
        for i in range(n):
            for j in range(i + 1, n):
                zz_bits = bits[:, i] * bits[:, j]
                phase = phase + data_batch[:, pair_idx].unsqueeze(
                    1
                ) * zz_bits.unsqueeze(0)
                pair_idx += 1

    # Replace torch.polar with cos/sin (both real)
    real_part = torch.cos(phase)
    imag_part = torch.sin(phase)
    # FWHT on real tensors separately
    real_part = _fwht_real_out_of_place(real_part, num_qubits)
    imag_part = _fwht_real_out_of_place(imag_part, num_qubits)
    # Normalize
    real_part = real_part / state_len
    imag_part = imag_part / state_len
    return torch.stack([real_part, imag_part], dim=-1)


_iqp_batch_kernel_compiled = _maybe_compile(_iqp_batch_kernel)


def iqp_encode_batch(
    data_batch: torch.Tensor,
    num_qubits: int,
    *,
    enable_zz: bool = True,
    precision: str = "float32",
    device: torch.device | None = None,
) -> torch.Tensor:
    """Batch IQP encoding: (B, params) -> (B, 2^num_qubits) complex tensor.

    Args:
        data_batch: 2-D tensor of shape (B, params).
        num_qubits: Number of qubits.
        enable_zz: Whether to include ZZ interaction terms.
        precision: "float32" or "float64".
        device: Target device.

    Returns:
        Complex tensor of shape (B, 2^num_qubits).
    """
    _validate_num_qubits(num_qubits)
    state_len = 1 << num_qubits
    n = num_qubits

    if enable_zz:
        expected_len = n + n * (n - 1) // 2
    else:
        expected_len = n

    if data_batch.dim() != 2:
        raise ValueError(f"Expected 2-D tensor, got {data_batch.dim()}-D")
    B, D = data_batch.shape
    if D != expected_len:
        raise ValueError(
            f"IQP encoding ({'full' if enable_zz else 'z-only'}) requires "
            f"{expected_len} parameters for {n} qubits, got {D}"
        )

    rdtype = _real_dtype(precision)

    if device is None:
        device = data_batch.device
    data_batch = data_batch.to(device=device, dtype=rdtype)
    _validate_finite(data_batch)

    return torch.view_as_complex(
        _iqp_batch_kernel_compiled(data_batch, num_qubits, state_len, enable_zz, rdtype)
    )


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def encode(
    data: torch.Tensor,
    num_qubits: int,
    encoding_method: str = "amplitude",
    *,
    precision: str = "float32",
    device: torch.device | None = None,
) -> torch.Tensor:
    """Dispatch to the appropriate encoding function (single sample).

    Args:
        data: Input data tensor.
        num_qubits: Number of qubits.
        encoding_method: One of "amplitude", "basis", "angle", "iqp", "iqp-z".
        precision: "float32" or "float64".
        device: Target device.

    Returns:
        Complex tensor of shape (2^num_qubits,).
    """
    if encoding_method == "amplitude":
        return amplitude_encode(data, num_qubits, precision=precision, device=device)
    elif encoding_method == "basis":
        if data.dim() != 1 or data.shape[0] != 1:
            raise ValueError("Basis encoding requires exactly 1 element (the index)")
        return basis_encode(
            int(data[0].item()), num_qubits, precision=precision, device=device
        )
    elif encoding_method == "angle":
        return angle_encode(data, num_qubits, precision=precision, device=device)
    elif encoding_method == "iqp":
        return iqp_encode(
            data, num_qubits, enable_zz=True, precision=precision, device=device
        )
    elif encoding_method == "iqp-z":
        return iqp_encode(
            data, num_qubits, enable_zz=False, precision=precision, device=device
        )
    else:
        raise ValueError(
            f"Unknown encoding method {encoding_method!r}. "
            f"Supported: 'amplitude', 'basis', 'angle', 'iqp', 'iqp-z'"
        )


def encode_batch(
    batch: torch.Tensor,
    num_qubits: int,
    encoding_method: str = "amplitude",
    *,
    precision: str = "float32",
    device: torch.device | None = None,
) -> torch.Tensor:
    """Dispatch to the appropriate batch encoding function.

    Args:
        batch: 2-D input tensor of shape (B, sample_size).
        num_qubits: Number of qubits.
        encoding_method: One of "amplitude", "basis", "angle", "iqp", "iqp-z".
        precision: "float32" or "float64".
        device: Target device.

    Returns:
        Complex tensor of shape (B, 2^num_qubits).
    """
    if encoding_method == "amplitude":
        return amplitude_encode_batch(
            batch, num_qubits, precision=precision, device=device
        )
    elif encoding_method == "basis":
        if batch.dim() != 2 or batch.shape[1] != 1:
            raise ValueError("Basis batch encoding requires shape (B, 1)")
        indices = batch[:, 0].long()
        return basis_encode_batch(
            indices, num_qubits, precision=precision, device=device
        )
    elif encoding_method == "angle":
        return angle_encode_batch(batch, num_qubits, precision=precision, device=device)
    elif encoding_method == "iqp":
        return iqp_encode_batch(
            batch, num_qubits, enable_zz=True, precision=precision, device=device
        )
    elif encoding_method == "iqp-z":
        return iqp_encode_batch(
            batch, num_qubits, enable_zz=False, precision=precision, device=device
        )
    else:
        raise ValueError(
            f"Unknown encoding method {encoding_method!r}. "
            f"Supported: 'amplitude', 'basis', 'angle', 'iqp', 'iqp-z'"
        )
