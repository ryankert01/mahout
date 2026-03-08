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

"""Tests for PyTorch-native quantum encoding functions."""

from __future__ import annotations

import math

import pytest
import torch
from qumat_qdp.encodings import (
    amplitude_encode,
    amplitude_encode_batch,
    angle_encode,
    angle_encode_batch,
    basis_encode,
    basis_encode_batch,
    encode,
    encode_batch,
    iqp_encode,
    iqp_encode_batch,
)

# Tolerance for floating point comparisons
ATOL = 1e-6
RTOL = 1e-5


# ============================================================================
# Amplitude Encoding Tests
# ============================================================================


class TestAmplitudeEncode:
    def test_basic_normalization(self):
        data = torch.tensor([1.0, 2.0, 3.0, 4.0])
        state = amplitude_encode(data, num_qubits=2)
        assert state.shape == (4,)
        assert state.dtype == torch.complex64
        # Check normalization: sum of |amplitudes|^2 == 1
        assert torch.allclose(
            torch.sum(torch.abs(state) ** 2), torch.tensor(1.0), atol=ATOL
        )

    def test_padding(self):
        data = torch.tensor([1.0, 2.0])
        state = amplitude_encode(data, num_qubits=2)
        assert state.shape == (4,)
        # Last 2 entries should be zero
        assert state[2] == 0 + 0j
        assert state[3] == 0 + 0j

    def test_already_normalized(self):
        data = torch.tensor([1.0, 0.0, 0.0, 0.0])
        state = amplitude_encode(data, num_qubits=2)
        assert torch.allclose(state[0].real, torch.tensor(1.0), atol=ATOL)

    def test_precision_float64(self):
        data = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
        state = amplitude_encode(data, num_qubits=2, precision="float64")
        assert state.dtype == torch.complex128

    def test_zero_norm_raises(self):
        data = torch.zeros(4)
        with pytest.raises(ValueError, match="zero L2 norm"):
            amplitude_encode(data, num_qubits=2)

    def test_nan_raises(self):
        data = torch.tensor([1.0, float("nan"), 3.0, 4.0])
        with pytest.raises(ValueError, match="non-finite"):
            amplitude_encode(data, num_qubits=2)

    def test_input_too_large_raises(self):
        data = torch.randn(5)
        with pytest.raises(ValueError, match="exceeds"):
            amplitude_encode(data, num_qubits=2)

    def test_empty_raises(self):
        data = torch.tensor([])
        with pytest.raises(ValueError, match="empty"):
            amplitude_encode(data, num_qubits=2)

    def test_2d_raises(self):
        data = torch.randn(2, 4)
        with pytest.raises(ValueError, match="1-D"):
            amplitude_encode(data, num_qubits=2)


class TestAmplitudeEncodeBatch:
    def test_basic_batch(self):
        batch = torch.randn(3, 4)
        state = amplitude_encode_batch(batch, num_qubits=2)
        assert state.shape == (3, 4)
        assert state.dtype == torch.complex64
        # Each row should be normalized
        norms = torch.sum(torch.abs(state) ** 2, dim=1)
        assert torch.allclose(norms, torch.ones(3), atol=ATOL)

    def test_padding(self):
        batch = torch.randn(2, 2)
        state = amplitude_encode_batch(batch, num_qubits=2)
        assert state.shape == (2, 4)
        # Padded entries should be zero
        assert torch.allclose(state[:, 2:], torch.zeros(2, 2, dtype=torch.complex64))

    def test_zero_row_raises(self):
        batch = torch.tensor([[1.0, 2.0], [0.0, 0.0]])
        with pytest.raises(ValueError, match="zero L2 norm"):
            amplitude_encode_batch(batch, num_qubits=2)


# ============================================================================
# Basis Encoding Tests
# ============================================================================


class TestBasisEncode:
    def test_basic(self):
        state = basis_encode(0, num_qubits=2)
        assert state.shape == (4,)
        assert state[0] == 1.0 + 0j
        assert state[1] == 0.0 + 0j
        assert state[2] == 0.0 + 0j
        assert state[3] == 0.0 + 0j

    def test_last_index(self):
        state = basis_encode(3, num_qubits=2)
        assert state[3] == 1.0 + 0j
        assert torch.sum(torch.abs(state) ** 2).item() == pytest.approx(1.0)

    def test_out_of_range_raises(self):
        with pytest.raises(ValueError, match="Basis index"):
            basis_encode(4, num_qubits=2)

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="Basis index"):
            basis_encode(-1, num_qubits=2)


class TestBasisEncodeBatch:
    def test_basic_batch(self):
        indices = torch.tensor([0, 1, 2, 3])
        state = basis_encode_batch(indices, num_qubits=2)
        assert state.shape == (4, 4)
        # Each row should be one-hot
        for i in range(4):
            assert state[i, i] == 1.0 + 0j
            assert torch.sum(torch.abs(state[i]) ** 2).item() == pytest.approx(1.0)

    def test_out_of_range_raises(self):
        indices = torch.tensor([0, 4])
        with pytest.raises(ValueError, match="indices"):
            basis_encode_batch(indices, num_qubits=2)


# ============================================================================
# Angle Encoding Tests
# ============================================================================


class TestAngleEncode:
    def test_zero_angles(self):
        """All angles = 0 should give |00...0> state (all cos(0)=1, product=1)."""
        angles = torch.zeros(2)
        state = angle_encode(angles, num_qubits=2)
        assert state.shape == (4,)
        # |00> = state[0] = cos(0)*cos(0) = 1
        assert state[0].real.item() == pytest.approx(1.0, abs=ATOL)

    def test_normalization(self):
        """Product state should be normalized."""
        angles = torch.tensor([0.5, 1.0])
        state = angle_encode(angles, num_qubits=2)
        norm_sq = torch.sum(torch.abs(state) ** 2).item()
        assert norm_sq == pytest.approx(1.0, abs=ATOL)

    def test_single_qubit(self):
        """Single qubit: cos(t)|0> + sin(t)|1>."""
        t = torch.tensor([math.pi / 4])
        state = angle_encode(t, num_qubits=1)
        assert state.shape == (2,)
        assert state[0].real.item() == pytest.approx(math.cos(math.pi / 4), abs=ATOL)
        assert state[1].real.item() == pytest.approx(math.sin(math.pi / 4), abs=ATOL)

    def test_wrong_size_raises(self):
        angles = torch.tensor([0.5])
        with pytest.raises(ValueError, match="exactly 2 angles"):
            angle_encode(angles, num_qubits=2)

    def test_nan_raises(self):
        angles = torch.tensor([float("nan"), 0.5])
        with pytest.raises(ValueError, match="non-finite"):
            angle_encode(angles, num_qubits=2)


class TestAngleEncodeBatch:
    def test_basic_batch(self):
        batch = torch.randn(5, 3)
        state = angle_encode_batch(batch, num_qubits=3)
        assert state.shape == (5, 8)
        # Each row should be normalized
        norms = torch.sum(torch.abs(state) ** 2, dim=1)
        assert torch.allclose(norms, torch.ones(5), atol=ATOL)


# ============================================================================
# IQP Encoding Tests
# ============================================================================


class TestIqpEncode:
    def test_z_only_basic(self):
        """IQP z-only with 2 qubits."""
        data = torch.tensor([0.5, 1.0])
        state = iqp_encode(data, num_qubits=2, enable_zz=False)
        assert state.shape == (4,)
        # Should be normalized
        norm_sq = torch.sum(torch.abs(state) ** 2).item()
        assert norm_sq == pytest.approx(1.0, abs=ATOL)

    def test_full_iqp_data_length(self):
        """Full IQP with 3 qubits needs 3 + 3 = 6 parameters."""
        data = torch.randn(6)
        state = iqp_encode(data, num_qubits=3, enable_zz=True)
        assert state.shape == (8,)
        norm_sq = torch.sum(torch.abs(state) ** 2).item()
        assert norm_sq == pytest.approx(1.0, abs=ATOL)

    def test_wrong_data_length_raises(self):
        data = torch.randn(4)  # 2 qubits full IQP needs 2+1=3, not 4
        with pytest.raises(ValueError, match="requires"):
            iqp_encode(data, num_qubits=2, enable_zz=True)

    def test_z_only_wrong_length_raises(self):
        data = torch.randn(3)
        with pytest.raises(ValueError, match="requires"):
            iqp_encode(data, num_qubits=2, enable_zz=False)

    def test_uniform_phase_gives_known_state(self):
        """With all zeros, exp(i*0) = 1 everywhere, FWT of all-ones is [N, 0, 0, ...].
        After normalization: [1, 0, 0, ...]."""
        data = torch.zeros(2)
        state = iqp_encode(data, num_qubits=2, enable_zz=False)
        # H^2 applied to all-ones: only |00> component is nonzero
        assert abs(state[0].abs().item() - 1.0) < ATOL
        for i in range(1, 4):
            assert state[i].abs().item() < ATOL


class TestIqpEncodeBatch:
    def test_basic_batch(self):
        """Batch IQP encoding with 2 qubits, z-only."""
        batch = torch.randn(4, 2)
        state = iqp_encode_batch(batch, num_qubits=2, enable_zz=False)
        assert state.shape == (4, 4)
        norms = torch.sum(torch.abs(state) ** 2, dim=1)
        assert torch.allclose(norms, torch.ones(4), atol=ATOL)

    def test_full_batch(self):
        """Batch IQP full encoding with 2 qubits (2 + 1 = 3 params)."""
        batch = torch.randn(3, 3)
        state = iqp_encode_batch(batch, num_qubits=2, enable_zz=True)
        assert state.shape == (3, 4)
        norms = torch.sum(torch.abs(state) ** 2, dim=1)
        assert torch.allclose(norms, torch.ones(3), atol=ATOL)


# ============================================================================
# Dispatcher Tests
# ============================================================================


class TestDispatcher:
    def test_encode_amplitude(self):
        data = torch.randn(4)
        state = encode(data, num_qubits=2, encoding_method="amplitude")
        assert state.shape == (4,)

    def test_encode_basis(self):
        data = torch.tensor([2.0])
        state = encode(data, num_qubits=2, encoding_method="basis")
        assert state[2] == 1.0 + 0j

    def test_encode_angle(self):
        data = torch.randn(3)
        state = encode(data, num_qubits=3, encoding_method="angle")
        assert state.shape == (8,)

    def test_encode_iqp(self):
        data = torch.randn(3)  # 2 + 1 params for 2 qubits
        state = encode(data, num_qubits=2, encoding_method="iqp")
        assert state.shape == (4,)

    def test_encode_iqp_z(self):
        data = torch.randn(2)
        state = encode(data, num_qubits=2, encoding_method="iqp-z")
        assert state.shape == (4,)

    def test_unknown_encoding_raises(self):
        data = torch.randn(4)
        with pytest.raises(ValueError, match="Unknown encoding"):
            encode(data, num_qubits=2, encoding_method="unknown")

    def test_encode_batch_amplitude(self):
        batch = torch.randn(5, 4)
        state = encode_batch(batch, num_qubits=2, encoding_method="amplitude")
        assert state.shape == (5, 4)

    def test_encode_batch_basis(self):
        batch = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
        state = encode_batch(batch, num_qubits=2, encoding_method="basis")
        assert state.shape == (4, 4)


# ============================================================================
# Qubit Validation Tests
# ============================================================================


class TestQubitValidation:
    def test_zero_qubits_raises(self):
        with pytest.raises(ValueError, match="num_qubits"):
            amplitude_encode(torch.tensor([1.0]), num_qubits=0)

    def test_negative_qubits_raises(self):
        with pytest.raises(ValueError, match="num_qubits"):
            amplitude_encode(torch.tensor([1.0]), num_qubits=-1)

    def test_too_many_qubits_raises(self):
        with pytest.raises(ValueError, match="num_qubits"):
            amplitude_encode(torch.tensor([1.0]), num_qubits=31)


# ============================================================================
# FWHT Correctness Tests
# ============================================================================


class TestFWHT:
    def test_hadamard_2x2(self):
        """FWHT of [1, 0] should give [1, 1] (unnormalized Hadamard)."""
        from qumat_qdp.encodings import _fwht_inplace

        state = torch.tensor([1.0, 0.0])
        result = _fwht_inplace(state)
        assert torch.allclose(result, torch.tensor([1.0, 1.0]))

    def test_hadamard_identity(self):
        """Applying FWHT twice and dividing by N should give identity."""
        from qumat_qdp.encodings import _fwht_inplace

        original = torch.randn(8)
        state = original.clone()
        _fwht_inplace(state)
        _fwht_inplace(state)
        state = state / 8  # N = 2^3
        assert torch.allclose(state, original, atol=ATOL)
