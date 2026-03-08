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

"""Tests for PyTorch-native QdpEngine and QuantumDataLoader."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
import torch
from qumat_qdp.engine import QdpEngine
from qumat_qdp.loader import QuantumDataLoader

ATOL = 1e-5


# ============================================================================
# QdpEngine Tests
# ============================================================================


class TestQdpEngine:
    def test_init_default(self):
        engine = QdpEngine()
        assert engine.precision == "float32"

    def test_init_with_precision(self):
        engine = QdpEngine(precision="float64")
        assert engine.precision == "float64"

    def test_invalid_precision_raises(self):
        with pytest.raises(ValueError, match="precision"):
            QdpEngine(precision="float16")

    def test_negative_device_raises(self):
        with pytest.raises(ValueError, match="device_id"):
            QdpEngine(device_id=-1)

    def test_encode_list(self):
        engine = QdpEngine()
        state = engine.encode([1.0, 2.0, 3.0, 4.0], num_qubits=2)
        assert isinstance(state, torch.Tensor)
        assert state.shape == (4,)
        assert state.is_complex()

    def test_encode_numpy_1d(self):
        engine = QdpEngine()
        data = np.array([1.0, 2.0, 3.0, 4.0])
        state = engine.encode(data, num_qubits=2)
        assert state.shape == (4,)

    def test_encode_numpy_2d_batch(self):
        engine = QdpEngine()
        data = np.random.randn(3, 4)
        state = engine.encode(data, num_qubits=2)
        assert state.shape == (3, 4)

    def test_encode_torch_1d(self):
        engine = QdpEngine()
        data = torch.randn(4)
        state = engine.encode(data, num_qubits=2)
        assert state.shape == (4,)

    def test_encode_torch_2d_batch(self):
        engine = QdpEngine()
        data = torch.randn(5, 4)
        state = engine.encode(data, num_qubits=2)
        assert state.shape == (5, 4)

    def test_encode_angle(self):
        engine = QdpEngine()
        data = torch.randn(3)
        state = engine.encode(data, num_qubits=3, encoding_method="angle")
        assert state.shape == (8,)

    def test_encode_basis(self):
        engine = QdpEngine()
        state = engine.encode([2.0], num_qubits=2, encoding_method="basis")
        assert state.shape == (4,)
        assert state[2].real.item() == pytest.approx(1.0, abs=ATOL)

    def test_encode_iqp(self):
        engine = QdpEngine()
        data = torch.randn(3)  # 2 + 1 for 2 qubits
        state = engine.encode(data, num_qubits=2, encoding_method="iqp")
        assert state.shape == (4,)

    def test_encode_from_npy_file(self):
        engine = QdpEngine()
        data = np.array([1.0, 2.0, 3.0, 4.0])
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
            np.save(f, data)
            path = f.name
        try:
            state = engine.encode(path, num_qubits=2)
            assert state.shape == (4,)
        finally:
            os.unlink(path)

    def test_encode_from_pt_file(self):
        engine = QdpEngine()
        data = torch.randn(4)
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save(data, f)
            path = f.name
        try:
            state = engine.encode(path, num_qubits=2)
            assert state.shape == (4,)
        finally:
            os.unlink(path)

    def test_unsupported_type_raises(self):
        engine = QdpEngine()
        with pytest.raises(TypeError, match="Unsupported"):
            engine.encode(42, num_qubits=2)


# ============================================================================
# QuantumDataLoader Tests (PyTorch-native)
# ============================================================================


class TestQuantumDataLoaderPyTorch:
    def test_synthetic_loader_batch_count(self):
        total = 5
        batch_size = 4
        loader = (
            QuantumDataLoader(device_id=0)
            .qubits(4)
            .batches(total, size=batch_size)
            .source_synthetic()
        )
        batches = list(loader)
        assert len(batches) == total

    def test_synthetic_loader_shape(self):
        loader = (
            QuantumDataLoader(device_id=0)
            .qubits(3)
            .batches(2, size=8)
            .source_synthetic()
        )
        batches = list(loader)
        assert batches[0].shape == (8, 8)  # 2^3 = 8

    def test_synthetic_loader_complex_dtype(self):
        loader = (
            QuantumDataLoader(device_id=0)
            .qubits(2)
            .batches(1, size=4)
            .source_synthetic()
        )
        batch = next(iter(loader))
        assert batch.is_complex()

    def test_seed_reproducibility(self):
        def run(seed):
            loader = (
                QuantumDataLoader(device_id=0)
                .qubits(2)
                .batches(2, size=4)
                .seed(seed)
                .source_synthetic()
            )
            return [b.clone() for b in loader]

        a = run(42)
        b = run(42)
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert torch.allclose(x, y, atol=1e-6)

    def test_mutual_exclusion_raises(self):
        loader = (
            QuantumDataLoader(device_id=0)
            .qubits(4)
            .batches(10, size=4)
            .source_synthetic()
            .source_file("/tmp/any.parquet")
        )
        with pytest.raises(ValueError, match="Cannot set both"):
            list(loader)

    def test_file_loader_from_npy(self):
        data = np.random.randn(20, 4).astype(np.float64)
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
            np.save(f, data)
            path = f.name
        try:
            loader = (
                QuantumDataLoader(device_id=0)
                .qubits(2)
                .batches(3, size=5)
                .source_file(path)
            )
            batches = list(loader)
            assert len(batches) == 3
            assert batches[0].shape == (5, 4)
        finally:
            os.unlink(path)

    def test_null_handling_fill_zero(self):
        loader = (
            QuantumDataLoader(device_id=0)
            .qubits(4)
            .batches(10, size=4)
            .null_handling("fill_zero")
        )
        assert loader._null_handling == "fill_zero"

    def test_null_handling_reject(self):
        loader = (
            QuantumDataLoader(device_id=0)
            .qubits(4)
            .batches(10, size=4)
            .null_handling("reject")
        )
        assert loader._null_handling == "reject"

    def test_null_handling_invalid_raises(self):
        with pytest.raises(ValueError):
            QuantumDataLoader(device_id=0).null_handling("invalid_policy")

    def test_source_file_empty_path_raises(self):
        with pytest.raises(ValueError, match="path"):
            QuantumDataLoader(device_id=0).qubits(4).batches(10, size=4).source_file("")

    def test_streaming_requires_parquet(self):
        with pytest.raises(ValueError, match="parquet"):
            QuantumDataLoader(device_id=0).qubits(4).batches(10, size=4).source_file(
                "/tmp/data.npy", streaming=True
            )

    def test_angle_encoding_loader(self):
        loader = (
            QuantumDataLoader(device_id=0)
            .qubits(3)
            .encoding("angle")
            .batches(2, size=4)
            .source_synthetic()
        )
        batches = list(loader)
        assert len(batches) == 2
        assert batches[0].shape == (4, 8)  # 2^3 = 8

    def test_basis_encoding_loader(self):
        loader = (
            QuantumDataLoader(device_id=0)
            .qubits(3)
            .encoding("basis")
            .batches(2, size=4)
            .source_synthetic()
        )
        batches = list(loader)
        assert len(batches) == 2
        assert batches[0].shape == (4, 8)
