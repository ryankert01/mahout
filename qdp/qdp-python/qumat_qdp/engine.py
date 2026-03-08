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
PyTorch-native QdpEngine: drop-in replacement for the Rust+CUDA QdpEngine.

Implements the same public API as the PyO3-based engine but uses pure PyTorch
operations. No Rust, CUDA toolkit, or maturin build required.

Usage:
    from qumat_qdp.engine import QdpEngine

    engine = QdpEngine(device_id=0, precision="float32")
    tensor = engine.encode([1.0, 2.0, 3.0, 4.0], num_qubits=2)
    # tensor is a PyTorch complex tensor on the specified device
"""

from __future__ import annotations

import pathlib

import numpy as np
import torch

from qumat_qdp.encodings import encode, encode_batch


class QdpEngine:
    """PyTorch-native quantum data processing engine.

    Compatible API with the Rust-backed QdpEngine. All encoding runs via
    PyTorch operations (no custom CUDA kernels).
    """

    def __init__(
        self,
        device_id: int = 0,
        precision: str = "float32",
    ) -> None:
        if device_id < 0:
            raise ValueError(f"device_id must be non-negative, got {device_id!r}")
        if precision not in ("float32", "float64"):
            raise ValueError(
                f"precision must be 'float32' or 'float64', got {precision!r}"
            )

        self._device_id = device_id
        self._precision = precision

        if torch.cuda.is_available() and device_id < torch.cuda.device_count():
            self._device = torch.device(f"cuda:{device_id}")
        else:
            self._device = torch.device("cpu")

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def precision(self) -> str:
        return self._precision

    def _to_tensor(self, data) -> torch.Tensor:
        """Convert various input types to a torch.Tensor on the engine device."""
        if isinstance(data, torch.Tensor):
            return data.to(self._device)
        if isinstance(data, np.ndarray):
            return torch.from_numpy(data).to(self._device)
        if isinstance(data, (list, tuple)):
            return torch.tensor(data, dtype=torch.float64, device=self._device)
        raise TypeError(
            f"Unsupported data type {type(data).__name__}. "
            f"Expected torch.Tensor, numpy.ndarray, list, or tuple."
        )

    def _resolve_file_path(self, data) -> str | None:
        """Check if data is a file path (str or pathlib.Path)."""
        if isinstance(data, pathlib.PurePath):
            return str(data)
        if isinstance(data, str) and data and not data.startswith("["):
            # Heuristic: strings that look like paths (have extensions)
            suffixes = (
                ".parquet",
                ".arrow",
                ".feather",
                ".ipc",
                ".npy",
                ".pt",
                ".pth",
                ".pb",
            )
            if any(data.lower().endswith(s) for s in suffixes):
                return data
        return None

    def encode(
        self,
        data,
        num_qubits: int,
        encoding_method: str = "amplitude",
    ) -> torch.Tensor:
        """Encode data into a quantum state vector.

        Auto-detects input type:
        - str/pathlib.Path: load from file
        - numpy.ndarray (1-D): single sample encoding
        - numpy.ndarray (2-D): batch encoding
        - torch.Tensor (1-D): single sample encoding
        - torch.Tensor (2-D): batch encoding
        - list/tuple: convert to tensor, then single or batch

        Args:
            data: Input data (various types supported).
            num_qubits: Number of qubits.
            encoding_method: Encoding method name.

        Returns:
            Complex PyTorch tensor (state vector or batch of state vectors).
        """
        # Check for file path
        path = self._resolve_file_path(data)
        if path is not None:
            return self._encode_from_file(path, num_qubits, encoding_method)

        tensor = self._to_tensor(data)

        if tensor.dim() == 1:
            return encode(
                tensor,
                num_qubits,
                encoding_method,
                precision=self._precision,
                device=self._device,
            )
        elif tensor.dim() == 2:
            return encode_batch(
                tensor,
                num_qubits,
                encoding_method,
                precision=self._precision,
                device=self._device,
            )
        else:
            raise ValueError(f"Expected 1-D or 2-D input, got {tensor.dim()}-D tensor")

    def _encode_from_file(
        self,
        path: str,
        num_qubits: int,
        encoding_method: str,
    ) -> torch.Tensor:
        """Load data from file and encode."""
        lower = path.lower()

        if lower.endswith(".npy"):
            data = np.load(path)
        elif lower.endswith((".pt", ".pth")):
            data = torch.load(path, weights_only=True)
            if isinstance(data, torch.Tensor):
                return self.encode(data, num_qubits, encoding_method)
            raise ValueError(f"Expected a tensor in {path}, got {type(data).__name__}")
        elif lower.endswith(".parquet"):
            try:
                import pyarrow.parquet as pq
            except ImportError:
                raise ImportError(
                    "pyarrow is required for Parquet file support. "
                    "Install with: pip install pyarrow"
                )
            table = pq.read_table(path)
            data = table.to_pandas().values
        elif lower.endswith((".arrow", ".feather", ".ipc")):
            try:
                import pyarrow.ipc as ipc
            except ImportError:
                raise ImportError(
                    "pyarrow is required for Arrow IPC file support. "
                    "Install with: pip install pyarrow"
                )
            with open(path, "rb") as f:
                reader = ipc.open_file(f)
                table = reader.read_all()
            data = table.to_pandas().values
        elif lower.endswith(".pb"):
            raise NotImplementedError(
                "TensorFlow protobuf loading is not yet supported in the PyTorch-native engine. "
                "Convert to .npy or .pt first."
            )
        else:
            raise ValueError(
                f"Unsupported file extension for {path!r}. "
                f"Supported: .parquet, .arrow, .feather, .ipc, .npy, .pt, .pth"
            )

        tensor = self._to_tensor(data)
        if tensor.dim() == 1:
            return encode(
                tensor,
                num_qubits,
                encoding_method,
                precision=self._precision,
                device=self._device,
            )
        elif tensor.dim() == 2:
            return encode_batch(
                tensor,
                num_qubits,
                encoding_method,
                precision=self._precision,
                device=self._device,
            )
        else:
            raise ValueError(f"File data must be 1-D or 2-D, got {tensor.dim()}-D")
