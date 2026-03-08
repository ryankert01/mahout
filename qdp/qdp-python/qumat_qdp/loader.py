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
Quantum Data Loader: Python builder for batch encoding iterator.

Uses the PyTorch-native engine by default (no Rust/CUDA dependency required).

Usage:
    from qumat_qdp import QuantumDataLoader

    loader = (QuantumDataLoader(device_id=0).qubits(16).encoding("amplitude")
              .batches(100, size=64).source_synthetic())
    for batch_tensor in loader:
        # batch_tensor is a PyTorch complex tensor of shape (batch_size, 2^num_qubits)
        ...
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import torch

# Seed must fit Rust u64: 0 <= seed <= 2^64 - 1.
_U64_MAX = 2**64 - 1


def _validate_loader_args(
    *,
    device_id: int,
    num_qubits: int,
    batch_size: int,
    total_batches: int,
    encoding_method: str,
    seed: int | None,
) -> None:
    """Validate arguments before building the loader."""
    if device_id < 0:
        raise ValueError(f"device_id must be non-negative, got {device_id!r}")
    if not isinstance(num_qubits, int) or num_qubits < 1:
        raise ValueError(f"num_qubits must be a positive integer, got {num_qubits!r}")
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError(f"batch_size must be a positive integer, got {batch_size!r}")
    if not isinstance(total_batches, int) or total_batches < 1:
        raise ValueError(
            f"total_batches must be a positive integer, got {total_batches!r}"
        )
    if not encoding_method or not isinstance(encoding_method, str):
        raise ValueError(
            f"encoding_method must be a non-empty string, got {encoding_method!r}"
        )
    if seed is not None:
        if not isinstance(seed, int):
            raise ValueError(
                f"seed must be None or an integer, got {type(seed).__name__!r}"
            )
        if seed < 0 or seed > _U64_MAX:
            raise ValueError(
                f"seed must be in range [0, {_U64_MAX}] (Rust u64), got {seed!r}"
            )


def _sample_size_for_encoding(encoding_method: str, num_qubits: int) -> int:
    """Determine the number of input features per sample for the given encoding."""
    if encoding_method == "amplitude":
        return 1 << num_qubits
    elif encoding_method == "basis":
        return 1
    elif encoding_method in ("angle", "iqp-z"):
        return num_qubits
    elif encoding_method == "iqp":
        n = num_qubits
        return n + n * (n - 1) // 2
    else:
        raise ValueError(f"Unknown encoding method: {encoding_method!r}")


class QuantumDataLoader:
    """
    Builder for a quantum encoding iterator.

    Yields one PyTorch tensor (batch) per iteration. Uses the PyTorch-native
    engine for encoding.
    """

    def __init__(
        self,
        device_id: int = 0,
        num_qubits: int = 16,
        batch_size: int = 64,
        total_batches: int = 100,
        encoding_method: str = "amplitude",
        seed: int | None = None,
    ) -> None:
        _validate_loader_args(
            device_id=device_id,
            num_qubits=num_qubits,
            batch_size=batch_size,
            total_batches=total_batches,
            encoding_method=encoding_method,
            seed=seed,
        )
        self._device_id = device_id
        self._num_qubits = num_qubits
        self._batch_size = batch_size
        self._total_batches = total_batches
        self._encoding_method = encoding_method
        self._seed = seed
        self._file_path: str | None = None
        self._streaming_requested = False
        self._synthetic_requested = False
        self._file_requested = False
        self._null_handling: str | None = None

    def qubits(self, n: int) -> QuantumDataLoader:
        """Set number of qubits. Returns self for chaining."""
        if not isinstance(n, int) or n < 1:
            raise ValueError(f"num_qubits must be a positive integer, got {n!r}")
        self._num_qubits = n
        return self

    def encoding(self, method: str) -> QuantumDataLoader:
        """Set encoding method (e.g. 'amplitude', 'angle', 'basis'). Returns self."""
        if not method or not isinstance(method, str):
            raise ValueError(
                f"encoding_method must be a non-empty string, got {method!r}"
            )
        self._encoding_method = method
        return self

    def batches(self, total: int, size: int = 64) -> QuantumDataLoader:
        """Set total number of batches and batch size. Returns self."""
        if not isinstance(total, int) or total < 1:
            raise ValueError(f"total_batches must be a positive integer, got {total!r}")
        if not isinstance(size, int) or size < 1:
            raise ValueError(f"batch_size must be a positive integer, got {size!r}")
        self._total_batches = total
        self._batch_size = size
        return self

    def source_synthetic(
        self,
        total_batches: int | None = None,
    ) -> QuantumDataLoader:
        """Use synthetic data source (default). Optionally override total_batches. Returns self."""
        self._synthetic_requested = True
        if total_batches is not None:
            if not isinstance(total_batches, int) or total_batches < 1:
                raise ValueError(
                    f"total_batches must be a positive integer, got {total_batches!r}"
                )
            self._total_batches = total_batches
        return self

    def source_file(self, path: str, streaming: bool = False) -> QuantumDataLoader:
        """Use file data source. Path must point to a supported format. Returns self.

        For streaming=True, only .parquet is supported; data is read in chunks to reduce memory.
        For streaming=False, supports .parquet, .arrow, .feather, .ipc, .npy, .pt, .pth.
        """
        if not path or not isinstance(path, str):
            raise ValueError(f"path must be a non-empty string, got {path!r}")
        if streaming and not (path.lower().endswith(".parquet")):
            raise ValueError(
                "streaming=True supports only .parquet files; use streaming=False for other formats."
            )
        self._file_path = path
        self._file_requested = True
        self._streaming_requested = streaming
        return self

    def seed(self, s: int | None = None) -> QuantumDataLoader:
        """Set RNG seed for reproducible synthetic data. Returns self."""
        if s is not None:
            if not isinstance(s, int):
                raise ValueError(
                    f"seed must be None or an integer, got {type(s).__name__!r}"
                )
            if s < 0 or s > _U64_MAX:
                raise ValueError(
                    f"seed must be in range [0, {_U64_MAX}] (Rust u64), got {s!r}"
                )
        self._seed = s
        return self

    def null_handling(self, policy: str) -> QuantumDataLoader:
        """Set null handling policy ('fill_zero' or 'reject'). Returns self for chaining."""
        if policy not in ("fill_zero", "reject"):
            raise ValueError(
                f"null_handling must be 'fill_zero' or 'reject', got {policy!r}"
            )
        self._null_handling = policy
        return self

    def _create_iterator(self) -> Iterator[torch.Tensor]:
        """Build engine and return the loader iterator (synthetic or file)."""
        if self._synthetic_requested and self._file_requested:
            raise ValueError(
                "Cannot set both synthetic and file sources; use either .source_synthetic() or .source_file(path), not both."
            )
        if self._file_requested and self._file_path is None:
            raise ValueError(
                "source_file() was not called with a path; set file source with .source_file(path)."
            )

        use_synthetic = not self._file_requested

        if use_synthetic:
            _validate_loader_args(
                device_id=self._device_id,
                num_qubits=self._num_qubits,
                batch_size=self._batch_size,
                total_batches=self._total_batches,
                encoding_method=self._encoding_method,
                seed=self._seed,
            )
            return self._synthetic_iterator()
        else:
            return self._file_iterator()

    def _synthetic_iterator(self) -> Iterator[torch.Tensor]:
        """Generate synthetic random data and encode it batch by batch."""
        from qumat_qdp.encodings import encode_batch
        from qumat_qdp.engine import QdpEngine

        engine = QdpEngine(device_id=self._device_id, precision="float32")
        sample_size = _sample_size_for_encoding(self._encoding_method, self._num_qubits)

        if self._seed is not None:
            gen = torch.Generator(device="cpu").manual_seed(self._seed)
        else:
            gen = None

        for _ in range(self._total_batches):
            if self._encoding_method == "basis":
                state_len = 1 << self._num_qubits
                batch_data = torch.randint(
                    0,
                    state_len,
                    (self._batch_size, 1),
                    generator=gen,
                    dtype=torch.float64,
                )
            else:
                batch_data = torch.randn(self._batch_size, sample_size, generator=gen)

            result = encode_batch(
                batch_data,
                self._num_qubits,
                self._encoding_method,
                precision="float32",
                device=engine.device,
            )
            yield result

    def _file_iterator(self) -> Iterator[torch.Tensor]:
        """Load data from file and encode batch by batch."""
        from qumat_qdp.encodings import encode_batch
        from qumat_qdp.engine import QdpEngine

        engine = QdpEngine(device_id=self._device_id, precision="float32")
        path = self._file_path
        assert path is not None

        if self._streaming_requested:
            yield from self._streaming_parquet_iterator(engine)
            return

        # Load the full file
        lower = path.lower()
        if lower.endswith(".npy"):
            data = np.load(path)
        elif lower.endswith((".pt", ".pth")):
            loaded = torch.load(path, weights_only=True)
            if isinstance(loaded, torch.Tensor):
                data = loaded.numpy()
            else:
                raise ValueError(f"Expected a tensor in {path}")
        elif lower.endswith(".parquet"):
            try:
                import pyarrow.parquet as pq
            except ImportError as exc:
                raise ImportError(
                    "pyarrow required for Parquet: pip install pyarrow"
                ) from exc
            table = pq.read_table(path)
            data = table.to_pandas().values
        elif lower.endswith((".arrow", ".feather", ".ipc")):
            try:
                import pyarrow.ipc as ipc
            except ImportError as exc:
                raise ImportError(
                    "pyarrow required for Arrow IPC: pip install pyarrow"
                ) from exc
            with open(path, "rb") as f:
                reader = ipc.open_file(f)
                table = reader.read_all()
            data = table.to_pandas().values
        else:
            raise RuntimeError(
                f"Unsupported file extension for {path!r}. "
                f"Supported: .parquet, .arrow, .feather, .ipc, .npy, .pt, .pth"
            )

        data = np.asarray(data, dtype=np.float64)
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        # Null handling
        if self._null_handling == "reject":
            if np.any(~np.isfinite(data)):
                raise ValueError("Data contains non-finite values (NaN or Inf)")
        else:
            # fill_zero or None (default)
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

        total_samples = data.shape[0]
        batches_yielded = 0
        idx = 0
        while batches_yielded < self._total_batches and idx < total_samples:
            end = min(idx + self._batch_size, total_samples)
            batch_np = data[idx:end]
            batch_tensor = torch.from_numpy(batch_np)
            result = encode_batch(
                batch_tensor,
                self._num_qubits,
                self._encoding_method,
                precision="float32",
                device=engine.device,
            )
            yield result
            idx = end
            batches_yielded += 1

    def _streaming_parquet_iterator(self, engine) -> Iterator[torch.Tensor]:
        """Stream Parquet file in row groups for memory efficiency."""
        from qumat_qdp.encodings import encode_batch

        path = self._file_path
        assert path is not None

        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise ImportError(
                "pyarrow required for Parquet streaming: pip install pyarrow"
            ) from exc

        parquet_file = pq.ParquetFile(path)
        batches_yielded = 0
        buffer: np.ndarray | None = None

        for rg_idx in range(parquet_file.metadata.num_row_groups):
            if batches_yielded >= self._total_batches:
                break

            table = parquet_file.read_row_group(rg_idx)
            chunk = table.to_pandas().values.astype(np.float64)

            if self._null_handling == "reject":
                if np.any(~np.isfinite(chunk)):
                    raise ValueError("Data contains non-finite values")
            else:
                chunk = np.nan_to_num(chunk, nan=0.0, posinf=0.0, neginf=0.0)

            buffer = chunk if buffer is None else np.vstack([buffer, chunk])

            while (
                buffer.shape[0] >= self._batch_size
                and batches_yielded < self._total_batches
            ):
                batch_np = buffer[: self._batch_size]
                buffer = buffer[self._batch_size :]
                batch_tensor = torch.from_numpy(batch_np)
                result = encode_batch(
                    batch_tensor,
                    self._num_qubits,
                    self._encoding_method,
                    precision="float32",
                    device=engine.device,
                )
                yield result
                batches_yielded += 1

    def __iter__(self) -> Iterator[torch.Tensor]:
        """Return iterator that yields one PyTorch tensor per batch."""
        return self._create_iterator()
