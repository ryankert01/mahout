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

# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Synthetic random dataset for benchmarking."""

from typing import Optional, Tuple

import numpy as np

from .base import BenchmarkDataset, register_dataset


class SyntheticDataset(BenchmarkDataset):
    """Synthetic dataset with reproducible random vectors.

    This is the simplest benchmark dataset, generating random vectors
    with a fixed seed for reproducibility. Useful for measuring raw
    encoding performance without I/O overhead.

    Args:
        n_samples: Number of samples to generate.
        seed: Random seed for reproducibility.

    Example:
        >>> dataset = SyntheticDataset(n_samples=100, seed=42)
        >>> X, y = dataset.prepare_for_qubits(n_qubits=4)
        >>> X.shape
        (100, 16)
    """

    def __init__(self, n_samples: int = 1000, seed: int = 42):
        self._n_samples = n_samples
        self._seed = seed
        self._cached_data: Optional[np.ndarray] = None
        self._cached_n_qubits: Optional[int] = None

    @property
    def name(self) -> str:
        return "synthetic"

    @property
    def n_samples(self) -> int:
        return self._n_samples

    def load(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Load generates random data - requires n_qubits context.

        For synthetic data, use prepare_for_qubits() directly.

        Returns:
            Empty array and None labels (use prepare_for_qubits instead).
        """
        # Return placeholder - real generation happens in prepare_for_qubits
        return np.array([]), None

    def prepare_for_qubits(
        self, n_qubits: int, normalize: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Generate random vectors for the specified number of qubits.

        Args:
            n_qubits: Number of qubits (features will be 2^n_qubits).
            normalize: Whether to L2-normalize each sample.

        Returns:
            Tuple of (X, None) where X has shape (n_samples, 2^n_qubits).
        """
        # Check cache
        if (
            self._cached_data is not None
            and self._cached_n_qubits == n_qubits
            and len(self._cached_data) == self._n_samples
        ):
            X = self._cached_data.copy()
        else:
            # Generate fresh data
            dim = 1 << n_qubits
            rng = np.random.default_rng(self._seed)
            X = rng.random((self._n_samples, dim), dtype=np.float64)
            self._cached_data = X.copy()
            self._cached_n_qubits = n_qubits

        if normalize:
            X = self._normalize(X)

        return X, None

    def generate_batch(
        self, batch_size: int, n_qubits: int, batch_idx: int = 0, normalize: bool = True
    ) -> np.ndarray:
        """Generate a single batch of random vectors.

        Uses deterministic seeding based on batch index for reproducibility.

        Args:
            batch_size: Number of samples in the batch.
            n_qubits: Number of qubits (features will be 2^n_qubits).
            batch_idx: Batch index for deterministic seeding.
            normalize: Whether to L2-normalize each sample.

        Returns:
            Batch array of shape (batch_size, 2^n_qubits).
        """
        dim = 1 << n_qubits
        # Combine seed with batch index for reproducibility
        batch_seed = self._seed + batch_idx * 1000
        rng = np.random.default_rng(batch_seed)
        X = rng.random((batch_size, dim), dtype=np.float64)

        if normalize:
            X = self._normalize(X)

        return X


# Register the dataset
register_dataset("synthetic", SyntheticDataset)
