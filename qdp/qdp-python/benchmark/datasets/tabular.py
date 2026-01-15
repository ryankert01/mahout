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

"""Tabular datasets for benchmarking."""

from typing import Optional, Tuple

import numpy as np

from .base import BenchmarkDataset, register_dataset


class IrisBinaryDataset(BenchmarkDataset):
    """Binary Iris classification dataset.

    Uses sklearn's Iris dataset with configurable binary classification
    (default: setosa vs versicolor, classes 0 and 1).

    Args:
        class_0: First class index (0, 1, or 2).
        class_1: Second class index (0, 1, or 2).
        seed: Random seed for shuffling.

    Example:
        >>> dataset = IrisBinaryDataset(class_0=0, class_1=1)
        >>> X, y = dataset.prepare_for_qubits(n_qubits=2)  # 4 features
        >>> X.shape
        (100, 4)
    """

    def __init__(self, class_0: int = 0, class_1: int = 1, seed: int = 42):
        if not (0 <= class_0 <= 2 and 0 <= class_1 <= 2):
            raise ValueError("class_0 and class_1 must be in [0, 1, 2]")
        if class_0 == class_1:
            raise ValueError("class_0 and class_1 must be different")

        self._class_0 = class_0
        self._class_1 = class_1
        self._seed = seed
        self._X: Optional[np.ndarray] = None
        self._y: Optional[np.ndarray] = None

    @property
    def name(self) -> str:
        return f"iris_binary_{self._class_0}_vs_{self._class_1}"

    @property
    def n_samples(self) -> int:
        if self._X is None:
            self._load_data()
        assert self._X is not None  # Populated by _load_data()
        return len(self._X)

    def _load_data(self):
        """Load and filter Iris data."""
        from sklearn.datasets import load_iris

        iris = load_iris()
        X_all = iris.data  # (150, 4)
        y_all = iris.target  # (150,)

        # Filter to binary classes
        mask = (y_all == self._class_0) | (y_all == self._class_1)
        X = X_all[mask]
        y = np.where(y_all[mask] == self._class_0, 0, 1)

        # Shuffle
        rng = np.random.default_rng(self._seed)
        indices = rng.permutation(len(X))
        self._X = X[indices]
        self._y = y[indices]

    def load(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Load the raw Iris binary dataset.

        Returns:
            Tuple of (X, y) where X has shape (n_samples, 4) and
            y has shape (n_samples,) with binary labels.
        """
        if self._X is None:
            self._load_data()
        assert self._X is not None and self._y is not None  # Populated by _load_data()
        return self._X.copy(), self._y.copy()

    def prepare_for_qubits(
        self, n_qubits: int, normalize: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Prepare Iris data for quantum encoding.

        Iris has 4 features, so:
        - n_qubits=2 (4 features): Perfect fit
        - n_qubits<2: Truncate features
        - n_qubits>2: Zero-pad features

        Args:
            n_qubits: Number of qubits (features will be 2^n_qubits).
            normalize: Whether to L2-normalize each sample.

        Returns:
            Tuple of (X, y) where X has shape (n_samples, 2^n_qubits).
        """
        X, y = self.load()
        target_dim = 1 << n_qubits

        X = self._resize_features(X, target_dim, method="pad")

        if normalize:
            X = self._normalize(X)

        return X, y


class SyntheticBlobsDataset(BenchmarkDataset):
    """Synthetic blob clusters dataset.

    Uses sklearn's make_blobs to generate configurable clusters
    for binary classification.

    Args:
        n_samples: Number of samples to generate.
        n_features: Number of features (will be adjusted for qubits).
        n_clusters: Number of clusters per class (default 1).
        cluster_std: Standard deviation of clusters.
        seed: Random seed for reproducibility.

    Example:
        >>> dataset = SyntheticBlobsDataset(n_samples=500, seed=42)
        >>> X, y = dataset.prepare_for_qubits(n_qubits=4)
        >>> X.shape
        (500, 16)
    """

    def __init__(
        self,
        n_samples: int = 500,
        n_features: int = 16,
        n_clusters: int = 1,
        cluster_std: float = 1.0,
        seed: int = 42,
    ):
        self._n_samples = n_samples
        self._n_features = n_features
        self._n_clusters = n_clusters
        self._cluster_std = cluster_std
        self._seed = seed
        self._X: Optional[np.ndarray] = None
        self._y: Optional[np.ndarray] = None

    @property
    def name(self) -> str:
        return f"blobs_{self._n_clusters}c_{self._n_features}f"

    @property
    def n_samples(self) -> int:
        return self._n_samples

    def _generate_data(self, n_features: int):
        """Generate blob data with specific feature count."""
        from sklearn.datasets import make_blobs

        # Generate 2 classes with n_clusters centers each
        n_centers = 2 * self._n_clusters
        X, y_raw = make_blobs(
            n_samples=self._n_samples,
            n_features=n_features,
            centers=n_centers,
            cluster_std=self._cluster_std,
            random_state=self._seed,
        )

        # Convert to binary: first half of centers = class 0
        y = (y_raw >= self._n_clusters).astype(int)

        return X.astype(np.float64), y

    def load(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Load the synthetic blobs dataset.

        Returns:
            Tuple of (X, y) where X has shape (n_samples, n_features) and
            y has shape (n_samples,) with binary labels.
        """
        if self._X is None:
            self._X, self._y = self._generate_data(self._n_features)
        assert self._X is not None and self._y is not None  # Populated above
        return self._X.copy(), self._y.copy()

    def prepare_for_qubits(
        self, n_qubits: int, normalize: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Prepare blob data for quantum encoding.

        Generates data directly with 2^n_qubits features.

        Args:
            n_qubits: Number of qubits (features will be 2^n_qubits).
            normalize: Whether to L2-normalize each sample.

        Returns:
            Tuple of (X, y) where X has shape (n_samples, 2^n_qubits).
        """
        target_dim = 1 << n_qubits

        # Generate fresh data with target dimension
        X, y = self._generate_data(target_dim)

        if normalize:
            X = self._normalize(X)

        return X, y


# Register datasets
register_dataset("iris", IrisBinaryDataset)
register_dataset("iris_binary", IrisBinaryDataset)
register_dataset("blobs", SyntheticBlobsDataset)
register_dataset("synthetic_blobs", SyntheticBlobsDataset)
