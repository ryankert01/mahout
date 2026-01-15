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

"""Base class for benchmark datasets."""

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np


class BenchmarkDataset(ABC):
    """Abstract base class for benchmark datasets.

    All benchmark datasets should inherit from this class and implement
    the required methods.

    Attributes:
        name: Human-readable name of the dataset.
        n_samples: Number of samples in the dataset.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the dataset."""
        pass

    @property
    @abstractmethod
    def n_samples(self) -> int:
        """Return the number of samples."""
        pass

    @abstractmethod
    def load(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Load the raw dataset.

        Returns:
            Tuple of (X, y) where X is the feature matrix and y is the
            optional label array. X has shape (n_samples, n_features),
            y has shape (n_samples,) if present.
        """
        pass

    @abstractmethod
    def prepare_for_qubits(
        self, n_qubits: int, normalize: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Prepare the dataset for a specific number of qubits.

        This method should:
        1. Load the raw data
        2. Resize/pad features to match 2^n_qubits dimensions
        3. Optionally L2-normalize each sample

        Args:
            n_qubits: Number of qubits (features will be 2^n_qubits).
            normalize: Whether to L2-normalize each sample (default True).

        Returns:
            Tuple of (X, y) where X has shape (n_samples, 2^n_qubits).
        """
        pass

    def _resize_features(
        self, X: np.ndarray, target_dim: int, method: str = "pad"
    ) -> np.ndarray:
        """Resize feature vectors to target dimension.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            target_dim: Target dimension (must be power of 2 for qubits).
            method: Resize method - 'pad' (zero-pad), 'truncate', or 'interpolate'.

        Returns:
            Resized feature matrix of shape (n_samples, target_dim).
        """
        n_samples, n_features = X.shape

        if n_features == target_dim:
            return X

        if method == "pad":
            if n_features > target_dim:
                # Truncate to target dimension
                return X[:, :target_dim]
            else:
                # Zero-pad to target dimension
                result = np.zeros((n_samples, target_dim), dtype=X.dtype)
                result[:, :n_features] = X
                return result

        elif method == "truncate":
            return X[:, :target_dim]

        elif method == "interpolate":
            # Use linear interpolation to resize
            from scipy.ndimage import zoom

            zoom_factor = target_dim / n_features
            return zoom(X, (1, zoom_factor), order=1)

        else:
            raise ValueError(f"Unknown resize method: {method}")

    def _normalize(self, X: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
        """L2-normalize each row of the feature matrix.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            epsilon: Small value to avoid division by zero.

        Returns:
            Normalized feature matrix where each row has unit L2 norm.
        """
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.maximum(norms, epsilon)
        return X / norms

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', n_samples={self.n_samples})"


# Dataset registry - simple dict-based lookup
_DATASETS = {}


def register_dataset(name: str, dataset_class: type):
    """Register a dataset class by name.

    Args:
        name: Name to register the dataset under.
        dataset_class: The dataset class to register.
    """
    _DATASETS[name.lower()] = dataset_class


def get_dataset(name: str, **kwargs) -> BenchmarkDataset:
    """Get a dataset by name.

    Args:
        name: Name of the dataset (case-insensitive).
        **kwargs: Arguments to pass to the dataset constructor.

    Returns:
        Instantiated dataset.

    Raises:
        ValueError: If the dataset name is not recognized.
    """
    name_lower = name.lower()
    if name_lower not in _DATASETS:
        available = ", ".join(sorted(_DATASETS.keys()))
        raise ValueError(
            f"Unknown dataset: '{name}'. Available datasets: {available or 'none'}"
        )
    return _DATASETS[name_lower](**kwargs)


def list_datasets() -> list:
    """Return a list of registered dataset names."""
    return sorted(_DATASETS.keys())
