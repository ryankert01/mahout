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

"""Image datasets for benchmarking."""

from typing import List, Optional, Tuple, Union

import numpy as np

from .base import BenchmarkDataset, register_dataset


class MNISTBinaryDataset(BenchmarkDataset):
    """Binary MNIST classification dataset.

    Uses sklearn's MNIST digits dataset with configurable binary
    classification task (e.g., 0 vs 1, odd vs even).

    Args:
        class_0: Digit(s) for class 0 (can be int or list).
        class_1: Digit(s) for class 1 (can be int or list).
        n_samples: Maximum number of samples to use (None for all).
        seed: Random seed for shuffling.

    Example:
        >>> dataset = MNISTBinaryDataset(class_0=0, class_1=1, n_samples=200)
        >>> X, y = dataset.prepare_for_qubits(n_qubits=6)  # 64 features
        >>> X.shape
        (200, 64)
    """

    def __init__(
        self,
        class_0: Union[int, List[int]] = 0,
        class_1: Union[int, List[int]] = 1,
        n_samples: Optional[int] = None,
        seed: int = 42,
    ):
        self._class_0 = [class_0] if isinstance(class_0, int) else list(class_0)
        self._class_1 = [class_1] if isinstance(class_1, int) else list(class_1)
        self._max_samples = n_samples
        self._seed = seed
        self._X: Optional[np.ndarray] = None
        self._y: Optional[np.ndarray] = None

    @property
    def name(self) -> str:
        c0 = ",".join(map(str, self._class_0))
        c1 = ",".join(map(str, self._class_1))
        return f"mnist_binary_{c0}_vs_{c1}"

    @property
    def n_samples(self) -> int:
        if self._X is None:
            self._load_data()
        return len(self._X)

    def _load_data(self):
        """Load and filter MNIST data."""
        from sklearn.datasets import load_digits

        # Load digits (8x8 images, 0-9)
        digits = load_digits()
        X_all = digits.data  # (n_samples, 64)
        y_all = digits.target  # (n_samples,)

        # Filter to binary classes
        mask_0 = np.isin(y_all, self._class_0)
        mask_1 = np.isin(y_all, self._class_1)
        mask = mask_0 | mask_1

        X = X_all[mask]
        y = np.where(mask_0[mask], 0, 1)

        # Shuffle
        rng = np.random.default_rng(self._seed)
        indices = rng.permutation(len(X))
        X = X[indices]
        y = y[indices]

        # Limit samples
        if self._max_samples is not None:
            X = X[: self._max_samples]
            y = y[: self._max_samples]

        self._X = X
        self._y = y

    def load(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Load the raw MNIST binary dataset.

        Returns:
            Tuple of (X, y) where X has shape (n_samples, 64) and
            y has shape (n_samples,) with binary labels.
        """
        if self._X is None:
            self._load_data()
        return self._X.copy(), self._y.copy()

    def prepare_for_qubits(
        self, n_qubits: int, normalize: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Prepare MNIST data for quantum encoding.

        For small n_qubits (< 6), uses average pooling to downsample.
        For large n_qubits (> 6), zero-pads to match dimension.

        Args:
            n_qubits: Number of qubits (features will be 2^n_qubits).
            normalize: Whether to L2-normalize each sample.

        Returns:
            Tuple of (X, y) where X has shape (n_samples, 2^n_qubits).
        """
        X, y = self.load()
        target_dim = 1 << n_qubits

        # Original MNIST digits are 8x8 = 64 features
        if target_dim == 64:
            pass  # No resize needed
        elif target_dim < 64:
            # Downsample using reshape and mean
            X = self._downsample_images(X, target_dim)
        else:
            # Zero-pad
            X = self._resize_features(X, target_dim, method="pad")

        if normalize:
            X = self._normalize(X)

        return X, y

    def _downsample_images(self, X: np.ndarray, target_dim: int) -> np.ndarray:
        """Downsample 8x8 images to smaller resolution.

        Args:
            X: Image data of shape (n_samples, 64).
            target_dim: Target dimension (must be power of 2, <= 64).

        Returns:
            Downsampled images of shape (n_samples, target_dim).
        """
        n_samples = len(X)

        # Calculate target grid size
        target_size = int(np.sqrt(target_dim))
        if target_size * target_size != target_dim:
            # Not a perfect square, use padding/truncation
            return self._resize_features(X, target_dim, method="pad")

        # Pooling factor
        pool_factor = 8 // target_size

        # Vectorized average pooling using reshape
        # Reshape: (n, 8, 8) -> (n, target_size, pool_factor, target_size, pool_factor)
        # Then mean over the pooling dimensions (2, 4)
        images = X.reshape(n_samples, 8, 8)
        result = images.reshape(
            n_samples, target_size, pool_factor, target_size, pool_factor
        ).mean(axis=(2, 4))

        return result.reshape(n_samples, target_dim)


# Register the dataset
register_dataset("mnist", MNISTBinaryDataset)
register_dataset("mnist_binary", MNISTBinaryDataset)
