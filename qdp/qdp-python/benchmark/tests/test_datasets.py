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

"""Tests for benchmark.datasets module."""

import numpy as np
import pytest

from benchmark.datasets import SyntheticDataset, get_dataset
from benchmark.datasets.base import list_datasets, register_dataset


class TestSyntheticDataset:
    """Tests for SyntheticDataset."""

    def test_basic_creation(self):
        """Test basic dataset creation."""
        dataset = SyntheticDataset(n_samples=100, seed=42)
        assert dataset.name == "synthetic"
        assert dataset.n_samples == 100

    def test_prepare_for_qubits(self):
        """Test prepare_for_qubits generates correct shape."""
        dataset = SyntheticDataset(n_samples=50, seed=42)
        X, y = dataset.prepare_for_qubits(n_qubits=4)

        assert X.shape == (50, 16)  # 2^4 = 16
        assert y is None  # Synthetic has no labels

    def test_normalization(self):
        """Test that normalization produces unit vectors."""
        dataset = SyntheticDataset(n_samples=10, seed=42)
        X, _ = dataset.prepare_for_qubits(n_qubits=4, normalize=True)

        norms = np.linalg.norm(X, axis=1)
        np.testing.assert_array_almost_equal(norms, np.ones(10), decimal=10)

    def test_no_normalization(self):
        """Test that normalize=False skips normalization."""
        dataset = SyntheticDataset(n_samples=10, seed=42)
        X, _ = dataset.prepare_for_qubits(n_qubits=4, normalize=False)

        norms = np.linalg.norm(X, axis=1)
        # Raw random vectors should not be unit vectors
        assert not np.allclose(norms, np.ones(10))

    def test_reproducibility(self):
        """Test that same seed produces same data."""
        dataset1 = SyntheticDataset(n_samples=20, seed=123)
        dataset2 = SyntheticDataset(n_samples=20, seed=123)

        X1, _ = dataset1.prepare_for_qubits(n_qubits=3)
        X2, _ = dataset2.prepare_for_qubits(n_qubits=3)

        np.testing.assert_array_equal(X1, X2)

    def test_different_seeds(self):
        """Test that different seeds produce different data."""
        dataset1 = SyntheticDataset(n_samples=20, seed=1)
        dataset2 = SyntheticDataset(n_samples=20, seed=2)

        X1, _ = dataset1.prepare_for_qubits(n_qubits=3, normalize=False)
        X2, _ = dataset2.prepare_for_qubits(n_qubits=3, normalize=False)

        assert not np.allclose(X1, X2)

    def test_generate_batch(self):
        """Test batch generation."""
        dataset = SyntheticDataset(n_samples=100, seed=42)
        batch = dataset.generate_batch(batch_size=32, n_qubits=4, batch_idx=0)

        assert batch.shape == (32, 16)

    def test_batch_reproducibility(self):
        """Test that same batch_idx produces same batch."""
        dataset = SyntheticDataset(n_samples=100, seed=42)
        batch1 = dataset.generate_batch(batch_size=32, n_qubits=4, batch_idx=5)
        batch2 = dataset.generate_batch(batch_size=32, n_qubits=4, batch_idx=5)

        np.testing.assert_array_equal(batch1, batch2)


class TestGetDataset:
    """Tests for get_dataset function."""

    def test_get_synthetic(self):
        """Test getting synthetic dataset by name."""
        dataset = get_dataset("synthetic", n_samples=50)
        assert dataset.name == "synthetic"
        assert dataset.n_samples == 50

    def test_case_insensitive(self):
        """Test that dataset names are case-insensitive."""
        dataset = get_dataset("SYNTHETIC", n_samples=10)
        assert dataset.name == "synthetic"

    def test_unknown_dataset_raises(self):
        """Test that unknown dataset raises ValueError."""
        with pytest.raises(ValueError, match="Unknown dataset"):
            get_dataset("nonexistent")


class TestListDatasets:
    """Tests for list_datasets function."""

    def test_contains_synthetic(self):
        """Test that synthetic is in the list."""
        datasets = list_datasets()
        assert "synthetic" in datasets


class TestRegisterDataset:
    """Tests for register_dataset function."""

    def test_register_custom(self):
        """Test registering a custom dataset."""

        class CustomDataset(SyntheticDataset):
            @property
            def name(self):
                return "custom_test"

        register_dataset("custom_test", CustomDataset)

        dataset = get_dataset("custom_test", n_samples=10)
        assert dataset.name == "custom_test"


# Skip sklearn-dependent tests if sklearn not available
sklearn = pytest.importorskip("sklearn")


class TestMNISTBinaryDataset:
    """Tests for MNISTBinaryDataset."""

    def test_basic_creation(self):
        """Test basic dataset creation."""
        from benchmark.datasets import MNISTBinaryDataset

        dataset = MNISTBinaryDataset(class_0=0, class_1=1, n_samples=50)
        assert "mnist" in dataset.name.lower()

    def test_prepare_for_qubits(self):
        """Test prepare_for_qubits generates correct shape."""
        from benchmark.datasets import MNISTBinaryDataset

        dataset = MNISTBinaryDataset(class_0=0, class_1=1, n_samples=50)
        X, y = dataset.prepare_for_qubits(n_qubits=6)  # 64 features (8x8)

        assert X.shape[0] <= 50
        assert X.shape[1] == 64
        assert y is not None
        assert set(np.unique(y)).issubset({0, 1})

    def test_downsampling(self):
        """Test downsampling to smaller qubit count."""
        from benchmark.datasets import MNISTBinaryDataset

        dataset = MNISTBinaryDataset(class_0=0, class_1=1, n_samples=20)
        X, y = dataset.prepare_for_qubits(n_qubits=4)  # 16 features (4x4)

        assert X.shape == (20, 16)

    def test_upsampling(self):
        """Test padding to larger qubit count."""
        from benchmark.datasets import MNISTBinaryDataset

        dataset = MNISTBinaryDataset(class_0=0, class_1=1, n_samples=20)
        X, y = dataset.prepare_for_qubits(n_qubits=8)  # 256 features

        assert X.shape == (20, 256)

    def test_binary_labels(self):
        """Test that labels are binary."""
        from benchmark.datasets import MNISTBinaryDataset

        dataset = MNISTBinaryDataset(class_0=3, class_1=8, n_samples=100)
        X, y = dataset.prepare_for_qubits(n_qubits=6)

        assert set(np.unique(y)).issubset({0, 1})

    def test_get_by_name(self):
        """Test getting MNIST by name."""
        dataset = get_dataset("mnist", class_0=0, class_1=1, n_samples=10)
        assert "mnist" in dataset.name.lower()


class TestIrisBinaryDataset:
    """Tests for IrisBinaryDataset."""

    def test_basic_creation(self):
        """Test basic dataset creation."""
        from benchmark.datasets import IrisBinaryDataset

        dataset = IrisBinaryDataset(class_0=0, class_1=1)
        assert "iris" in dataset.name.lower()

    def test_prepare_for_qubits(self):
        """Test prepare_for_qubits generates correct shape."""
        from benchmark.datasets import IrisBinaryDataset

        dataset = IrisBinaryDataset(class_0=0, class_1=1)
        X, y = dataset.prepare_for_qubits(n_qubits=2)  # 4 features

        assert X.shape[1] == 4
        assert y is not None
        assert set(np.unique(y)) == {0, 1}

    def test_padding(self):
        """Test padding to more qubits."""
        from benchmark.datasets import IrisBinaryDataset

        dataset = IrisBinaryDataset(class_0=0, class_1=1)
        X, y = dataset.prepare_for_qubits(n_qubits=4)  # 16 features

        assert X.shape[1] == 16

    def test_invalid_class(self):
        """Test that invalid class raises error."""
        from benchmark.datasets import IrisBinaryDataset

        with pytest.raises(ValueError):
            IrisBinaryDataset(class_0=0, class_1=5)

    def test_same_class_raises(self):
        """Test that same class for both raises error."""
        from benchmark.datasets import IrisBinaryDataset

        with pytest.raises(ValueError):
            IrisBinaryDataset(class_0=1, class_1=1)

    def test_get_by_name(self):
        """Test getting Iris by name."""
        dataset = get_dataset("iris", class_0=0, class_1=2)
        assert "iris" in dataset.name.lower()


class TestSyntheticBlobsDataset:
    """Tests for SyntheticBlobsDataset."""

    def test_basic_creation(self):
        """Test basic dataset creation."""
        from benchmark.datasets import SyntheticBlobsDataset

        dataset = SyntheticBlobsDataset(n_samples=100, seed=42)
        assert "blobs" in dataset.name.lower()
        assert dataset.n_samples == 100

    def test_prepare_for_qubits(self):
        """Test prepare_for_qubits generates correct shape."""
        from benchmark.datasets import SyntheticBlobsDataset

        dataset = SyntheticBlobsDataset(n_samples=100, seed=42)
        X, y = dataset.prepare_for_qubits(n_qubits=4)

        assert X.shape == (100, 16)
        assert y is not None
        assert set(np.unique(y)) == {0, 1}

    def test_reproducibility(self):
        """Test that same seed produces same data."""
        from benchmark.datasets import SyntheticBlobsDataset

        dataset1 = SyntheticBlobsDataset(n_samples=50, seed=123)
        dataset2 = SyntheticBlobsDataset(n_samples=50, seed=123)

        X1, y1 = dataset1.prepare_for_qubits(n_qubits=3)
        X2, y2 = dataset2.prepare_for_qubits(n_qubits=3)

        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)

    def test_get_by_name(self):
        """Test getting blobs by name."""
        dataset = get_dataset("blobs", n_samples=50)
        assert "blobs" in dataset.name.lower()


class TestFullMNISTDataset:
    """Tests for FullMNISTDataset."""

    def test_basic_creation(self):
        """Test basic dataset creation."""
        from benchmark.datasets import FullMNISTDataset

        dataset = FullMNISTDataset(n_samples=100)
        assert "mnist_full" in dataset.name.lower()

    def test_sample_count(self):
        """Test that we can load many samples."""
        from benchmark.datasets import FullMNISTDataset

        # Test with a moderate sample count
        dataset = FullMNISTDataset(n_samples=5000)
        assert dataset.n_samples == 5000

    def test_prepare_for_qubits(self):
        """Test prepare_for_qubits generates correct shape."""
        from benchmark.datasets import FullMNISTDataset

        dataset = FullMNISTDataset(n_samples=100)
        X, y = dataset.prepare_for_qubits(n_qubits=9)  # 512 features

        assert X.shape == (100, 512)
        assert y is not None
        assert len(y) == 100

    def test_downsampling(self):
        """Test downsampling to smaller qubit count."""
        from benchmark.datasets import FullMNISTDataset

        dataset = FullMNISTDataset(n_samples=50)
        X, y = dataset.prepare_for_qubits(n_qubits=6)  # 64 features (8x8)

        assert X.shape == (50, 64)

    def test_upsampling(self):
        """Test padding to larger qubit count."""
        from benchmark.datasets import FullMNISTDataset

        dataset = FullMNISTDataset(n_samples=50)
        X, y = dataset.prepare_for_qubits(n_qubits=10)  # 1024 features

        assert X.shape == (50, 1024)

    def test_binary_filtering(self):
        """Test binary classification filtering."""
        from benchmark.datasets import FullMNISTDataset

        dataset = FullMNISTDataset(class_0=0, class_1=1, n_samples=200)
        X, y = dataset.prepare_for_qubits(n_qubits=9)

        assert set(np.unique(y)).issubset({0, 1})
        assert "0" in dataset.name and "1" in dataset.name

    def test_multiclass_labels(self):
        """Test multiclass labels (0-9)."""
        from benchmark.datasets import FullMNISTDataset

        dataset = FullMNISTDataset(n_samples=1000)
        X, y = dataset.prepare_for_qubits(n_qubits=9)

        # Should have multiple digit classes
        assert len(np.unique(y)) > 2

    def test_normalization(self):
        """Test that normalization produces unit vectors."""
        from benchmark.datasets import FullMNISTDataset

        dataset = FullMNISTDataset(n_samples=50)
        X, _ = dataset.prepare_for_qubits(n_qubits=8, normalize=True)

        norms = np.linalg.norm(X, axis=1)
        np.testing.assert_array_almost_equal(norms, np.ones(50), decimal=5)

    def test_get_by_name(self):
        """Test getting full MNIST by name."""
        dataset = get_dataset("mnist_full", n_samples=100)
        assert "mnist_full" in dataset.name.lower()

    def test_get_by_alias(self):
        """Test getting full MNIST by alias."""
        dataset = get_dataset("full_mnist", n_samples=100)
        assert "mnist_full" in dataset.name.lower()
