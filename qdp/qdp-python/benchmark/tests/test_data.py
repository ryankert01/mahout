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

"""Tests for benchmark.core.data module."""

import numpy as np
import pytest

from benchmark.core.data import (
    build_sample,
    generate_random_batch,
    normalize_batch,
    prefetched_batches,
)


class TestBuildSample:
    """Tests for build_sample function."""

    def test_deterministic(self):
        """Test that same seed produces same output."""
        s1 = build_sample(42, 16)
        s2 = build_sample(42, 16)

        np.testing.assert_array_equal(s1, s2)

    def test_different_seeds(self):
        """Test that different seeds produce different outputs."""
        s1 = build_sample(0, 16)
        s2 = build_sample(1, 16)

        assert not np.array_equal(s1, s2)

    def test_shape(self):
        """Test output shape."""
        sample = build_sample(0, 64)

        assert sample.shape == (64,)

    def test_dtype(self):
        """Test output dtype."""
        sample = build_sample(0, 16)

        assert sample.dtype == np.float64

    def test_value_range(self):
        """Test that values are in expected range."""
        sample = build_sample(0, 256)

        assert np.all(sample >= 0)
        assert np.all(sample < 1)

    @pytest.mark.parametrize("vector_len", [4, 16, 64, 256, 1024])
    def test_various_sizes(self, vector_len):
        """Test with various vector lengths."""
        sample = build_sample(42, vector_len)

        assert sample.shape == (vector_len,)
        assert np.all(np.isfinite(sample))


class TestNormalizeBatch:
    """Tests for normalize_batch function."""

    def test_unit_norm(self, sample_batch):
        """Test that normalized rows have unit norm."""
        normalized = normalize_batch(sample_batch)
        norms = np.linalg.norm(normalized, axis=1)

        np.testing.assert_allclose(norms, 1.0, atol=1e-10)

    def test_zero_vector(self):
        """Test handling of zero vectors."""
        batch = np.zeros((3, 4))
        normalized = normalize_batch(batch)

        # Zero vectors should not produce NaN/inf
        assert np.all(np.isfinite(normalized))

    def test_zero_vector_with_epsilon(self):
        """Test epsilon parameter for zero handling."""
        batch = np.zeros((3, 4))
        normalized = normalize_batch(batch, epsilon=1e-10)

        assert np.all(np.isfinite(normalized))

    def test_preserves_shape(self, sample_batch):
        """Test that shape is preserved."""
        normalized = normalize_batch(sample_batch)

        assert normalized.shape == sample_batch.shape

    def test_single_nonzero_row(self):
        """Test batch with mix of zero and non-zero rows."""
        batch = np.array([[3.0, 4.0], [0.0, 0.0], [1.0, 0.0]])
        normalized = normalize_batch(batch)

        # First row: 3,4 -> 0.6, 0.8 (norm = 5)
        np.testing.assert_allclose(normalized[0], [0.6, 0.8], atol=1e-10)
        # Third row: 1,0 -> 1,0 (already unit)
        np.testing.assert_allclose(normalized[2], [1.0, 0.0], atol=1e-10)


class TestPrefetchedBatches:
    """Tests for prefetched_batches generator."""

    def test_generates_correct_count(self):
        """Test correct number of batches generated."""
        batches = list(prefetched_batches(5, 10, 16, prefetch=2))

        assert len(batches) == 5

    def test_batch_shape(self):
        """Test batch shapes."""
        batches = list(prefetched_batches(3, 10, 16))

        for batch in batches:
            assert batch.shape == (10, 16)

    def test_deterministic_with_seed_offset(self):
        """Test reproducibility with seed_offset."""
        b1 = list(prefetched_batches(2, 4, 8, seed_offset=100))
        b2 = list(prefetched_batches(2, 4, 8, seed_offset=100))

        for a, b in zip(b1, b2):
            np.testing.assert_array_equal(a, b)

    def test_different_seed_offsets(self):
        """Test that different seed offsets produce different data."""
        # Use larger vector_len to avoid collisions in build_sample's bit manipulation
        b1 = list(prefetched_batches(1, 4, 256, seed_offset=0))
        b2 = list(prefetched_batches(1, 4, 256, seed_offset=1000))

        assert not np.array_equal(b1[0], b2[0])

    def test_normalize_option(self):
        """Test normalize option produces unit-norm rows."""
        batches = list(prefetched_batches(2, 5, 16, normalize=True))

        for batch in batches:
            norms = np.linalg.norm(batch, axis=1)
            np.testing.assert_allclose(norms, 1.0, atol=1e-10)

    def test_without_normalize(self):
        """Test default (no normalize) doesn't produce unit norms."""
        batches = list(prefetched_batches(2, 5, 16, normalize=False))

        for batch in batches:
            norms = np.linalg.norm(batch, axis=1)
            # Most rows won't have unit norm by chance
            assert not np.allclose(norms, 1.0)

    def test_prefetch_queue_size(self):
        """Test that prefetch parameter is respected."""
        # Small prefetch should still work
        batches = list(prefetched_batches(10, 4, 8, prefetch=1))
        assert len(batches) == 10

        # Large prefetch should also work
        batches = list(prefetched_batches(10, 4, 8, prefetch=20))
        assert len(batches) == 10


class TestGenerateRandomBatch:
    """Tests for generate_random_batch function."""

    def test_shape(self):
        """Test output shape."""
        batch = generate_random_batch(10, 16)

        assert batch.shape == (10, 16)

    def test_reproducible_with_seed(self):
        """Test reproducibility with seed."""
        b1 = generate_random_batch(5, 8, seed=42)
        b2 = generate_random_batch(5, 8, seed=42)

        np.testing.assert_array_equal(b1, b2)

    def test_different_seeds(self):
        """Test different seeds produce different results."""
        b1 = generate_random_batch(5, 8, seed=1)
        b2 = generate_random_batch(5, 8, seed=2)

        assert not np.array_equal(b1, b2)

    def test_normalized_by_default(self):
        """Test default normalization."""
        batch = generate_random_batch(10, 16, seed=42)
        norms = np.linalg.norm(batch, axis=1)

        np.testing.assert_allclose(norms, 1.0, atol=1e-10)

    def test_unnormalized_option(self):
        """Test normalize=False option."""
        batch = generate_random_batch(10, 16, seed=42, normalize=False)
        norms = np.linalg.norm(batch, axis=1)

        # Random data won't have unit norms
        assert not np.allclose(norms, 1.0)

    def test_dtype(self):
        """Test output dtype."""
        batch = generate_random_batch(5, 8)

        assert batch.dtype == np.float64
