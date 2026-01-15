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

"""Data generation and preprocessing utilities for benchmarks.

This module provides shared utilities for generating test data, normalizing
batches, and prefetching data in background threads.

Example:
    >>> from benchmark.core.data import build_sample, prefetched_batches
    >>> sample = build_sample(seed=42, vector_len=16)
    >>> for batch in prefetched_batches(total_batches=10, batch_size=64,
    ...                                  vector_len=256, prefetch=4):
    ...     process(batch)
"""

from __future__ import annotations

import queue
import threading
from typing import Generator, Optional

import numpy as np


def build_sample(seed: int, vector_len: int) -> np.ndarray:
    """Generate a deterministic test sample.

    Uses bit manipulation for fast, reproducible generation. This matches
    the pattern used in qdp-core/examples/dataloader_throughput.rs for
    consistent cross-language testing.

    Args:
        seed: Seed value for deterministic generation.
        vector_len: Length of the output vector (should be power of 2).

    Returns:
        1D numpy array of float64 values in range [0, 1).

    Example:
        >>> sample = build_sample(42, 16)
        >>> sample.shape
        (16,)
        >>> sample.dtype
        dtype('float64')
    """
    mask = np.uint64(vector_len - 1)
    scale = 1.0 / vector_len
    idx = np.arange(vector_len, dtype=np.uint64)
    mixed = (idx + np.uint64(seed)) & mask
    return mixed.astype(np.float64) * scale


def normalize_batch(batch: np.ndarray, epsilon: float = 0.0) -> np.ndarray:
    """L2 normalize each row of a 2D array.

    For amplitude encoding, quantum states must be normalized such that
    the sum of squared amplitudes equals 1.

    Args:
        batch: 2D array of shape (n_samples, features).
        epsilon: Small value added to norms to avoid division by zero.
            If 0.0 (default), zero vectors have their norm set to 1.0.

    Returns:
        Normalized array with same shape where each row has unit L2 norm.

    Example:
        >>> batch = np.array([[3.0, 4.0], [0.0, 0.0]])
        >>> normalized = normalize_batch(batch)
        >>> np.linalg.norm(normalized[0])  # Should be 1.0
        1.0
    """
    norms = np.linalg.norm(batch, axis=1, keepdims=True)
    if epsilon > 0:
        norms = np.maximum(norms, epsilon)
    else:
        norms[norms == 0] = 1.0
    return batch / norms


def prefetched_batches(
    total_batches: int,
    batch_size: int,
    vector_len: int,
    prefetch: int = 16,
    seed_offset: int = 0,
    normalize: bool = False,
) -> Generator[np.ndarray, None, None]:
    """Generate batches in a background thread with prefetching.

    This simulates a realistic DataLoader pipeline where CPU prepares
    data while GPU processes the previous batch. The background thread
    fills a queue with pre-generated batches for low-latency consumption.

    Args:
        total_batches: Number of batches to generate.
        batch_size: Number of samples per batch.
        vector_len: Length of each sample vector (2^n_qubits).
        prefetch: Size of prefetch queue (default: 16).
        seed_offset: Base seed for deterministic generation.
        normalize: If True, L2 normalize each batch before yielding.

    Yields:
        numpy arrays of shape (batch_size, vector_len).

    Example:
        >>> batches = list(prefetched_batches(5, 10, 16, prefetch=2))
        >>> len(batches)
        5
        >>> batches[0].shape
        (10, 16)
    """
    q: queue.Queue[Optional[np.ndarray]] = queue.Queue(maxsize=prefetch)

    def producer() -> None:
        for batch_idx in range(total_batches):
            base = seed_offset + batch_idx * batch_size
            batch = np.stack(
                [build_sample(base + i, vector_len) for i in range(batch_size)]
            )
            if normalize:
                batch = normalize_batch(batch)
            q.put(batch)
        q.put(None)  # Sentinel to signal completion

    threading.Thread(target=producer, daemon=True).start()

    while True:
        batch = q.get()
        if batch is None:
            break
        yield batch


def generate_random_batch(
    batch_size: int,
    vector_len: int,
    seed: Optional[int] = None,
    normalize: bool = True,
) -> np.ndarray:
    """Generate a random batch of vectors.

    Unlike build_sample which is deterministic, this uses numpy's random
    number generator for more varied test data.

    Args:
        batch_size: Number of samples in the batch.
        vector_len: Length of each sample vector.
        seed: Random seed for reproducibility. If None, uses random state.
        normalize: If True, L2 normalize the batch.

    Returns:
        numpy array of shape (batch_size, vector_len).
    """
    rng = np.random.default_rng(seed)
    batch = rng.random((batch_size, vector_len), dtype=np.float64)
    if normalize:
        batch = normalize_batch(batch)
    return batch
