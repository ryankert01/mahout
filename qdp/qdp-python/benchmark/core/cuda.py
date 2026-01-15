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

"""CUDA utility functions for benchmarking.

This module provides helper functions for GPU synchronization, cache management,
and warmup operations needed for fair and reproducible GPU benchmarks.

Example:
    >>> from benchmark.core.cuda import clear_gpu_caches, warmup_gpu
    >>> warmup_gpu(device_id=0)
    >>> clear_gpu_caches()
    >>> # Now run benchmarks...
"""

from __future__ import annotations

import gc
from typing import Optional


def is_cuda_available() -> bool:
    """Check if CUDA is available via PyTorch.

    Returns:
        True if PyTorch is installed and CUDA is available.
    """
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def get_device_count() -> int:
    """Get the number of available CUDA devices.

    Returns:
        Number of CUDA devices, or 0 if CUDA is not available.
    """
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.device_count()
        return 0
    except ImportError:
        return 0


def sync_cuda(device_id: Optional[int] = None) -> None:
    """Synchronize CUDA device to ensure all operations are complete.

    This is a no-op if CUDA is not available.

    Args:
        device_id: Specific device to synchronize. If None, synchronizes
            the current device.
    """
    try:
        import torch

        if torch.cuda.is_available():
            if device_id is not None:
                torch.cuda.synchronize(device_id)
            else:
                torch.cuda.synchronize()
    except ImportError:
        pass


def clear_gpu_caches(gc_collect: bool = True) -> None:
    """Clear GPU caches and optionally run garbage collection.

    This should be called between benchmark runs to ensure fair comparison
    by preventing cached memory from affecting subsequent measurements.

    Args:
        gc_collect: If True, also run Python garbage collection before
            clearing GPU caches.
    """
    if gc_collect:
        gc.collect()

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        pass


def warmup_gpu(device_id: int = 0, iterations: int = 10) -> None:
    """Run dummy operations to warm up GPU for stable benchmarks.

    GPU kernels need to be compiled and loaded on first use, which can
    cause cold-start latency. This function runs small tensor operations
    to warm up the GPU before benchmarking.

    Args:
        device_id: CUDA device index to warm up.
        iterations: Number of warmup iterations to run.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return

        with torch.cuda.device(device_id):
            # Small tensor operations to warm up kernels
            for _ in range(iterations):
                x = torch.randn(256, 256, device=f"cuda:{device_id}")
                _ = x @ x.T
            torch.cuda.synchronize(device_id)
            torch.cuda.empty_cache()
    except ImportError:
        pass


def get_gpu_memory_info(device_id: int = 0) -> dict:
    """Get GPU memory usage information.

    Args:
        device_id: CUDA device index.

    Returns:
        Dictionary with memory info:
        - allocated: Currently allocated memory in bytes
        - reserved: Total reserved memory in bytes
        - max_allocated: Peak allocated memory in bytes
        - total: Total device memory in bytes
        Returns empty dict if CUDA is not available.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return {}

        with torch.cuda.device(device_id):
            return {
                "allocated": torch.cuda.memory_allocated(device_id),
                "reserved": torch.cuda.memory_reserved(device_id),
                "max_allocated": torch.cuda.max_memory_allocated(device_id),
                "total": torch.cuda.get_device_properties(device_id).total_memory,
            }
    except ImportError:
        return {}


def reset_peak_memory_stats(device_id: int = 0) -> None:
    """Reset peak memory statistics for a device.

    Call this before a benchmark run to track peak memory usage
    during that specific run.

    Args:
        device_id: CUDA device index.
    """
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device_id)
    except ImportError:
        pass
