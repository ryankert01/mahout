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
CUDA event timing and warmup utilities for fair benchmarking.

This module provides functions for precise GPU timing using CUDA events,
warmup mechanisms to eliminate JIT compilation overhead, and cache clearing
strategies for reproducible benchmarks.
"""

import gc
from typing import Callable, List, Optional

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def warmup(func: Callable, warmup_iters: int = 3) -> None:
    """
    Run function multiple times to warm up JIT, caches, etc.

    This function executes the provided callable multiple times to ensure
    that JIT compilation, GPU kernel compilation, and cache warmup are
    completed before actual benchmark measurements begin.

    Args:
        func: Callable to warm up
        warmup_iters: Number of warmup iterations (default: 3)

    Returns:
        None

    Raises:
        RuntimeError: If CUDA synchronization fails

    Example:
        >>> def my_kernel():
        ...     x = torch.randn(1000, 1000, device='cuda')
        ...     return x @ x.T
        >>> warmup(my_kernel, warmup_iters=5)
    """
    for _ in range(warmup_iters):
        func()
        if HAS_TORCH and torch.cuda.is_available():
            torch.cuda.synchronize()


def clear_all_caches() -> None:
    """
    Clear all caches to ensure fair benchmarking.

    This function performs comprehensive cache clearing including:
    - Python garbage collection
    - PyTorch GPU cache
    - CUDA synchronization
    - Peak memory statistics reset

    Should be called:
    - Before warmup runs
    - Between different frameworks
    - Between repeated runs (optional, for conservative measurements)

    Returns:
        None

    Example:
        >>> clear_all_caches()
        >>> # Run benchmark
        >>> result = benchmark_function()
    """
    # Python garbage collection
    gc.collect()

    if HAS_TORCH and torch.cuda.is_available():
        # Clear PyTorch GPU cache
        torch.cuda.empty_cache()

        # Synchronize all CUDA operations
        torch.cuda.synchronize()

        # Reset peak memory stats (useful for memory profiling)
        torch.cuda.reset_peak_memory_stats()


def clear_l2_cache(cache_size_mb: int = 1024) -> None:
    """
    Clear GPU L2 cache by allocating and freeing a large tensor.

    This is more invasive than clear_all_caches() and may not be needed
    for all benchmarks. Use only when you need very conservative measurements.

    Args:
        cache_size_mb: Size of tensor to allocate in MB (default: 1024)
                      Should exceed typical L2 cache size (~40-50 MB on most GPUs)

    Returns:
        None

    Example:
        >>> clear_l2_cache(cache_size_mb=2048)  # Clear with 2GB tensor
    """
    if not HAS_TORCH or not torch.cuda.is_available():
        return

    # Calculate tensor dimensions for requested size
    # 4 bytes per float32 element
    num_elements = (cache_size_mb * 1024 * 1024) // 4

    # Allocate and immediately free
    cache_clear_tensor = torch.empty((num_elements,), dtype=torch.float32, device="cuda")
    del cache_clear_tensor
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def benchmark_with_cuda_events(
    func: Callable,
    warmup_iters: int = 3,
    repeat: int = 100,
    clear_cache_between_runs: bool = False,
) -> List[float]:
    """
    Benchmark a function using CUDA events for precise timing.

    This function provides the most accurate GPU timing by using CUDA events
    to measure kernel execution time. It includes warmup iterations and
    optional cache clearing between runs.

    Args:
        func: Callable to benchmark
        warmup_iters: Number of warmup iterations (default: 3)
        repeat: Number of measurement iterations (default: 100)
        clear_cache_between_runs: Whether to clear cache between each run
                                  (default: False, for less conservative but faster measurement)

    Returns:
        List of execution times in milliseconds

    Raises:
        RuntimeError: If CUDA is not available
        ValueError: If warmup_iters or repeat are not positive

    Example:
        >>> def my_operation():
        ...     x = torch.randn(1000, 1000, device='cuda')
        ...     return (x @ x.T).sum()
        >>> timings = benchmark_with_cuda_events(my_operation, warmup=5, repeat=100)
        >>> print(f"Mean time: {sum(timings)/len(timings):.2f} ms")
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required for CUDA event timing")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    if warmup_iters <= 0 or repeat <= 0:
        raise ValueError("warmup_iters and repeat must be positive integers")

    # Warmup phase
    warmup(func, warmup_iters=warmup_iters)

    # Clear cache before measurement
    clear_all_caches()

    # Measurement phase
    timings = []
    for _ in range(repeat):
        # Optional: clear cache between runs for more conservative measurement
        if clear_cache_between_runs:
            clear_all_caches()

        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        func()
        end_event.record()

        torch.cuda.synchronize()
        timings.append(start_event.elapsed_time(end_event))  # Returns ms

    return timings


def benchmark_cpu_function(
    func: Callable,
    warmup_iters: int = 3,
    repeat: int = 100,
) -> List[float]:
    """
    Benchmark a CPU function using time.perf_counter().

    For CPU-only operations, CUDA events are not applicable.
    This function uses high-resolution CPU timing.

    Args:
        func: Callable to benchmark (CPU operation)
        warmup_iters: Number of warmup iterations (default: 3)
        repeat: Number of measurement iterations (default: 100)

    Returns:
        List of execution times in milliseconds

    Example:
        >>> import numpy as np
        >>> def cpu_operation():
        ...     x = np.random.rand(1000, 1000)
        ...     return x @ x.T
        >>> timings = benchmark_cpu_function(cpu_operation, repeat=50)
    """
    import time

    # Warmup phase
    for _ in range(warmup_iters):
        func()

    # Clear cache before measurement
    gc.collect()

    # Measurement phase
    timings = []
    for _ in range(repeat):
        start = time.perf_counter()
        func()
        end = time.perf_counter()
        timings.append((end - start) * 1000)  # Convert to ms

    return timings
