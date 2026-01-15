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

"""Core benchmark infrastructure for QDP.

This package provides the foundational components for statistically rigorous
benchmarking:

- **statistics**: Statistical measurement framework (BenchmarkStats, StatisticalRunner)
- **timing**: Timer implementations (WallClockTimer, CUDATimer)
- **cuda**: GPU utilities (sync_cuda, clear_gpu_caches, warmup_gpu)
- **data**: Data generation utilities (build_sample, normalize_batch, prefetched_batches)

Example:
    >>> from benchmark.core import StatisticalRunner, BenchmarkStats
    >>> runner = StatisticalRunner(warmup_runs=3, measurement_runs=10)
    >>> stats, result = runner.run(my_benchmark_fn)
    >>> print(stats.summary())
    10.5 +/- 0.3 ms (median=10.4, p95=11.2, n=10)
"""

from .cuda import (
    clear_gpu_caches,
    get_device_count,
    get_gpu_memory_info,
    is_cuda_available,
    reset_peak_memory_stats,
    sync_cuda,
    warmup_gpu,
)
from .data import (
    build_sample,
    generate_random_batch,
    normalize_batch,
    prefetched_batches,
)
from .results import BenchmarkRun, ResultsStore
from .statistics import BenchmarkStats, StatisticalRunner
from .timing import BaseTimer, CUDATimer, WallClockTimer, timed

__all__ = [
    # Statistics
    "BenchmarkStats",
    "StatisticalRunner",
    # Timing
    "BaseTimer",
    "WallClockTimer",
    "CUDATimer",
    "timed",
    # CUDA utilities
    "is_cuda_available",
    "get_device_count",
    "sync_cuda",
    "clear_gpu_caches",
    "warmup_gpu",
    "get_gpu_memory_info",
    "reset_peak_memory_stats",
    # Data utilities
    "build_sample",
    "normalize_batch",
    "prefetched_batches",
    "generate_random_batch",
    # Results
    "BenchmarkRun",
    "ResultsStore",
]
