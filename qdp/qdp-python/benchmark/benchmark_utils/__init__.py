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
Benchmark utilities for Apache Mahout QDP.

This package provides utilities for fair, reproducible, and statistically
rigorous benchmarking of quantum data processing pipelines.

Modules:
    timing: CUDA event-based timing and warmup utilities
    statistics: Statistical computations for benchmark results
    visualization: Publication-ready plot generation
    config: Configuration loading and management
"""

from .timing import warmup, clear_all_caches, benchmark_with_cuda_events
from .statistics import compute_statistics, filter_outliers
from .visualization import BenchmarkVisualizer

__all__ = [
    "warmup",
    "clear_all_caches",
    "benchmark_with_cuda_events",
    "compute_statistics",
    "filter_outliers",
    "BenchmarkVisualizer",
]
