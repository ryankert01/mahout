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

"""Benchmark datasets for QDP.

This package provides standardized datasets for benchmarking quantum
data processing pipelines:

- **SyntheticDataset**: Random vectors with reproducible seeds
- **MNISTBinaryDataset**: Binary MNIST classification
- **IrisBinaryDataset**: Binary Iris classification
- **SyntheticBlobsDataset**: Configurable synthetic clusters

Example:
    >>> from benchmark.datasets import SyntheticDataset
    >>> dataset = SyntheticDataset(n_samples=100, seed=42)
    >>> X, y = dataset.prepare_for_qubits(n_qubits=4)
    >>> X.shape  # (100, 16) - ready for 4-qubit encoding
"""

from .base import BenchmarkDataset, get_dataset

# Import implementations
from .synthetic import SyntheticDataset

# Tabular datasets (require sklearn)
try:
    from .tabular import IrisBinaryDataset, SyntheticBlobsDataset

    HAS_TABULAR = True
except ImportError:
    HAS_TABULAR = False

# Image datasets (require sklearn for MNIST)
try:
    from .image import MNISTBinaryDataset

    HAS_IMAGE = True
except ImportError:
    HAS_IMAGE = False

__all__ = [
    "BenchmarkDataset",
    "get_dataset",
    "SyntheticDataset",
]

if HAS_TABULAR:
    __all__.extend(["IrisBinaryDataset", "SyntheticBlobsDataset"])

if HAS_IMAGE:
    __all__.append("MNISTBinaryDataset")
