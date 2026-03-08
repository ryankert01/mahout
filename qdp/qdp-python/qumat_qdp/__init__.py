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
QDP (Quantum Data Processing) Python API.

Public API: QdpEngine, QdpBenchmark, ThroughputResult, LatencyResult,
QuantumDataLoader.

The package prefers the PyTorch-native engine (no Rust/CUDA dependency).
If the Rust extension (_qdp) is available, it is also exposed for advanced use.

Usage:
    from qumat_qdp import QdpEngine
    from qumat_qdp import QdpBenchmark, ThroughputResult, LatencyResult
    from qumat_qdp import QuantumDataLoader
"""

from __future__ import annotations

from qumat_qdp.api import (
    LatencyResult,
    QdpBenchmark,
    ThroughputResult,
)
from qumat_qdp.engine import QdpEngine
from qumat_qdp.loader import QuantumDataLoader

# Try to import Rust extension for backward compatibility.
# The Rust-backed types are available as RustQdpEngine / QuantumTensor if needed.
try:
    import _qdp

    RustQdpEngine = getattr(_qdp, "QdpEngine", None)
    QuantumTensor = getattr(_qdp, "QuantumTensor", None)
    run_throughput_pipeline_py = getattr(_qdp, "run_throughput_pipeline_py", None)
except ImportError:
    _qdp = None  # type: ignore[assignment]
    RustQdpEngine = None
    QuantumTensor = None
    run_throughput_pipeline_py = None

__all__ = [
    "LatencyResult",
    "QdpBenchmark",
    "QdpEngine",
    "QuantumDataLoader",
    "QuantumTensor",
    "RustQdpEngine",
    "ThroughputResult",
    "run_throughput_pipeline_py",
]
