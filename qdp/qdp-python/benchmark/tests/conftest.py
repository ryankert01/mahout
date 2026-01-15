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

"""Shared test fixtures for benchmark tests."""

import numpy as np
import pytest


@pytest.fixture
def sample_timings():
    """Sample timing data for statistics tests."""
    return [10.5, 11.2, 10.8, 11.0, 10.9, 11.1, 10.7, 11.3, 10.6, 11.0]


@pytest.fixture
def sample_timings_with_outliers():
    """Timing data with obvious outliers."""
    return [10.5, 11.2, 10.8, 50.0, 10.9, 11.1, 10.7, 0.1, 10.6, 11.0]


@pytest.fixture
def sample_batch():
    """Sample 2D array for normalization tests."""
    np.random.seed(42)
    return np.random.randn(10, 16).astype(np.float64)


def is_cuda_available():
    """Check if CUDA is available for GPU tests."""
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False
