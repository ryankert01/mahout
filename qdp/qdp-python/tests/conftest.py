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

"""Pytest configuration for Mahout QDP Python tests."""

import pytest


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "gpu: tests that require GPU/CUDA (deselect with '-m \"not gpu\"')"
    )


def pytest_collection_modifyitems(config, items):
    """Automatically skip GPU tests when CUDA is not available."""
    # Check if CUDA/GPU is available by attempting to import and instantiate
    cuda_available = False
    try:
        import mahout_qdp
        # Try to create a QdpEngine instance to check if CUDA runtime is available
        try:
            _ = mahout_qdp.QdpEngine(device_id=0)
            cuda_available = True
        except BaseException:
            # QdpEngine initialization failed, CUDA not available
            # Using BaseException to catch PanicException as well
            cuda_available = False
    except ImportError:
        # Module import failed, likely due to missing CUDA
        cuda_available = False

    if not cuda_available:
        skip_gpu = pytest.mark.skip(reason="CUDA not available")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)
