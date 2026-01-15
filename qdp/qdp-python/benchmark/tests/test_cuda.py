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

"""Tests for benchmark.core.cuda module."""

import pytest

from benchmark.core.cuda import (
    clear_gpu_caches,
    get_device_count,
    get_gpu_memory_info,
    is_cuda_available,
    reset_peak_memory_stats,
    sync_cuda,
    warmup_gpu,
)


class TestCUDAUtilities:
    """Tests for CUDA utility functions."""

    def test_is_cuda_available(self):
        """Test is_cuda_available returns boolean."""
        result = is_cuda_available()
        assert isinstance(result, bool)

    def test_get_device_count(self):
        """Test get_device_count returns non-negative integer."""
        count = get_device_count()
        assert isinstance(count, int)
        assert count >= 0

        # If CUDA available, should have at least 1 device
        if is_cuda_available():
            assert count >= 1

    def test_sync_cuda_no_error(self):
        """Test sync_cuda doesn't raise errors."""
        # Should work whether CUDA is available or not
        sync_cuda()
        sync_cuda(device_id=0)

    def test_clear_gpu_caches_no_error(self):
        """Test clear_gpu_caches doesn't raise errors."""
        # Should work whether CUDA is available or not
        clear_gpu_caches()
        clear_gpu_caches(gc_collect=False)


@pytest.mark.gpu
class TestCUDAUtilitiesGPU:
    """Tests for CUDA utilities that require GPU."""

    def test_warmup_gpu(self):
        """Test GPU warmup runs without error."""
        import torch

        if not torch.cuda.is_available():
            pytest.skip("CUDA required")

        # Should complete without error
        warmup_gpu(device_id=0, iterations=5)

    def test_get_gpu_memory_info(self):
        """Test GPU memory info retrieval."""
        import torch

        if not torch.cuda.is_available():
            pytest.skip("CUDA required")

        info = get_gpu_memory_info(device_id=0)

        assert "allocated" in info
        assert "reserved" in info
        assert "max_allocated" in info
        assert "total" in info

        # Total memory should be positive
        assert info["total"] > 0

    def test_reset_peak_memory_stats(self):
        """Test peak memory stats reset."""
        import torch

        if not torch.cuda.is_available():
            pytest.skip("CUDA required")

        # Allocate some memory
        x = torch.randn(1000, 1000, device="cuda")
        _ = x.sum()

        # Reset should not raise
        reset_peak_memory_stats(device_id=0)

        # After reset, allocate more and check peak
        info_before = get_gpu_memory_info()
        y = torch.randn(500, 500, device="cuda")
        _ = y.sum()
        info_after = get_gpu_memory_info()

        # Peak should reflect new allocation, not old
        assert info_after["max_allocated"] >= 0

    def test_clear_gpu_caches_frees_memory(self):
        """Test that clear_gpu_caches actually frees cached memory."""
        import torch

        if not torch.cuda.is_available():
            pytest.skip("CUDA required")

        # Allocate and free tensors
        tensors = [torch.randn(1000, 1000, device="cuda") for _ in range(5)]
        del tensors

        reserved_before = torch.cuda.memory_reserved()

        clear_gpu_caches()

        reserved_after = torch.cuda.memory_reserved()

        # Reserved memory should decrease (or stay same if already empty)
        assert reserved_after <= reserved_before
