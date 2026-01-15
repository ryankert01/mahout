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

"""Tests for benchmark.core.timing module."""

import time

import pytest

from benchmark.core.timing import CUDATimer, WallClockTimer


class TestWallClockTimer:
    """Tests for WallClockTimer."""

    def test_basic_timing(self):
        """Test basic timing functionality."""
        timer = WallClockTimer()

        timer.start()
        time.sleep(0.01)  # 10ms
        timer.stop()

        # Allow variance for system scheduling
        assert 5 < timer.elapsed_ms < 50

    def test_context_manager(self):
        """Test timer as context manager."""
        with WallClockTimer() as timer:
            time.sleep(0.01)

        assert timer.elapsed_ms > 0

    def test_reset(self):
        """Test timer reset."""
        timer = WallClockTimer()
        timer.start()
        time.sleep(0.01)
        timer.stop()

        old_elapsed = timer.elapsed_ms
        assert old_elapsed > 0

        timer.reset()
        assert timer.elapsed_ms == 0.0

    def test_stop_without_start_raises(self):
        """Test that stop without start raises RuntimeError."""
        timer = WallClockTimer()

        with pytest.raises(RuntimeError, match="not started"):
            timer.stop()

    def test_multiple_start_stop_cycles(self):
        """Test multiple timing cycles."""
        timer = WallClockTimer()

        # First cycle
        timer.start()
        time.sleep(0.005)
        timer.stop()
        first = timer.elapsed_ms

        # Second cycle (reset and start fresh)
        timer.reset()
        timer.start()
        time.sleep(0.005)
        timer.stop()
        second = timer.elapsed_ms

        assert first > 0
        assert second > 0


@pytest.mark.gpu
class TestCUDATimer:
    """Tests for CUDATimer (requires CUDA)."""

    def test_basic_timing(self):
        """Test basic GPU timing."""
        import torch

        if not torch.cuda.is_available():
            pytest.skip("CUDA required")

        timer = CUDATimer()

        timer.start()
        # Do some GPU work
        x = torch.randn(1000, 1000, device="cuda")
        _ = x @ x.T
        timer.stop()

        assert timer.elapsed_ms > 0

    def test_context_manager(self):
        """Test CUDA timer as context manager."""
        import torch

        if not torch.cuda.is_available():
            pytest.skip("CUDA required")

        with CUDATimer() as timer:
            x = torch.randn(1000, 1000, device="cuda")
            _ = x @ x.T

        assert timer.elapsed_ms > 0

    def test_reset(self):
        """Test CUDA timer reset."""
        import torch

        if not torch.cuda.is_available():
            pytest.skip("CUDA required")

        timer = CUDATimer()

        timer.start()
        x = torch.randn(100, 100, device="cuda")
        _ = x @ x.T
        timer.stop()

        assert timer.elapsed_ms > 0

        timer.reset()
        assert timer.elapsed_ms == 0.0

    def test_no_cuda_raises(self):
        """Test that CUDATimer raises when CUDA unavailable."""
        import torch

        if torch.cuda.is_available():
            pytest.skip("Test requires CUDA to be unavailable")

        with pytest.raises(RuntimeError, match="CUDA not available"):
            CUDATimer()

    def test_device_id(self):
        """Test specifying device_id."""
        import torch

        if not torch.cuda.is_available():
            pytest.skip("CUDA required")

        # Should work with default device 0
        timer = CUDATimer(device_id=0)

        with timer:
            x = torch.randn(100, 100, device="cuda:0")
            _ = x.sum()

        assert timer.elapsed_ms >= 0
