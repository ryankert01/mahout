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

"""Timer implementations for benchmark measurements.

This module provides timer classes for measuring execution time:
- WallClockTimer: Simple wall-clock timing using time.perf_counter()
- CUDATimer: GPU timing using CUDA events for accurate GPU measurements

Example:
    >>> from benchmark.core.timing import CUDATimer
    >>> with CUDATimer() as timer:
    ...     result = encode_data(batch)
    >>> print(f"GPU time: {timer.elapsed_ms:.3f} ms")
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from functools import wraps
from typing import Any, Callable, Optional, Type, TypeVar

T = TypeVar("T")


class BaseTimer(ABC):
    """Abstract base timer for benchmark measurements.

    Timers can be used as context managers for convenient timing:

        with SomeTimer() as timer:
            do_work()
        print(f"Elapsed: {timer.elapsed_ms} ms")
    """

    @abstractmethod
    def reset(self) -> None:
        """Reset timer state to initial values."""
        pass

    @abstractmethod
    def start(self) -> None:
        """Start the timer."""
        pass

    @abstractmethod
    def stop(self) -> None:
        """Stop the timer and compute elapsed time."""
        pass

    @property
    @abstractmethod
    def elapsed_ms(self) -> float:
        """Return elapsed time in milliseconds."""
        pass

    def __enter__(self) -> BaseTimer:
        """Context manager entry: reset and start the timer."""
        self.reset()
        self.start()
        return self

    def __exit__(self, *args: Any) -> None:
        """Context manager exit: stop the timer."""
        self.stop()


class WallClockTimer(BaseTimer):
    """Simple wall-clock timer using time.perf_counter().

    This timer measures real elapsed time including any blocking operations.
    For GPU-bound operations, prefer CUDATimer for more accurate measurements.

    Example:
        >>> timer = WallClockTimer()
        >>> timer.start()
        >>> do_work()
        >>> timer.stop()
        >>> print(f"Elapsed: {timer.elapsed_ms:.3f} ms")
    """

    def __init__(self) -> None:
        """Initialize the wall clock timer."""
        self._start: Optional[float] = None
        self._elapsed: float = 0.0

    def reset(self) -> None:
        """Reset timer to initial state."""
        self._start = None
        self._elapsed = 0.0

    def start(self) -> None:
        """Start timing."""
        self._start = time.perf_counter()

    def stop(self) -> None:
        """Stop timing and compute elapsed milliseconds.

        Raises:
            RuntimeError: If timer was not started.
        """
        if self._start is None:
            raise RuntimeError("Timer not started")
        self._elapsed = (time.perf_counter() - self._start) * 1000.0

    @property
    def elapsed_ms(self) -> float:
        """Return elapsed time in milliseconds."""
        return self._elapsed


class CUDATimer(BaseTimer):
    """GPU timer using CUDA events for accurate GPU timing.

    CUDA events measure time on the GPU timeline, eliminating CPU/GPU
    synchronization overhead from measurements. This provides more accurate
    timing for GPU-bound operations than wall-clock timing.

    The timer automatically synchronizes before starting and after stopping
    to ensure accurate measurement boundaries.

    Example:
        >>> timer = CUDATimer(device_id=0)
        >>> timer.start()
        >>> result = gpu_operation()
        >>> timer.stop()
        >>> print(f"GPU time: {timer.elapsed_ms:.3f} ms")

    Note:
        Requires PyTorch with CUDA support.
    """

    def __init__(self, device_id: int = 0) -> None:
        """Initialize the CUDA timer.

        Args:
            device_id: CUDA device index to use for timing events.

        Raises:
            RuntimeError: If CUDA is not available.
        """
        self.device_id = device_id
        self._start_event: Any = None
        self._stop_event: Any = None
        self._elapsed: float = 0.0
        self._initialize_events()

    def _initialize_events(self) -> None:
        """Create CUDA events for timing."""
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")

        with torch.cuda.device(self.device_id):
            self._start_event = torch.cuda.Event(enable_timing=True)
            self._stop_event = torch.cuda.Event(enable_timing=True)

    def reset(self) -> None:
        """Reset elapsed time to zero."""
        self._elapsed = 0.0

    def start(self) -> None:
        """Start GPU timing.

        Synchronizes the device to drain the command queue before recording
        the start event.
        """
        import torch

        torch.cuda.synchronize(self.device_id)
        self._start_event.record()

    def stop(self) -> None:
        """Stop GPU timing and compute elapsed time.

        Records the stop event and synchronizes to wait for completion,
        then computes the elapsed time between events.
        """
        import torch

        self._stop_event.record()
        torch.cuda.synchronize(self.device_id)
        self._elapsed = self._start_event.elapsed_time(self._stop_event)

    @property
    def elapsed_ms(self) -> float:
        """Return elapsed GPU time in milliseconds."""
        return self._elapsed


def timed(
    timer_class: Optional[Type[BaseTimer]] = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator that times a function and prints the result.

    Args:
        timer_class: Timer class to use. If None, auto-selects CUDATimer
            if CUDA is available, otherwise WallClockTimer.

    Returns:
        Decorated function that prints timing after each call.

    Example:
        >>> @timed()
        ... def my_benchmark():
        ...     return expensive_computation()
        >>> result = my_benchmark()
        my_benchmark: 42.123 ms
    """

    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            from .cuda import is_cuda_available

            timer_cls = timer_class
            if timer_cls is None:
                timer_cls = CUDATimer if is_cuda_available() else WallClockTimer

            with timer_cls() as timer:
                result = fn(*args, **kwargs)
            print(f"{fn.__name__}: {timer.elapsed_ms:.3f} ms")
            return result

        return wrapper

    return decorator
