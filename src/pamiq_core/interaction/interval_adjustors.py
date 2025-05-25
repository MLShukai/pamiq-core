import math
from abc import ABC, abstractmethod
from typing import override

from pamiq_core import time


class IntervalAdjustor(ABC):
    """Adjusts the loop to run at a fixed interval.

    Initial reset time is set to negative infinity for mathematical
    reasons.
    """

    _last_reset_time: float = -math.inf

    def __init__(self, interval: float, offset: float = 0.0) -> None:
        """Constructs the IntervalAdjustor.

        Args:
            interval: The desired time between each invocation in seconds.
            offset: The initial time offset to subtract from the interval on the first loop iteration,
                allowing adjustment of the actual interval for each computer using this parameter.
        """
        self._interval = interval
        self._offset = offset
        self._time_to_wait = interval - offset

    def reset(self) -> float:
        """Resets the start time of this adjustor to the current time.

        Returns:
            float: The start time after resetting the timer, as a high precision time counter.
        """
        self._last_reset_time = time.perf_counter()
        return self._last_reset_time

    @abstractmethod
    def adjust_impl(self) -> None:
        """The actual implementation of the interval adjustor, which is wrapped
        by the public method `adjust`."""
        pass

    def adjust(self) -> float:
        """Waits until the interval has elapsed since the last `adjust` or
        `reset` call.

        Returns:
            float: The elapsed time since the last call, ensuring the loop runs at the specified interval.
        """
        self.adjust_impl()
        delta_time = time.perf_counter() - self._last_reset_time
        self.reset()
        return delta_time


class SleepIntervalAdjustor(IntervalAdjustor):
    """Adjusts the interval using `time.sleep` to pause execution until the
    next interval begins."""

    @override
    def adjust_impl(self) -> None:
        if (
            remaining_time := (self._last_reset_time + self._time_to_wait)
            - time.perf_counter()
        ) > 0:
            time.sleep(remaining_time)
