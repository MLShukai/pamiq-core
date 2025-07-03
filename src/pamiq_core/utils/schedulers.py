from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from typing import override

from pamiq_core import time

type Callback = Callable[[], None]
type CallbackIterable = Iterable[Callback]


class Scheduler(ABC):
    """Abstract base class for schedulers.

    Schedulers periodically execute registered callbacks based on
    specific conditions defined by subclasses. The base class provides
    functionality for managing callbacks.
    """

    def __init__(self, callbacks: Callback | CallbackIterable | None = None) -> None:
        """Initialize the scheduler.

        Args:
            callbacks: Optional iterable of callback functions to register or callback.
        """
        if callbacks is None:
            callbacks = []
        if isinstance(callbacks, Callable):
            callbacks = [callbacks]

        self._callbacks = list(callbacks)

    def register_callback(self, callback: Callback) -> None:
        """Register a new callback function.

        Args:
            callback: The callback function to register.
        """
        self._callbacks.append(callback)

    def remove_callback(self, callback: Callback) -> None:
        """Remove a registered callback function.

        Args:
            callback: The callback function to remove.

        Raises:
            ValueError: If the callback is not registered.
        """
        self._callbacks.remove(callback)

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the scheduler should execute callbacks now.

        Returns:
            True if callbacks should be executed, False otherwise.
        """
        pass

    @abstractmethod
    def update(self) -> None:
        """Check if callbacks should be executed and run them if so.

        This method should be called regularly to check if execution
        conditions are met.
        """
        if self.is_available():
            for callback in self._callbacks:
                callback()


class TimeIntervalScheduler(Scheduler):
    """Scheduler that executes callbacks at specified time intervals.

    This scheduler triggers callbacks when a specified amount of time
    has elapsed since the last execution.
    """

    @override
    def __init__(
        self, interval: float, callbacks: Callback | CallbackIterable | None = None
    ) -> None:
        """Initialize the time interval scheduler.

        Args:
            interval: Time interval in seconds between executions.
            callbacks: Optional iterable of callback functions to register.

        Raises:
            ValueError: If interval is negative.
        """
        super().__init__(callbacks)
        if interval < 0:
            raise ValueError("Interval must be non-negative")
        self._interval = interval
        self._previous_available_time = time.time()

    @override
    def is_available(self) -> bool:
        """Check if the specified time interval has elapsed.

        Returns:
            True if the time interval has elapsed, False otherwise.
        """
        return time.time() - self._previous_available_time > self._interval

    @override
    def update(self) -> None:
        """Check if the time interval has elapsed and execute callbacks if so.

        Updates the previous execution time after callbacks are
        executed.
        """
        super().update()  # Call parent to execute callbacks if available
        if self.is_available():
            self._previous_available_time = time.time()


class StepIntervalScheduler(Scheduler):
    """Scheduler that executes callbacks after a specified number of steps.

    This scheduler triggers callbacks when a specified number of steps
    have been completed since the last execution.
    """

    @override
    def __init__(
        self, interval: int, callbacks: Callback | CallbackIterable | None = None
    ) -> None:
        """Initialize the step interval scheduler.

        Args:
            interval: Number of steps between executions.
            callbacks: Optional iterable of callback functions to register.

        Raises:
            ValueError: If interval is not positive.
        """
        super().__init__(callbacks)
        if interval <= 0:
            raise ValueError("Interval must be positive")
        self._interval = interval
        self._steps_since_last_call = 0

    @override
    def is_available(self) -> bool:
        """Check if the specified number of steps has been reached.

        Returns:
            True if the step interval has been reached, False otherwise.
        """
        return self._steps_since_last_call >= self._interval

    @override
    def update(self) -> None:
        """Increment step counter, execute callbacks if interval is reached.

        Resets the step counter after callbacks are executed.
        """
        self._steps_since_last_call += 1
        super().update()  # Call parent to execute callbacks if available
        if self.is_available():
            self._steps_since_last_call = 0
