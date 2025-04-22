"""PAMIQ-Core Time Module.

This module provides a custom implementation of time-related functions with time acceleration and pause/resume features.
It wraps the standard Python time module and allows for consistent time control across the system.

Key features:
    - Time acceleration: Adjust the speed of time passage in the system.
    - Pause/Resume: Ability to pause and resume the flow of time in the system.
    - Thread-safe: All operations are protected by locks for use in multi-threaded environments.
    - Compatible API: Provides familiar time functions like sleep, time, perf_counter, and monotonic.
    - Original time functions: Provides access to the original time functions with 'fixed_' prefix.

Usage:
    ```py
    from pamiq_core import time

    # Get current time (affected by time scale and pause)
    current_time = time.time()

    # Get current time (not affected by time scale or pause)
    fixed_current_time = time.fixed_time()

    # Sleep for 1 second (affected by time scale and pause)
    time.sleep(1)

    # Sleep for 1 second (not affected by time scale or pause)
    time.fixed_sleep(1)

    # Set time scale (e.g., 2x speed)
    time.set_time_scale(2.0)

    # Get current time scale
    current_scale = time.get_time_scale()

    # Pause time
    time.pause()

    # Resume time
    time.resume()
    ```

Note: This module is designed for use within the system and may not be suitable for general-purpose time management.
"""

from __future__ import annotations

import time as _original_time
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from threading import RLock
from typing import Concatenate, TypedDict, override

from pamiq_core.state_persistence import PersistentStateMixin, load_pickle, save_pickle


class TimeController(PersistentStateMixin):
    def __init__(self) -> None:
        self._lock = RLock()
        self._anchor_time = _original_time.time()
        self._anchor_perf_counter = _original_time.perf_counter()
        self._anchor_monotonic = _original_time.monotonic()
        self._scaled_anchor_time = _original_time.time()
        self._scaled_anchor_perf_counter = _original_time.perf_counter()
        self._scaled_anchor_monotonic = _original_time.monotonic()

        self._time_scale = 1.0
        self._is_paused = False

    @staticmethod
    def with_lock[**P, T](
        method: Callable[Concatenate[TimeController, P], T],
    ) -> Callable[Concatenate[TimeController, P], T]:
        @wraps(method)
        def _impl(
            self: TimeController, *method_args: P.args, **method_kwargs: P.kwargs
        ) -> T:
            with self._lock:
                return method(self, *method_args, **method_kwargs)

        return _impl

    @with_lock
    def is_paused(self) -> bool:
        """Public interface for checking pause."""
        return self._is_paused

    @with_lock
    def _update_anchor_values(self) -> None:
        self._anchor_time = _original_time.time()
        self._anchor_perf_counter = _original_time.perf_counter()
        self._anchor_monotonic = _original_time.monotonic()

    @with_lock
    def _update_scaled_anchor_values(self) -> None:
        self._scaled_anchor_time = self.time()
        self._scaled_anchor_perf_counter = self.perf_counter()
        self._scaled_anchor_monotonic = self.monotonic()

    @with_lock
    def time(self) -> float:
        """Return the current time in seconds since the epoch.

        This function is affected by the current time scale and pause state.

        Returns:
            float: The current time in seconds since the epoch.
        """
        if self._is_paused:
            return self._scaled_anchor_time

        delta = _original_time.time() - self._anchor_time
        return self._scaled_anchor_time + delta * self._time_scale

    @with_lock
    def perf_counter(self) -> float:
        """Return the value (in fractional seconds) of a performance counter.

        This function is affected by the current time scale and pause state.

        Returns:
            float: The current value of the performance counter.
        """
        if self._is_paused:
            return self._scaled_anchor_perf_counter

        delta = _original_time.perf_counter() - self._anchor_perf_counter
        return self._scaled_anchor_perf_counter + delta * self._time_scale

    @with_lock
    def monotonic(self) -> float:
        """Return the value (in fractional seconds) of a monotonic time
        counter.

        This function is affected by the current time scale and pause state.

        Returns:
            float: The current value of the monotonic time counter.
        """
        if self._is_paused:
            return self._scaled_anchor_monotonic

        delta = _original_time.monotonic() - self._anchor_monotonic
        return self._scaled_anchor_monotonic + delta * self._time_scale

    @with_lock
    def set_time_scale(self, time_scale: float) -> None:
        """Set the time scale for the system.

        Args:
            time_scale (float): The new time scale. Must be greater than 0.

        Raises:
            AssertionError: If time_scale is not greater than 0.
        """
        assert time_scale > 0, "Time scale must be > 0"
        self._update_scaled_anchor_values()
        self._update_anchor_values()
        self._time_scale = time_scale

    @with_lock
    def get_time_scale(self) -> float:
        """Get the current time scale of the system.

        Returns:
            float: The current time scale.
        """
        return self._time_scale

    def sleep(self, secs: float) -> None:
        """Suspend execution for the given number of seconds.

        This function is affected by the current time scale and pause state.

        Args:
            secs (float): The number of seconds to sleep.
        """

        with self._lock:
            if self._is_paused:
                return
            time_scale = self._time_scale
        _original_time.sleep(secs / time_scale)

    @with_lock
    def pause(self) -> None:
        """Pause the flow of time in the system."""
        if not self._is_paused:
            self._update_scaled_anchor_values()
            self._is_paused = True

    @with_lock
    def resume(self) -> None:
        """Resume the flow of time in the system."""
        if self._is_paused:
            self._is_paused = False
            self._update_anchor_values()

    class TimeControllerState(TypedDict):
        """Time controller state for restarting the system."""

        scaled_anchor_time: float
        scaled_anchor_monotonic: float
        scaled_anchor_perf_counter: float

    @with_lock
    def state_dict(self) -> TimeControllerState:
        """Return the time controller state.

        Returns:
            TimeControllerState: State information that can reproduce the current time flow progression of the system.
        """
        self._update_scaled_anchor_values()
        return self.TimeControllerState(
            scaled_anchor_time=self._scaled_anchor_time,
            scaled_anchor_monotonic=self._scaled_anchor_monotonic,
            scaled_anchor_perf_counter=self._scaled_anchor_perf_counter,
        )

    @with_lock
    def load_state_dict(self, state_dict: TimeControllerState) -> None:
        """Loads states.

        When a state is loaded, the system time starts from the time the state was retrieved.

        Args:
            state_dict (TimeControllerState): The dict which contains state values.
        """
        self._scaled_anchor_time = state_dict["scaled_anchor_time"]
        self._scaled_anchor_monotonic = state_dict["scaled_anchor_monotonic"]
        self._scaled_anchor_perf_counter = state_dict["scaled_anchor_perf_counter"]
        self._update_anchor_values()

    @override
    def save_state(self, path: Path) -> None:
        """Save the current TimeController state to disk.

        Args:
            path: Base path where the state will be saved. Will be appended with
                 .pkl extension.
        """
        save_pickle(self.state_dict(), path.with_suffix(".pkl"))

    @override
    def load_state(self, path: Path) -> None:
        """Load TimeController state from disk.

        Args:
            path: Base path where the state was previously saved. Will be appended
                 with .pkl extension.
        """
        self.load_state_dict(load_pickle(path.with_suffix(".pkl")))


# Create a global instance of TimeController
_time_controller = TimeController()


def get_global_time_controller() -> TimeController:
    """Retrieves the global instance of TimeController.

    Returns:
        Global instance of TimeController
    """
    return _time_controller


# Expose the public methods
is_paused = _time_controller.is_paused
sleep = _time_controller.sleep
time = _time_controller.time
perf_counter = _time_controller.perf_counter
monotonic = _time_controller.monotonic
set_time_scale = _time_controller.set_time_scale
get_time_scale = _time_controller.get_time_scale
pause = _time_controller.pause
resume = _time_controller.resume
state_dict = _time_controller.state_dict
load_state_dict = _time_controller.load_state_dict

# Expose the original time functions.
fixed_sleep = _original_time.sleep
fixed_time = _original_time.time
