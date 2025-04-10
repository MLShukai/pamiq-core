import math
from typing import override

from pamiq_core import time

from .base import Thread
from .thread_control import (
    ReadOnlyController,
    ReadOnlyThreadStatus,
    ThreadController,
    ThreadStatusesMonitor,
)
from .thread_types import ThreadTypes


class ControlThread(Thread):
    """Control thread for managing the system's threading lifecycle."""

    THREAD_TYPE = ThreadTypes.CONTROL

    def __init__(
        self,
        timeout_for_all_threads_pause: float = 60.0,
        max_attempts_to_pause_all_threads: int = 3,
        max_uptime: float = math.inf,
    ) -> None:
        """Initialize the control thread.

        Args:
            timeout_for_all_threads_pause: Maximum time in seconds to wait for
                all threads to pause before timing out a pause attempt.
            max_attempts_to_pause_all_threads: Maximum number of retry attempts
                when pausing threads fails.
            max_uptime: Maximum time in seconds the system is allowed to run before
                automatic shutdown. Default is infinity (no time limit).
        """
        super().__init__()

        self._timeout_for_all_threads_pause = timeout_for_all_threads_pause
        self._max_attempts_to_pause_all_threads = max_attempts_to_pause_all_threads
        self._max_uptime = max_uptime
        self._system_start_time = -math.inf

        self._controller = ThreadController()
        self._running = True

    @property
    def controller(self) -> ReadOnlyController:
        """Get a read-only interface to the thread controller.

        Returns:
            A read-only view of the internal thread controller that can be
            safely shared with other threads.
        """
        return self._controller.read_only

    def attach_thread_statuses(
        self, thread_statuses: dict[ThreadTypes, ReadOnlyThreadStatus]
    ) -> None:
        """Attach thread status monitors for the threads being controlled.

        This method must be called before using pause/resume functionality
        to enable monitoring of all managed threads.

        Args:
            thread_statuses: Dictionary mapping thread types to their
                read-only status interfaces.
        """
        self._thread_statuses_monitor = ThreadStatusesMonitor(thread_statuses)

    def try_pause(self) -> bool:
        """Attempt to pause all threads in the system.

        Makes multiple attempts to pause all threads within the configured
        timeout period. If successful, also pauses the system time.

        Returns:
            True if all threads were successfully paused, False otherwise.
        """
        if self._controller.is_pause():
            self._logger.info("System has already been paused.")
            return True

        self._logger.info("Trying to pause...")
        for i in range(self._max_attempts_to_pause_all_threads):
            self._controller.pause()
            if self._thread_statuses_monitor.wait_for_all_threads_pause(
                self._timeout_for_all_threads_pause
            ):
                self._logger.info("Success to pause the all background threads.")
                self.on_paused()
                return True
            else:
                self._logger.warning(
                    f"Failed to pause the background threads in timeout {self._timeout_for_all_threads_pause} seconds."
                )
                self._logger.warning(
                    f"Attempting retry {i+1} / {self._max_attempts_to_pause_all_threads} ..."
                )
                self._controller.resume()

        self._logger.error("Failed to pause... ")
        return False

    def resume(self) -> None:
        """Resume all paused threads in the system.

        Invokes the on_resumed event handler first, then signals all
        threads to resume execution.
        """
        self._logger.info("Resuming...")
        self.on_resumed()
        self._controller.resume()

    def shutdown(self) -> None:
        """Shutdown the control thread and signal all other threads to stop.

        Sets the controller to shutdown state and marks this thread as
        no longer running.
        """
        self._logger.info("Shutting down...")
        self._controller.shutdown()
        self._running = False

    @property
    def is_max_uptime_reached(self) -> bool:
        """Check if the system has exceeded its maximum allowed uptime.

        Returns:
            True if the system has been running longer than the configured max_uptime,
            False otherwise.
        """
        return time.time() - self._system_start_time > self._max_uptime

    @override
    def is_running(self) -> bool:
        """Check if the control thread should continue running.

        Returns:
            True if the thread should continue running, False otherwise.
        """
        return self._running

    @override
    def on_start(self) -> None:
        """Initialize the control thread's start time.

        Records the system start time to enable max uptime tracking.
        """
        super().on_start()
        self._system_start_time = time.time()
        self._logger.info(
            f"Maxmum uptime is set to {self._max_uptime:.1f} [secs]. "
            f"(actually {self._max_uptime / time.get_time_scale():.1f} [secs] "
            f"in time scale x{time.get_time_scale():.1f})"
        )

    @override
    def on_tick(self) -> None:
        """Execute a single iteration of the control thread's main loop.

        Checks if any exceptions have been raised in other threads and
        initiates shutdown if needed.
        """
        super().on_tick()
        if self._thread_statuses_monitor.check_exception_raised():
            self._logger.error(
                "An exception occurred. The system will terminate immediately."
            )
            self.shutdown()

        if self.is_max_uptime_reached:
            self._logger.info("Max uptime reached.")
            self.shutdown()

    @override
    def on_finally(self) -> None:
        """Perform cleanup operations when the thread is about to exit.

        Ensures the system is properly shut down even if the thread
        exits unexpectedly.
        """
        super().on_finally()
        self.shutdown()

    @override
    def on_paused(self) -> None:
        """Handle system-wide pause event.

        Pauses the system time to ensure consistent behavior across all
        time-dependent components.
        """
        super().on_paused()
        time.pause()

    @override
    def on_resumed(self) -> None:
        """Handle system-wide resume event.

        Resumes the system time to restore normal operation of all time-
        dependent components.
        """
        super().on_resumed()
        time.resume()
