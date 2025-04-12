import math
from typing import override

from pamiq_core import time
from pamiq_core.console import ControlCommands, SystemStatusProvider, WebApiHandler
from pamiq_core.state_persistence import StateStore
from pamiq_core.utils.schedulers import TimeIntervalScheduler

from ..thread_control import (
    ReadOnlyController,
    ReadOnlyThreadStatus,
    ThreadController,
    ThreadStatusesMonitor,
)
from ..thread_types import ThreadTypes
from .base import Thread


class ControlThread(Thread):
    """Control thread for managing the system's threading lifecycle."""

    THREAD_TYPE = ThreadTypes.CONTROL

    def __init__(
        self,
        state_store: StateStore,
        save_state_interval: float = math.inf,
        timeout_for_all_threads_pause: float = 60.0,
        max_attempts_to_pause_all_threads: int = 3,
        max_uptime: float = math.inf,
        web_api_address: tuple[str, int] = ("localhost", 8391),
        web_api_command_queue_size: int = 1,
    ) -> None:
        """Initialize the control thread.

        Args:
            state_store: Store for saving and loading system state.
            save_state_interval: Interval in seconds between automatic state saves.
            timeout_for_all_threads_pause: Maximum time in seconds to wait for
                all threads to pause before timing out a pause attempt.
            max_attempts_to_pause_all_threads: Maximum number of retry attempts
                when pausing threads fails.
            max_uptime: Maximum time in seconds the system is allowed to run before
                automatic shutdown. Default is infinity (no time limit).
            web_api_address: Tuple of (host, port) specifying where the Web API
                should listen for commands.
            web_api_command_queue_size: Maximum number of commands that can be
                queued from the Web API.
        """
        super().__init__()

        self._state_store = state_store
        self._save_state_scheduler = TimeIntervalScheduler(
            save_state_interval, self.save_state
        )

        self._timeout_for_all_threads_pause = timeout_for_all_threads_pause
        self._max_attempts_to_pause_all_threads = max_attempts_to_pause_all_threads
        self._max_uptime = max_uptime
        self._system_start_time = -math.inf

        self._web_api_address = web_api_address
        self._web_api_command_queue_size = web_api_command_queue_size

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

    def save_state(self) -> None:
        """Save the current state of the system.

        This method temporarily pauses all threads if they are not
        already paused, saves the system state using the state store,
        and then resumes threads if they were not paused initially.

        If the method fails to pause the threads, the state saving
        operation will be aborted with a log message.
        """

        already_paused = self._controller.is_pause()
        if not self.try_pause():
            self._logger.info("Failed to pause. Aborting saving a state.")
            return
        saved_path = self._state_store.save_state()
        self._logger.info(f"Saved a state to '{saved_path}'")
        if not already_paused:
            self.resume()

    @property
    def is_max_uptime_reached(self) -> bool:
        """Check if the system has exceeded its maximum allowed uptime.

        Returns:
            True if the system has been running longer than the configured max_uptime,
            False otherwise.
        """
        return time.time() - self._system_start_time > self._max_uptime

    def process_received_web_api_commands(self) -> None:
        """Process any pending commands received from the Web API.

        Retrieves and processes all available commands from the Web API
        handler.
        """
        while self._web_api_handler.has_commands():
            match self._web_api_handler.receive_command():
                case ControlCommands.PAUSE:
                    self.try_pause()
                case ControlCommands.RESUME:
                    self.resume()
                case ControlCommands.SHUTDOWN:
                    self.shutdown()
                    return
                case ControlCommands.SAVE_STATE:
                    self.save_state()

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

        Records the system start time to enable max uptime tracking and
        initializes the web api handler.
        """
        super().on_start()
        self._system_start_time = time.time()
        self._logger.info(
            f"Maxmum uptime is set to {self._max_uptime:.1f} [secs]. "
            f"(actually {self._max_uptime / time.get_time_scale():.1f} [secs] "
            f"in time scale x{time.get_time_scale():.1f})"
        )

        self._web_api_handler = WebApiHandler(
            system_status=SystemStatusProvider(
                self.controller, self._thread_statuses_monitor
            ),
            host=self._web_api_address[0],
            port=self._web_api_address[1],
            max_queue_size=self._web_api_command_queue_size,
        )
        self._web_api_handler.run_in_background()

    @override
    def on_tick(self) -> None:
        """Execute a single iteration of the control thread's main loop.

        Checks if any exceptions have been raised in other threads and
        initiates shutdown if needed.
        """
        super().on_tick()
        self._save_state_scheduler.update()

        self.process_received_web_api_commands()

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
