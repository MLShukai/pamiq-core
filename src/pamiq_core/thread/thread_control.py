from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor

from pamiq_core.thread.thread_types import ThreadTypes
from pamiq_core.utils.reflection import get_class_module_path

type OnPausedCallback = Callable[[], None]
type OnResumedCallback = Callable[[], None]


class ThreadEventMixin:
    """A mixin class to provide event handling methods for a thread."""

    def on_paused(self) -> None:
        """The method to be called when the thread is paused."""
        pass

    def on_resumed(self) -> None:
        """The method to be called when the thread is resumed."""
        pass


class ThreadController:
    """The controller class for sending commands from the control thread to
    other threads.

    NOTE: **Only one thread can control this object.**
    """

    @property
    def read_only(self) -> ReadOnlyController:
        """Get a read-only view of this controller.

        Returns:
            A read-only interface to this controller.
        """
        return ReadOnlyController(self)

    def __init__(self) -> None:
        self._shutdown_event = threading.Event()
        self._resume_event = threading.Event()
        self.resume()
        self.activate()

    def resume(self) -> None:
        """Resume the thread, by setting the resume event.

        Raises:
            RuntimeError: If the thread is shutdown.
        """
        if self.is_shutdown():
            raise RuntimeError("ThreadController must be activated before resume().")
        self._resume_event.set()

    def pause(self) -> None:
        """Pause the thread, by clearing the resume event.

        Raises:
            RuntimeError: If the thread is shutdown.
        """
        if self.is_shutdown():
            raise RuntimeError("ThreadController must be activated before pause().")
        self._resume_event.clear()

    def is_resume(self) -> bool:
        """Returns whether the thread is resumed."""
        return self._resume_event.is_set()

    def is_pause(self) -> bool:
        """Returns whether the thread is paused."""
        return not self.is_resume()

    def shutdown(self) -> None:
        """Shutdown the thread, by setting the shutdown event."""

        if self.is_shutdown():
            return

        # `resume()` must be called before `_shutdown_event.set()`
        # to quickly unblock any threads waiting in `wait_for_resume()`.
        self.resume()
        self._shutdown_event.set()

    def activate(self) -> None:
        """Activate the thread, by clearing the shutdown event."""
        self._shutdown_event.clear()

    def is_shutdown(self) -> bool:
        """Returns whether the thread is shutdown."""
        return self._shutdown_event.is_set()

    def is_active(self) -> bool:
        """Returns whether the thread is active."""
        return not self.is_shutdown()

    def wait_for_resume(self, timeout: float) -> bool:
        """Wait for the resume event to be set.

        Args:
            timeout: The maximum time (second) to wait for the resume event to be set.

        Returns:
            bool: True if the thread is already resumed or the resume event is set within the timeout, False otherwise.
        """
        return self._resume_event.wait(timeout)


class ReadOnlyController:
    """A read-only interface to the ThreadController class.

    Args:
        controller: The ThreadController object to be read.
    """

    def __init__(self, controller: ThreadController) -> None:
        self.is_resume = controller.is_resume
        self.is_pause = controller.is_pause
        self.is_shutdown = controller.is_shutdown
        self.is_active = controller.is_active
        self.wait_for_resume = controller.wait_for_resume


class ControllerCommandHandler:
    """A class, handles commands for thread management, facilitating
    communication and control between the control thread and other threads."""

    def __init__(
        self,
        controller: ReadOnlyController,
        on_paused_callback: OnPausedCallback = lambda: None,
        on_resumed_callback: OnResumedCallback = lambda: None,
    ) -> None:
        """Initialize the ControllerCommandHandler object.

        Args:
            controller: The ReadOnlyController object to be read.
            on_paused_callback: The callback function to be called when the thread is paused.
            on_resumed_callback: The callback function to be called when the thread is resumed.
        """
        self._controller = controller
        self.on_paused = on_paused_callback
        self.on_resumed = on_resumed_callback

    def stop_if_pause(self) -> None:
        """Wait until the thread is resumed, or return immediately if the
        thread is resumed. on_paused_callback and on_resumed_callback will be
        called when the thread is paused and resumed, respectively.

        Behavior of this function:
        * If the thread is resume: the function will return immediately.
        * If the thread is paused: the function will block until the thread is resumed or shutdown.
        """
        paused = False
        if self._controller.is_pause():
            # In this implementation, `self._on_pause()` is invoked almost immediately when a pause occurs.
            # Because the `ControllerCommandHandler` primarily runs `manage_loop()`,
            # the `stop_if_pause()` method is frequently executed.
            self.on_paused()
            paused = True

        while not self._controller.wait_for_resume(1.0):
            pass

        if paused:
            self.on_resumed()

    def manage_loop(self) -> bool:
        """Manages the infinite loop: blocking during thread is paused, and returning thread's activity flag.

        This method facilitates the implementation of a pause-resume mechanism within a running loop.
        Use this function in a while loop to manage thread execution based on pause and resume commands.

        Example:
            ```python
            while controller_command_handler.manage_loop():
                ... # your process
            ```

        Returns:
            bool: True if the thread is active, False otherwise.
        """
        self.stop_if_pause()
        return self._controller.is_active()


class ThreadStatus:
    """A class to manage the status of a thread.

    The readonly interface is provided by the ReadOnlyThreadStatus class
    and mainly used for monitoring the status of a thread.
    """

    @property
    def read_only(self) -> ReadOnlyThreadStatus:
        """Get a read-only view of this thread status.

        Returns:
            A read-only interface to this thread status.
        """

        return ReadOnlyThreadStatus(self)

    def __init__(self) -> None:
        self._paused_event = threading.Event()
        self._exception_event = threading.Event()

    def pause(self) -> None:
        """Marks the thread as paused.

        This function is invoked when the thread enters a paused state.
        """
        self._paused_event.set()

    def resume(self) -> None:
        """Marks the thread as resumed.

        This function is invoked when the thread enters a resumed state.
        """
        self._paused_event.clear()

    def is_pause(self) -> bool:
        """Returns whether the thread is paused.

        Returns:
            bool: True if the thread is paused, False otherwise.
        """
        return self._paused_event.is_set()

    def is_resume(self) -> bool:
        """Returns whether the thread is resumed.

        Returns:
            bool: True if the thread is resumed, False otherwise.
        """
        return not self.is_pause()

    def exception_raised(self) -> None:
        """Marks the thread as having an exception.

        This function is invoked when the thread encounters an
        exception.
        """
        self._exception_event.set()

    def is_exception_raised(self) -> bool:
        """Returns whether the thread has an exception.

        Returns:
            bool: True if the thread has an "exception raised flag", False otherwise.
        """
        return self._exception_event.is_set()

    def wait_for_pause(self, timeout: float) -> bool:
        """Wait for the thread to be paused.

        Args:
            timeout: The maximum time (second) to wait for the thread to be paused.

        Returns:
            bool: True if the thread is already paused or the thread is paused within the timeout, False otherwise.
        """
        return self._paused_event.wait(timeout)


class ReadOnlyThreadStatus:
    """A read-only interface to the ThreadStatus class."""

    def __init__(self, status: ThreadStatus) -> None:
        self.is_pause = status.is_pause
        self.is_resume = status.is_resume
        self.wait_for_pause = status.wait_for_pause
        self.is_exception_raised = status.is_exception_raised


class ThreadStatusesMonitor:
    """A class to monitor the statuses of multiple threads."""

    def __init__(self, statuses: dict[ThreadTypes, ReadOnlyThreadStatus]) -> None:
        """Initialize the ThreadStatusesMonitor object.

        Args:
            statuses: A dictionary of ReadOnlyThreadStatus objects.
        """
        self._statuses = statuses
        self._logger = logging.getLogger(get_class_module_path(self.__class__))

    def wait_for_all_threads_pause(self, timeout: float) -> bool:
        """Wait for all threads to be paused.

        Args:
            timeout: The maximum time (second) to wait for all threads to be paused.

        Returns:
            bool: True if all threads are already paused or all threads are paused within the timeout, False otherwise.
        """
        if len(self._statuses) == 0:
            # Need to return first to avoid ValueError in ThreadPoolExecutor
            # (max_workers must be greater than 0)
            return True

        tasks: dict[ThreadTypes, Future[bool]] = {}
        with ThreadPoolExecutor(max_workers=len(self._statuses)) as executor:
            for thread_type, stat in self._statuses.items():
                tasks[thread_type] = executor.submit(stat.wait_for_pause, timeout)

        success = True
        for thread_type, tsk in tasks.items():
            if not (result := tsk.result()):
                self._logger.error(
                    f"Timeout waiting for '{thread_type.thread_name}' thread to pause after {timeout} seconds."
                )
            success &= result
        return success

    def check_exception_raised(self) -> bool:
        """Check if any thread has an exception.

        Returns:
            bool: True if at least one thread has an exception, False otherwise.
        """
        flag = False
        for thread_type, stat in self._statuses.items():
            if stat.is_exception_raised():
                self._logger.error(
                    f"An exception has occurred in the '{thread_type.thread_name}' thread."
                )
                flag = True
        return flag

    def check_all_threads_paused(self) -> bool:
        """Check if all threads are paused.

        Returns:
            bool: True if all threads are paused, False otherwise.
        """
        return all(status.is_pause() for status in self._statuses.values())

    def check_any_threads_paused(self) -> bool:
        """Check if any thread is paused.

        Returns:
            bool: True if at least one thread is paused, False otherwise.
        """
        return any(status.is_pause() for status in self._statuses.values())
