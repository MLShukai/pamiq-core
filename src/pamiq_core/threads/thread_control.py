import threading


class ThreadController:
    """The controller class for sending commands from the control thread to
    other threads.

    NOTE: **Only one thread can control this object.**
    """

    def __init__(self):
        self._shutdown_event = threading.Event()
        self._resume_event = threading.Event()
        self.resume()
        self.activate()

    def resume(self):
        """Resume the thread, by setting the resume event.

        Raises:
            RuntimeError: If the thread is shutdown.
        """
        if self.is_shutdown():
            raise RuntimeError("ThreadController must be activated before resume().")
        self._resume_event.set()

    def pause(self):
        """Pause the thread, by clearing the resume event.

        Raises:
            RuntimeError: If the thread is shutdown.
        """
        if self.is_shutdown():
            raise RuntimeError("ThreadController must be activated before pause().")
        self._resume_event.clear()

    def is_resume(self):
        """Returns whether the thread is resumed."""
        return self._resume_event.is_set()

    def is_pause(self):
        """Returns whether the thread is paused."""
        return not self.is_resume()

    def shutdown(self):
        """Shutdown the thread, by setting the shutdown event."""

        if self.is_shutdown():
            return

        # `resume()` must be called before `_shutdown_event.set()`
        # to quickly unblock any threads waiting in `wait_for_resume()`.
        self.resume()
        self._shutdown_event.set()

    def activate(self):
        """Activate the thread, by clearing the shutdown event."""
        self._shutdown_event.clear()

    def is_shutdown(self):
        """Returns whether the thread is shutdown."""
        return self._shutdown_event.is_set()

    def is_active(self):
        """Returns whether the thread is active."""
        return not self.is_shutdown()

    def wait_for_resume(self, timeout: float):
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

    def __init__(self, controller: ThreadController):
        self.is_resume = controller.is_resume
        self.is_pause = controller.is_pause
        self.is_shutdown = controller.is_shutdown
        self.is_active = controller.is_active
        self.wait_for_resume = controller.wait_for_resume


class ControllerCommandHandler:
    """A class, handles commands for thread management, facilitating
    communication and control between the control thread and other threads.

    Args:
        controller: The ReadOnlyController object to be read.
    """

    def __init__(self, controller: ReadOnlyController):
        self._controller = controller

    def stop_if_pause(self) -> None:
        """This function is used to stop the execution of the current thread if
        the thread is paused.

        Behavior of this function:
        * If the thread is resume: the function will return immediately.
        * If the thread is paused: the function will block until the thread is resumed or shutdown.
        """
        while not self._controller.wait_for_resume(1.0):
            pass

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
