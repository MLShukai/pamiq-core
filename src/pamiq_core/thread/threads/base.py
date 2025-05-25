import logging
import threading
import time
from typing import ClassVar, override

from pamiq_core.utils.reflection import get_class_module_path

from ..thread_control import (
    ControllerCommandHandler,
    ReadOnlyController,
    ReadOnlyThreadStatus,
    ThreadEventMixin,
    ThreadStatus,
)
from ..thread_types import ThreadTypes


class Thread(ThreadEventMixin):
    """Base class for all threads.

    This class provides a common interface for all threads in the system.

    Attributes:
        THREAD_TYPE: The type of the thread. Subclasses must define this class variable.
        LOOP_DELAY: The delay between each loop iteration in seconds. Default is 1e-6 seconds.
    Methods:
        run: The main method that runs the thread.
        is_running: Returns True if the thread is running, False otherwise.
        on_start: Called when the thread starts.
        on_tick: Called on each loop iteration. Main processing logic should be implemented here.
        on_end: Called when the thread ends.
        on_exception: Called when an exception occurs in the thread.
        on_finally: Called when the thread is finally terminated.
    Raises:
        AttributeError: If the `THREAD_TYPE` attribute is not defined in the subclass.
    """

    THREAD_TYPE: ClassVar[ThreadTypes]
    LOOP_DELAY: ClassVar[float] = 1e-6  # prevent busy loops (and high CPU usage)

    def __init__(self) -> None:
        """Initialize Thread class.

        Raises:
            AttributeError: If `THREAD_TYPE` attribute is not defined.
        """
        if not hasattr(self, "THREAD_TYPE"):
            raise AttributeError(
                "Subclasses must define `THREAD_TYPE` attribute before instantiation."
            )
        self._logger = logging.getLogger(get_class_module_path(self.__class__))

    def run(self) -> None:
        """The main method that runs the thread.

        The methods `on_start`, `on_tick`, `on_end`, `on_exception`, and `on_finally`
        are called at the appropriate times during the thread's lifecycle.
        """
        self._logger.info(f"Start '{self.THREAD_TYPE.thread_name}' thread.")
        try:
            self.on_start()
            while self.is_running():
                self.on_tick()
                time.sleep(self.LOOP_DELAY)
            self.on_end()
        except Exception:
            self._logger.exception(
                f"An exception has occurred in '{self.THREAD_TYPE.thread_name}' thread."
            )
            self.on_exception()
            raise
        finally:
            self.on_finally()
            self._logger.info(f"End '{self.THREAD_TYPE.thread_name}' thread.")

    def is_running(self) -> bool:
        """Whether the thread is running or not.

        `on_tick` is called in a loop when this method returns True.
        """
        return True  # Return True or override in subclasses

    def on_start(self) -> None:
        """Called when the thread starts."""
        pass

    def on_tick(self) -> None:
        """Called on each loop iteration.

        Main processing logic should be implemented here.
        """
        pass

    def on_end(self) -> None:
        """Called when the thread ends."""
        pass

    def on_exception(self) -> None:
        """Called when an exception occurs in the thread."""
        pass

    def on_finally(self) -> None:
        """Called when the thread is finally terminated."""
        pass


class BackgroundThread(Thread):
    """Background thread class, a subclass of Thread and is used for background
    processing."""

    # Variable declaration for delay settings
    _controller_command_handler: ControllerCommandHandler

    def __init__(self) -> None:
        """Initialize BackgroundThread class.

        Raises:
            ValueError: If THREAD_TYPE is set to 'control'.
        """
        super().__init__()
        if self.THREAD_TYPE is ThreadTypes.CONTROL:
            raise ValueError("BackgroundThread cannot be of type 'control'.")

        self._thread = threading.Thread(target=self.run)
        self._thread_status = ThreadStatus()

    @property
    def thread_status(self) -> ReadOnlyThreadStatus:
        """Get a read-only view of the ThreadStatus object of this thread.

        Returns:
            A read-only interface to the thread status.
        """

        return self._thread_status.read_only

    def attach_controller(self, controller: ReadOnlyController) -> None:
        """Attach a controller to the thread.

        The controller is used to manage the thread's lifecycle and handle commands.

        Args:
            controller: The controller to attach.
        """
        self._controller_command_handler = ControllerCommandHandler(
            controller,
            self.on_paused,
            self.on_resumed,
        )

    @override
    def on_paused(self) -> None:
        """The method to be called when the thread is paused."""
        super().on_paused()
        self._thread_status.pause()

    @override
    def on_resumed(self) -> None:
        """The method to be called when the thread is resumed."""
        super().on_resumed()
        self._thread_status.resume()

    ### threading.Thread-like-API

    def start(self) -> None:
        """Start the thread like `threading.Thread.start()`"""
        self._thread.start()

    def join(self) -> None:
        """Join the thread like `threading.Thread.join()`"""
        self._thread.join()

    def is_alive(self) -> bool:
        """Check if the thread is alive like `threading.Thread.is_alive()`

        Returns:
            bool: True if the thread is alive, False otherwise.
        """
        return self._thread.is_alive()

    ### override event hook

    @override
    def is_running(self) -> bool:
        """Check if the thread is running.

        This is determined by the controller command handler.
        Returns:
            bool: True if the thread is running, False otherwise.
        """
        return self._controller_command_handler.manage_loop()

    @override
    def on_exception(self) -> None:
        """Called when an exception occurs in the thread."""
        self._thread_status.exception_raised()
