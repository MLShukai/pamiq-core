import logging
import time
from typing import ClassVar

from pamiq_core.state_persistence import PersistentStateMixin
from pamiq_core.utils.reflection import get_class_module_path

from .thread_control import ThreadEventMixin
from .thread_types import ThreadTypes


class Thread(PersistentStateMixin, ThreadEventMixin):
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
