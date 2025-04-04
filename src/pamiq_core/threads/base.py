import logging
from abc import ABC, abstractmethod
from typing import ClassVar

from pamiq_core.state_persistence import PersistentStateMixin
from .thread_control import ThreadEventMixin
from .thread_types import ThreadTypes
from pamiq_core.utils.reflection import get_class_module_path


class Thread(ABC, PersistentStateMixin, ThreadEventMixin):
    """Base class for all threads.

    This class provides a common interface for all threads in the system.

    Attributes:
        THREAD_TYPE: The type of the thread. Subclasses must define this class variable.
    """

    THREAD_TYPE: ClassVar[ThreadTypes]

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

    @abstractmethod
    def worker(self) -> None: ...

    def run(self) -> None:
        """Run the thread's worker method."""
        try:
            self.worker()
        except Exception:
            self._logger.exception(
                f"An exception has occurred in '{self.THREAD_TYPE.thread_name}' thread."
            )
            raise
