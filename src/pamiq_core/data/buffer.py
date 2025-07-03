import warnings
from abc import ABC, abstractmethod

from pamiq_core.state_persistence import PersistentStateMixin


class DataBuffer[T, R](ABC, PersistentStateMixin):
    """Interface for managing experience data collected during system
    execution.

    DataBuffer provides an interface for collecting and managing
    experience data generated during system execution. It maintains a
    buffer of fixed maximum size.

    Type Parameters:
        - T: The type of individual data elements.
        - R: The return type of the get_data() method.
    """

    def __init__(self, max_queue_size: int | None = None) -> None:
        """Initializes the DataBuffer.

        Args:
            max_queue_size: Maximum number of samples to store in the collector's queue.
                When the queue size exceeds this limit, old data will be deleted.
                If None, the queue will have unlimited size (may cause memory issues).

        Raises:
            ValueError: If max_queue_size is negative.
        """
        super().__init__()
        if max_queue_size is not None:
            if max_queue_size < 0:
                raise ValueError("max_queue_size must be non-negative")
        else:
            warnings.warn(
                "max_queue_size is None. The collector's queue will have unlimited size, "
                "which may cause memory issues if data collection is faster than processing.",
                RuntimeWarning,
                stacklevel=2,
            )
        self._max_queue_size = max_queue_size

    @property
    def max_queue_size(self) -> int | None:
        """Returns the maximum number of samples that can be stored in the
        collector's queue."""
        return self._max_queue_size

    @abstractmethod
    def add(self, data: T) -> None:
        """Adds a new data sample to the buffer.

        Args:
            data: Data element to add to the buffer.
        """
        pass

    @abstractmethod
    def get_data(self) -> R:
        """Retrieves all stored data from the buffer.

        Returns:
            Data structure containing all stored samples.
        """
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Returns the current number of samples in the buffer.

        Returns:
            int: The number of samples currently stored in the buffer.
        """
        pass
