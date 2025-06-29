from abc import ABC, abstractmethod

from pamiq_core.state_persistence import PersistentStateMixin


class DataBuffer[T, R](ABC, PersistentStateMixin):
    """Interface for managing experience data collected during system
    execution.

    DataBuffer provides an interface for collecting and managing
    experience data generated during system execution. It maintains a
    buffer of fixed maximum size.

    Type Parameters:
        T: The type of data stored in each step.
        R: The return type of the get_data() method.
    """

    def __init__(self, max_size: int) -> None:
        """Initializes the DataBuffer.

        Args:
            max_size: Maximum number of samples to store in the buffer.

        Raises:
            ValueError: If max_size is negative.
        """
        super().__init__()
        if max_size < 0:
            raise ValueError("max_size must be non-negative")
        self._max_size = max_size

    @property
    def max_size(self) -> int:
        """Returns the maximum number of samples that can be stored."""
        return self._max_size

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
