from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence

from pamiq_core.state_persistence import PersistentStateMixin

type StepData[T] = Mapping[str, T]
type BufferData[T] = Mapping[str, Sequence[T]]


class DataBuffer[T](ABC, PersistentStateMixin):
    """Interface for managing experience data collected during system
    execution.

    DataBuffer provides an interface for collecting and managing
    experience data generated during system execution. It maintains a
    buffer of fixed maximum size that stores data for specified data
    names.
    """

    def __init__(self, collecting_data_names: Iterable[str], max_size: int) -> None:
        """Initializes the DataBuffer.

        Args:
            collecting_data_names: Names of data fields to collect and store.
            max_size: Maximum number of samples to store in the buffer.

        Raises:
            ValueError: If max_size is negative.
        """
        super().__init__()
        self._collecting_data_names = set(collecting_data_names)
        if max_size < 0:
            raise ValueError("max_size must be non-negative")
        self._max_size = max_size

    @property
    def collecting_data_names(self) -> set[str]:
        """Returns the set of data field names being collected."""
        return self._collecting_data_names.copy()

    @property
    def max_size(self) -> int:
        """Returns the maximum number of samples that can be stored."""
        return self._max_size

    @abstractmethod
    def add(self, step_data: StepData[T]) -> None:
        """Adds a new data sample to the buffer.

        Args:
            step_data: Dictionary containing data for one step. Must contain
                all fields specified in collecting_data_names.
        """
        ...

    @abstractmethod
    def get_data(self) -> BufferData[T]:
        """Retrieves all stored data from the buffer.

        Returns:
            Dictionary mapping data field names to sequences of their values.
            Each sequence has the same length.
        """
        ...

    @abstractmethod
    def __len__(self) -> int:
        """Returns the current number of samples in the buffer.

        Returns:
            int: The number of samples currently stored in the buffer.
        """
        ...
