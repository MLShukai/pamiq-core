from collections import deque
from collections.abc import Iterable
from typing import override

from ..buffer import DataBuffer, StepData


class SequentialBuffer[T](DataBuffer[T]):
    """Implementation of DataBuffer that maintains data in sequential order.

    This buffer stores collected data points in ordered queues,
    preserving the insertion order. Each data field is stored in a
    separate queue with a maximum size limit.
    """

    @override
    def __init__(self, collecting_data_names: Iterable[str], max_size: int):
        """Initialize a new SequentialBuffer.

        Args:
            collecting_data_names: Names of data fields to collect.
            max_size: Maximum number of data points to store.
        """
        super().__init__(collecting_data_names, max_size)

        self._queues_dict: dict[str, deque[T]] = {
            name: deque(maxlen=max_size) for name in collecting_data_names
        }

    @override
    def add(self, step_data: StepData[T]) -> None:
        """Add a new data sample to the buffer.

        Args:
            step_data: Dictionary containing data for one step. Must contain
                all fields specified in collecting_data_names.

        Raises:
            KeyError: If a required data field is missing from step_data.
        """
        for name in self.collecting_data_names:
            if name not in step_data:
                raise KeyError(f"Required data '{name}' not found in step_data")
            self._queues_dict[name].append(step_data[name])

    @override
    def get_data(self) -> dict[str, list[T]]:
        """Retrieve all stored data from the buffer.

        Returns:
            Dictionary mapping data field names to lists of their values.
            Each list preserves the original insertion order.
        """
        return {name: list(queue) for name, queue in self._queues_dict.items()}
