import pickle
from collections import deque
from collections.abc import Iterable
from pathlib import Path
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

        self._current_size = 0

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

        if self._current_size < self.max_size:
            self._current_size += 1

    @override
    def get_data(self) -> dict[str, list[T]]:
        """Retrieve all stored data from the buffer.

        Returns:
            Dictionary mapping data field names to lists of their values.
            Each list preserves the original insertion order.
        """
        return {name: list(queue) for name, queue in self._queues_dict.items()}

    @override
    def __len__(self) -> int:
        """Returns the current number of samples in the buffer.

        Returns:
            int: The number of samples currently stored in the buffer.
        """
        return self._current_size

    @override
    def save_state(self, path: Path) -> None:
        """Save the buffer state to the specified path.

        Creates a directory at the given path and saves each data queue as a
        separate pickle file.

        Args:
            path: Directory path where to save the buffer state
        """
        path.mkdir()
        for name, queue in self._queues_dict.items():
            with open(path / f"{name}.pkl", "wb") as f:
                pickle.dump(queue, f)

    @override
    def load_state(self, path: Path) -> None:
        """Load the buffer state from the specified path.

        Loads data queues from pickle files in the given directory.

        Args:
            path: Directory path from where to load the buffer state
        """
        for name in self.collecting_data_names:
            with open(path / f"{name}.pkl", "rb") as f:
                queue = deque(pickle.load(f), maxlen=self.max_size)
                self._queues_dict[name] = queue
