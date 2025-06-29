import pickle
from collections import deque
from pathlib import Path
from typing import override

from ..buffer import DataBuffer


class SequentialBuffer[T](DataBuffer[T, list[T]]):
    """Implementation of DataBuffer that maintains data in sequential order.

    This buffer stores collected data points in an ordered queue,
    preserving the insertion order with a maximum size limit.
    """

    @override
    def __init__(self, max_size: int):
        """Initialize a new SequentialBuffer.

        Args:
            max_size: Maximum number of data points to store.
        """
        super().__init__(max_size)

        self._queue: deque[T] = deque(maxlen=max_size)

        self._current_size = 0
        self._max_size = max_size

    @property
    def max_size(self) -> int:
        """Returns the maximum number of data points that can be stored in the
        buffer."""
        return self._max_size

    @override
    def add(self, data: T) -> None:
        """Add a new data sample to the buffer.

        Args:
            data: Data element to add to the buffer.
        """
        self._queue.append(data)

        if self._current_size < self._max_size:
            self._current_size += 1

    @override
    def get_data(self) -> list[T]:
        """Retrieve all stored data from the buffer.

        Returns:
            List of all stored data elements preserving the original insertion order.
        """
        return list(self._queue)

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

        Saves the data queue to a pickle file with .pkl extension.

        Args:
            path: File path where to save the buffer state (without extension)
        """
        with open(path.with_suffix(".pkl"), "wb") as f:
            pickle.dump(self._queue, f)

    @override
    def load_state(self, path: Path) -> None:
        """Load the buffer state from the specified path.

        Loads data queue from pickle file with .pkl extension.

        Args:
            path: File path from where to load the buffer state (without extension)
        """
        with open(path.with_suffix(".pkl"), "rb") as f:
            self._queue = deque(pickle.load(f), maxlen=self._max_size)
        self._current_size = len(self._queue)
