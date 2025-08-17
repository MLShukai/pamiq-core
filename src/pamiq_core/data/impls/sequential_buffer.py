import pickle
from collections import deque
from collections.abc import Iterable, Mapping
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
        return len(self._queue)

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


class DictSequentialBuffer[T](DataBuffer[Mapping[str, T], dict[str, list[T]]]):
    """Buffer implementation that stores dictionary data in sequential order.

    See: [`SequentialBuffer`][pamiq_core.data.impls.SequentialBuffer]
    """

    def __init__(
        self,
        keys: Iterable[str],
        max_size: int,
    ) -> None:
        """Initialize a DictSequentialBuffer.

        Args:
            keys: The keys that must be present in all data dictionaries.
            max_size: Maximum number of data points to store.
        """
        self._buffer = SequentialBuffer[dict[str, T]](max_size)
        super().__init__(self._buffer.max_queue_size)

        self._keys = set(keys)

        self.save_state = self._buffer.save_state
        self.load_state = self._buffer.load_state

    @property
    def max_size(self) -> int:
        """Returns the maximum number of data points that can be stored in the
        buffer."""
        return self._buffer.max_size

    @property
    def keys(self) -> set[str]:
        """Returns a copy of the set of keys required for all data
        dictionaries.

        A copy is returned to prevent external modification of the
        internal key set.
        """
        return self._keys.copy()

    @override
    def add(self, data: Mapping[str, T]) -> None:
        """Add a new data sample to the buffer.

        The data must contain exactly the keys specified during initialization.
        If the buffer is full, the oldest entry will be removed.

        Args:
            data: Dictionary containing data for each key.

        Raises:
            ValueError: If the data keys don't match the expected keys.
        """
        if set(data.keys()) != self._keys:
            raise ValueError(
                f"Data keys {set(data.keys())} do not match expected keys {self._keys}"
            )
        return self._buffer.add(dict(data))

    @override
    def get_data(self) -> dict[str, list[T]]:
        """Retrieve all stored data from the buffer.

        Returns:
            Dictionary mapping each key to a list of its stored values.
            The lists maintain the sequential order in which data was added.
        """
        out = {k: list[T]() for k in self._keys}
        for data in self._buffer.get_data():
            for k, v in data.items():
                out[k].append(v)
        return out

    @override
    def __len__(self) -> int:
        """Returns the current number of samples in the buffer.

        Returns:
            int: The number of samples currently stored in the buffer.
        """
        return len(self._buffer)
