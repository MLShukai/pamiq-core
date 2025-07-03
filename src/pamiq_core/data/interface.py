import pickle
from collections import deque
from pathlib import Path
from threading import RLock
from typing import Any, override

from pamiq_core import time
from pamiq_core.state_persistence import PersistentStateMixin

from .buffer import DataBuffer


class TimestampingQueue[T]:
    """A queue that stores data with timestamps.

    This class maintains a single queue for data elements, synchronized
    with a timestamp queue. It provides functionality to append and
    retrieve data along with their corresponding timestamps.
    """

    def __init__(self, max_len: int | None = None) -> None:
        """Initialize queue with specified maximum length.

        Args:
            max_len: Maximum length of the queue.
        """
        self._queue: deque[T] = deque(maxlen=max_len)
        self._timestamps: deque[float] = deque(maxlen=max_len)

    def append(self, data: T) -> None:
        """Append new data to the queue with current timestamp.

        Args:
            data: Data element to append.
        """
        self._queue.append(data)
        self._timestamps.append(time.time())

    def popleft(self) -> tuple[T, float]:
        """Remove and return the leftmost element with timestamp.

        Returns:
            A tuple containing:
                - The leftmost data element
                - Timestamp corresponding to this value
        """
        return (
            self._queue.popleft(),
            self._timestamps.popleft(),
        )

    def __len__(self) -> int:
        """Return the current length of the queues.

        Returns:
            Number of elements currently in the queues.
        """
        return len(self._timestamps)


class DataUser[R](PersistentStateMixin):
    """A class that manages data buffering and timestamps for collected data.

    This class acts as a user of data buffers, handling the collection,
    storage, and retrieval of data along with their timestamps. It works
    in conjunction with a DataCollector to manage concurrent data
    collection.

    Type Parameters:
        R: The return type of the buffer's get_data() method.
    """

    def __init__(self, buffer: DataBuffer[Any, R]) -> None:
        """Initialize DataUser with a specified buffer.

        Args:
            buffer: Data buffer instance to store collected data.
        """
        self._buffer = buffer
        self._timestamps: deque[float] = deque(maxlen=buffer.max_queue_size)
        # DataCollector instance is only accessed from DataUser and Container classes
        self._collector = DataCollector[Any](buffer.max_queue_size)

    def update(self) -> None:
        """Update buffer with collected data from the collector.

        Moves all collected data from the collector to the buffer and
        records their timestamps.
        """
        queue = self._collector._move_data()  # pyright: ignore[reportPrivateUsage]
        for _ in range(len(queue)):
            data, t = queue.popleft()
            self._buffer.add(data)
            self._timestamps.append(t)

    def get_data(self) -> R:
        """Update and retrieve data from the buffer.

        Returns:
            Current data stored in the buffer.
        """
        self.update()
        return self._buffer.get_data()

    def count_data_added_since(self, timestamp: float) -> int:
        """Count the number of data points added after the specified timestamp.

        NOTE: Use `pamiq_core.time` to retrieve `timestamp`.

        Args:
            timestamp: Reference timestamp to count from.

        Returns:
            Number of data points added after the specified timestamp.
        """
        for i, t in enumerate(reversed(self._timestamps)):
            if t <= timestamp:
                return i
        return len(self._timestamps)

    @override
    def save_state(self, path: Path) -> None:
        """Save the state of this DataUser to the specified path.

        This method first updates the buffer with any pending collected data,
        then delegates the state saving to the underlying buffer.

        Args:
            path: Directory path where the state should be saved
        """
        self.update()
        path.mkdir()
        self._buffer.save_state(path / "buffer")
        with open(path / "timestamps.pkl", "wb") as f:
            pickle.dump(self._timestamps, f)

    @override
    def load_state(self, path: Path) -> None:
        """Load the state of this DataUser from the specified path.

        This method delegates the state loading to the underlying buffer.

        Args:
            path: Directory path from where the state should be loaded
        """
        self._buffer.load_state(path / "buffer")
        with open(path / "timestamps.pkl", "rb") as f:
            self._timestamps = deque(pickle.load(f), maxlen=self._buffer.max_queue_size)

    def __len__(self) -> int:
        """Returns the current number of samples in the buffer."""
        return len(self._buffer)


class DataCollector[T]:
    """A thread-safe collector for buffered data.

    This class provides concurrent data collection capabilities with
    thread safety, working in conjunction with DataUser to manage data
    collection and transfer.

    Type Parameters:
        T: The type of individual data elements.
    """

    def __init__(self, max_queue_size: int | None) -> None:
        """Initialize DataCollector with a specified queue size.

        Args:
            max_queue_size: Maximum size of the collection queue. If None, queue has no size limit.
        """
        self._max_queue_size = max_queue_size
        self._queue = TimestampingQueue[T](max_queue_size)
        self._lock = RLock()

    def collect(self, data: T) -> None:
        """Collect data in a thread-safe manner.

        Args:
            data: Data to be collected.
        """
        with self._lock:
            self._queue.append(data)

    def _move_data(self) -> TimestampingQueue[T]:
        """Move collected data to a new queue and reset the collector.

        This method is intended to be called only by the associated DataUser.

        Returns:
            TimestampingQueue containing the collected data.
        """
        with self._lock:
            out, self._queue = self._queue, TimestampingQueue[T](self._max_queue_size)
        return out
