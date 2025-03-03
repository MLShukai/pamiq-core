from collections import deque
from collections.abc import Iterable
from threading import RLock

from pamiq_core import time

from .buffer import BufferData, DataBuffer, StepData


class TimestampingQueuesDict:
    """A dictionary of queues that stores data with timestamps.

    This class maintains multiple queues for different data streams,
    where each queue is synchronized with a timestamp queue. It provides
    functionality to append and retrieve data along with their
    corresponding timestamps.
    """

    def __init__(self, queue_names: Iterable[str], max_len: int) -> None:
        """Initialize queues for given names with specified maximum length.

        Args:
            queue_names: Names of the queues to be created.
            max_len: Maximum length of each queue.
        """
        self._queues = {k: deque(maxlen=max_len) for k in queue_names}
        self._timestamps = deque(maxlen=max_len)

    def append(self, data: StepData) -> None:
        """Append new data to all queues with current timestamp.

        Args:
            data: Step data containing values for each queue.
        """
        for k, q in self._queues.items():
            q.append(data[k])
        self._timestamps.append(time.time())

    def popleft(self) -> tuple[StepData, float]:
        """Remove and return the leftmost elements from all queues with
        timestamp.

        Returns:
            A tuple containing:
                - Dictionary mapping queue names to their leftmost values
                - Timestamp corresponding to these values
        """
        return (
            {k: q.popleft() for k, q in self._queues.items()},
            self._timestamps.popleft(),
        )

    def __len__(self) -> int:
        """Return the current length of the queues.

        Returns:
            Number of elements currently in the queues.
        """
        return len(self._timestamps)


class DataUser[T: DataBuffer]:
    """A class that manages data buffering and timestamps for collected data.

    This class acts as a user of data buffers, handling the collection,
    storage, and retrieval of data along with their timestamps. It works
    in conjunction with a DataCollector to manage concurrent data
    collection.
    """

    def __init__(self, buffer: T) -> None:
        """Initialize DataUser with a specified buffer.

        Args:
            buffer: Data buffer instance to store collected data.
        """
        self._buffer = buffer
        self._timestamps = deque(maxlen=buffer.max_size)
        # DataCollector instance is only accessed from DataUser and Container classes
        self._collector = DataCollector(self)

    def create_empty_queues(self) -> TimestampingQueuesDict:
        """Create empty timestamping queues for data collection.

        Returns:
            New instance of TimestampingQueuesDict with appropriate configuration.
        """
        return TimestampingQueuesDict(
            self._buffer.collecting_data_names, self._buffer.max_size
        )

    def update(self) -> None:
        """Update buffer with collected data from the collector.

        Moves all collected data from the collector to the buffer and
        records their timestamps.
        """
        queues = self._collector._move_data()
        for _ in range(len(queues)):
            data, t = queues.popleft()
            self._buffer.add(data)
            self._timestamps.append(t)

    def get_data(self) -> BufferData:
        """Retrieve data from the buffer.

        Returns:
            Current data stored in the buffer.
        """
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
            if t < timestamp:
                return i
        return len(self._timestamps)


class DataCollector[T: DataBuffer]:
    """A thread-safe collector for buffered data.

    This class provides concurrent data collection capabilities with
    thread safety, working in conjunction with DataUser to manage data
    collection and transfer.
    """

    def __init__(self, user: DataUser[T]) -> None:
        """Initialize DataCollector with a specified DataUser.

        Args:
            user: DataUser instance this collector is associated with.
        """
        self._user = user
        self._queues_dict = user.create_empty_queues()
        self._lock = RLock()

    def collect(self, step_data: StepData) -> None:
        """Collect step data in a thread-safe manner.

        Args:
            step_data: Data to be collected.
        """
        with self._lock:
            self._queues_dict.append(step_data)

    def _move_data(self) -> TimestampingQueuesDict:
        """Move collected data to a new queue and reset the collector.

        This method is intended to be called only by the associated DataUser.

        Returns:
            TimestampingQueuesDict containing the collected data.
        """
        with self._lock:
            data = self._queues_dict
            self._queues_dict = self._user.create_empty_queues()
            return data
