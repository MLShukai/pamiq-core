import time
from collections import deque
from collections.abc import Iterable

from .buffer import StepData


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
