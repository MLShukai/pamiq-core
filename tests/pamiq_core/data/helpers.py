from typing import override

from pamiq_core.data.buffer import DataBuffer


class MockDataBuffer[T](DataBuffer[T, list[T]]):
    """Simple mock implementation of DataBuffer for testing.

    This implementation uses a list to store data and provides minimal
    functionality needed for testing higher-level components.
    """

    def __init__(self, max_queue_size: int | None = None) -> None:
        super().__init__(max_queue_size)
        # For mock, use a reasonable default if None
        self._max_size = 100 if max_queue_size is None else max_queue_size
        self.data: list[T] = []

    @override
    def add(self, data: T) -> None:
        if len(self.data) < self._max_size:
            self.data.append(data)

    @override
    def get_data(self) -> list[T]:
        return self.data.copy()

    @override
    def __len__(self) -> int:
        return len(self.data)
