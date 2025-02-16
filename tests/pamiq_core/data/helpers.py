from collections import deque
from collections.abc import Iterable
from typing import Any

from pamiq_core.data.buffer import BufferData, DataBuffer, StepData


class DataBufferImpl(DataBuffer):
    """Reference implementation of DataBuffer using deque.

    This implementation is closer to production usage and is used for
    testing the DataBuffer interface itself.
    """

    def __init__(self, collecting_data_names: Iterable[str], max_size: int) -> None:
        super().__init__(collecting_data_names, max_size)
        self._buffer: dict[str, deque[Any]] = {
            name: deque(maxlen=max_size) for name in collecting_data_names
        }

    def add(self, step_data: StepData) -> None:
        for name in self._collecting_data_names:
            if name not in step_data:
                raise KeyError(f"Required data '{name}' not found in step_data")
            self._buffer[name].append(step_data[name])

    def get_data(self) -> BufferData:
        return self._buffer.copy()


class MockDataBuffer(DataBuffer):
    """Simple mock implementation of DataBuffer for testing.

    This implementation uses a list to store data and provides minimal
    functionality needed for testing higher-level components.
    """

    def __init__(self, collecting_data_names: list[str], max_size: int) -> None:
        super().__init__(collecting_data_names, max_size)
        self.data: list[StepData] = []

    def add(self, step_data: StepData) -> None:
        if len(self.data) < self.max_size:
            self.data.append(step_data)

    def get_data(self) -> BufferData:
        return {
            name: [d[name] for d in self.data] for name in self._collecting_data_names
        }
