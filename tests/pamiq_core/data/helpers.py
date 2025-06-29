from typing import override

from pamiq_core.data.buffer import DataBuffer, StepData


class MockDataBuffer(DataBuffer):
    """Simple mock implementation of DataBuffer for testing.

    This implementation uses a list to store data and provides minimal
    functionality needed for testing higher-level components.
    """

    def __init__(self, collecting_data_names: list[str], max_size: int) -> None:
        super().__init__(collecting_data_names, max_size)
        self.data: list[StepData] = []

    @override
    def add(self, step_data: StepData) -> None:
        if len(self.data) < self.max_size:
            self.data.append(step_data)

    @override
    def get_data(self):
        return {
            name: [d[name] for d in self.data] for name in self._collecting_data_names
        }

    @override
    def __len__(self) -> int:
        return len(self.data)
