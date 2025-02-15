from collections.abc import Iterable
from typing import Any

import pytest

from pamiq_core.data.buffer import BufferData, DataBuffer, StepData


class DataBufferImpl(DataBuffer):
    """Simple implementation of DataBuffer using lists to store data."""

    def __init__(self, collecting_data_names: Iterable[str], max_size: int) -> None:
        super().__init__(collecting_data_names, max_size)
        self._buffer: dict[str, list[Any]] = {
            name: [] for name in collecting_data_names
        }

    def add(self, step_data: StepData) -> None:
        for name in self._collecting_data_names:
            if name not in step_data:
                raise KeyError(f"Required data '{name}' not found in step_data")
            self._buffer[name].append(step_data[name])

            if len(self._buffer[name]) > self._max_size:
                self._buffer[name].pop(0)

    def get_data(self) -> BufferData:
        return self._buffer.copy()


class TestDataBuffer:
    """Test cases for DataBuffer class."""

    def test_init(self):
        """Test DataBuffer initialization with valid parameters."""
        data_names = ["state", "action", "reward"]
        max_size = 1000
        buffer = DataBufferImpl(data_names, max_size)

        assert buffer.max_size == max_size
        assert buffer.collecting_data_names == set(data_names)

    def test_init_negative_size(self):
        """Test DataBuffer initialization with negative max_size raises
        ValueError."""
        data_names = ["state", "action"]
        max_size = -1

        with pytest.raises(ValueError, match="max_size must be non-negative"):
            DataBufferImpl(data_names, max_size)

    def test_collecting_data_names_immutable(self):
        """Test that collecting_data_names property returns a copy that cannot
        affect the internal state."""
        data_names = ["state", "action"]
        buffer = DataBufferImpl(data_names, 100)

        # Get the data names and try to modify them
        names = buffer.collecting_data_names
        names.add("reward")

        # Original internal state should be unchanged
        assert buffer.collecting_data_names == set(data_names)

    def test_max_size_property(self):
        """Test that max_size property returns the correct value."""
        max_size = 500
        buffer = DataBufferImpl(["state"], max_size)

        assert buffer.max_size == max_size
