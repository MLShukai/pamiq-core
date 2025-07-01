from collections import deque
from typing import Any, override

import pytest

from pamiq_core.data.buffer import DataBuffer
from pamiq_core.state_persistence import PersistentStateMixin


class DataBufferImpl(DataBuffer[Any, deque[Any]]):
    """Reference implementation of DataBuffer using deque.

    This implementation is closer to production usage and is used for
    testing the DataBuffer interface itself.
    """

    def __init__(self, max_queue_size: int | None = None) -> None:
        super().__init__(max_queue_size)
        # For testing purposes, use a fixed buffer size
        self._max_size = 1000 if max_queue_size is None else max_queue_size
        self._buffer: deque[Any] = deque(maxlen=self._max_size)
        self._current_size = 0

    @override
    def add(self, data: Any) -> None:
        self._buffer.append(data)
        if self._current_size < self._max_size:
            self._current_size += 1

    @override
    def get_data(self) -> deque[Any]:
        return self._buffer.copy()

    @override
    def __len__(self) -> int:
        return self._current_size


class TestDataBuffer:
    """Test cases for DataBuffer class."""

    def test_persistent_state_mixin_subclass(self):
        """Test DataBuffer is PersistentStateMixin subclass."""
        assert issubclass(DataBuffer, PersistentStateMixin)

    @pytest.mark.parametrize("name", ["__len__", "get_data", "add"])
    def test_abstract_methods(self, name):
        """Test that abstract method in DataBuffer."""
        assert name in DataBuffer.__abstractmethods__

    def test_init(self):
        """Test DataBuffer initialization with valid parameters."""
        max_queue_size = 1000
        buffer = DataBufferImpl(max_queue_size)

        assert buffer.max_queue_size == max_queue_size

    def test_init_negative_size(self):
        """Test DataBuffer initialization with negative max_queue_size raises
        ValueError."""
        max_queue_size = -1

        with pytest.raises(ValueError, match="max_queue_size must be non-negative"):
            DataBufferImpl(max_queue_size)

    def test_init_with_none_warns(self):
        """Test DataBuffer initialization with None max_queue_size warns about
        memory."""
        with pytest.warns(RuntimeWarning, match="max_queue_size is None"):
            buffer = DataBufferImpl(None)

        assert buffer.max_queue_size is None
