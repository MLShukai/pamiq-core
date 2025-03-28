import pytest

from pamiq_core.data.buffer import DataBuffer
from pamiq_core.state_persistence import PersistentStateMixin

from .helpers import DataBufferImpl


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
