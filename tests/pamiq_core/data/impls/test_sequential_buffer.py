from pathlib import Path

import pytest

from pamiq_core.data.impls.sequential_buffer import (
    DictSequentialBuffer,
    SequentialBuffer,
)


class TestSequentialBuffer:
    """Test suite for SequentialBuffer."""

    @pytest.fixture
    def buffer(self) -> SequentialBuffer[int]:
        """Fixture providing a standard SequentialBuffer for tests."""
        return SequentialBuffer(100)

    def test_init(self):
        """Test SequentialBuffer initialization with various parameters."""
        # Test with standard parameters
        max_size = 50
        buffer = SequentialBuffer[int](max_size)

        assert buffer.max_size == max_size

    def test_add_and_get_data(self, buffer: SequentialBuffer[int]):
        """Test adding data to the buffer and retrieving it."""
        # Add data
        buffer.add(1)

        # Check data retrieval after adding one sample
        data = buffer.get_data()
        assert data == [1]

        # Add another sample
        buffer.add(2)

        # Check data retrieval after adding second sample
        data = buffer.get_data()
        assert data == [1, 2]

    def test_max_size_constraint(self):
        """Test the buffer respects its maximum size constraint."""
        max_size = 3
        buffer = SequentialBuffer[int](max_size)

        # Add more items than the max size
        for i in range(5):
            buffer.add(i)

        # Check only the most recent max_size items are kept
        data = buffer.get_data()
        assert data == [2, 3, 4]
        assert len(data) == max_size

    def test_get_data_returns_copy(self, buffer: SequentialBuffer[int]):
        """Test that get_data returns a copy that doesn't affect the internal
        state."""
        buffer.add(1)

        # Get data and modify it
        data = buffer.get_data()
        data.append(2)

        # Verify internal state is unchanged
        new_data = buffer.get_data()
        assert new_data == [1]
        assert len(new_data) == 1

    def test_save_and_load_state(self, buffer: SequentialBuffer[int], tmp_path: Path):
        """Test saving and loading the buffer state."""
        # Add some data to the buffer
        buffer.add(1)
        buffer.add(2)

        # Save state
        save_path = tmp_path / "test_buffer"
        buffer.save_state(save_path)

        # Verify file was created with .pkl extension
        assert save_path.with_suffix(".pkl").is_file()

        # Create a new buffer and load state
        new_buffer = SequentialBuffer[int](buffer.max_size)
        new_buffer.load_state(save_path)

        # Check that loaded data matches original
        original_data = buffer.get_data()
        loaded_data = new_buffer.get_data()

        assert loaded_data == original_data

    def test_len(self, buffer: SequentialBuffer[int]):
        """Test the __len__ method returns the correct buffer size."""
        assert len(buffer) == 0

        buffer.add(1)
        assert len(buffer) == 1

        buffer.add(2)
        assert len(buffer) == 2


class TestDictSequentialBuffer:
    """Test suite for DictSequentialBuffer class."""

    @pytest.mark.parametrize(
        "max_size,expected",
        [
            (10, 10),
            (100, 100),
            (1, 1),
        ],
    )
    def test_init_with_max_size(self, max_size, expected):
        """Test initialization with different max_size values."""
        buffer = DictSequentialBuffer[int](["a", "b"], max_size=max_size)
        assert buffer.max_size == expected
        assert len(buffer) == 0

    def test_init_with_empty_keys(self):
        """Test initialization with empty keys is allowed."""
        buffer = DictSequentialBuffer[int]([], max_size=10)
        assert len(buffer) == 0
        assert buffer.get_data() == {}

    @pytest.mark.parametrize(
        "keys,data",
        [
            (["a", "b"], {"a": 1, "b": 2}),
            ([], {}),  # Empty keys and data
        ],
    )
    def test_add_valid_data(self, keys, data):
        """Test adding valid data to buffer."""
        buffer = DictSequentialBuffer[int](keys, max_size=10)
        buffer.add(data)
        assert len(buffer) == 1

    @pytest.mark.parametrize(
        "keys,data",
        [
            (["a", "b"], {"a": 1}),  # Missing key
            (["a", "b"], {"a": 1, "b": 2, "c": 3}),  # Extra key
            (["a"], {"b": 1}),  # Different key
        ],
    )
    def test_add_invalid_data_raises(self, keys, data):
        """Test adding invalid data raises ValueError."""
        buffer = DictSequentialBuffer[int](keys, max_size=10)
        with pytest.raises(ValueError, match="Data keys.*do not match expected keys"):
            buffer.add(data)

    def test_get_data_structure(self):
        """Test get_data returns correct structure."""
        buffer = DictSequentialBuffer[int](["x", "y"], max_size=3)

        # Empty buffer
        assert buffer.get_data() == {"x": [], "y": []}

        # Add data
        buffer.add({"x": 1, "y": 10})
        buffer.add({"x": 2, "y": 20})

        result = buffer.get_data()
        assert result == {"x": [1, 2], "y": [10, 20]}

    def test_sequential_order_preserved(self):
        """Test that data maintains FIFO order."""
        buffer = DictSequentialBuffer[int](["a"], max_size=5)

        # Add data in order
        for i in range(5):
            buffer.add({"a": i})

        assert buffer.get_data()["a"] == [0, 1, 2, 3, 4]

    def test_oldest_removed_when_full(self):
        """Test that oldest entries are removed when buffer is full."""
        buffer = DictSequentialBuffer[int](["x", "y"], max_size=3)

        # Add more than capacity
        for i in range(5):
            buffer.add({"x": i, "y": i * 10})

        # Should keep only last 3
        result = buffer.get_data()
        assert result == {"x": [2, 3, 4], "y": [20, 30, 40]}
        assert len(buffer) == 3

    def test_save_and_load_state(self, tmp_path):
        """Test state persistence."""
        keys = ["a", "b"]
        buffer = DictSequentialBuffer[float](keys, max_size=10)

        # Add data
        buffer.add({"a": 1.0, "b": 2.0})
        buffer.add({"a": 3.0, "b": 4.0})

        # Save
        path = tmp_path / "state"
        buffer.save_state(path)

        # Load into new buffer
        new_buffer = DictSequentialBuffer[float](keys, max_size=10)
        new_buffer.load_state(path)

        assert new_buffer.get_data() == buffer.get_data()
        assert len(new_buffer) == len(buffer)

    def test_add_creates_shallow_copy(self):
        """Test that add creates a shallow copy of the input dictionary."""
        buffer = DictSequentialBuffer[int](["x", "y"], max_size=5)

        # Create original data
        original_data = {"x": 1, "y": 2}

        # Add to buffer
        buffer.add(original_data)

        # Modify original data by adding new key (which should not affect buffer)
        original_data["z"] = 999
        # Modify values
        original_data["x"] = 999
        original_data["y"] = 999

        # Get data from buffer
        buffer_data = buffer.get_data()

        # Buffer should have the original values
        assert buffer_data == {"x": [1], "y": [2]}
        assert "z" not in buffer_data

    def test_keys_property(self):
        """Test that keys property returns correct keys."""
        keys = {"a", "b", "c"}
        buffer = DictSequentialBuffer[int](keys, max_size=10)

        # Check keys property returns same keys
        assert buffer.keys == keys

        # Check keys property returns a copy
        returned_keys = buffer.keys
        returned_keys.add("d")
        assert buffer.keys == keys  # Should be unchanged
