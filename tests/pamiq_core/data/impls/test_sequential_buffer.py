from pathlib import Path

import pytest

from pamiq_core.data.impls.sequential_buffer import SequentialBuffer


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
