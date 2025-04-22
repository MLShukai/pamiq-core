from pathlib import Path

import pytest

from pamiq_core.data.buffer import StepData
from pamiq_core.data.impls.sequential_buffer import SequentialBuffer


class TestSequentialBuffer:
    """Test suite for SequentialBuffer."""

    @pytest.fixture
    def buffer(self) -> SequentialBuffer:
        """Fixture providing a standard SequentialBuffer for tests."""
        return SequentialBuffer(["state", "action", "reward"], 100)

    def test_init(self):
        """Test SequentialBuffer initialization with various parameters."""
        # Test with standard parameters
        data_names = ["state", "action", "reward"]
        max_size = 50
        buffer = SequentialBuffer(data_names, max_size)

        assert buffer.max_size == max_size
        assert buffer.collecting_data_names == set(data_names)

    def test_add_and_get_data(self, buffer: SequentialBuffer):
        """Test adding data to the buffer and retrieving it."""
        # Sample data
        sample1: StepData = {"state": [1.0, 0.0], "action": 1, "reward": 0.5}
        sample2: StepData = {"state": [0.0, 1.0], "action": 0, "reward": -0.5}

        # Add data
        buffer.add(sample1)

        # Check data retrieval after adding one sample
        data = buffer.get_data()
        assert data["state"] == [[1.0, 0.0]]
        assert data["action"] == [1]
        assert data["reward"] == [0.5]

        # Add another sample
        buffer.add(sample2)

        # Check data retrieval after adding second sample
        data = buffer.get_data()
        assert data["state"] == [[1.0, 0.0], [0.0, 1.0]]
        assert data["action"] == [1, 0]
        assert data["reward"] == [0.5, -0.5]

    def test_max_size_constraint(self):
        """Test the buffer respects its maximum size constraint."""
        max_size = 3
        buffer = SequentialBuffer(["value"], max_size)

        # Add more items than the max size
        for i in range(5):
            buffer.add({"value": i})

        # Check only the most recent max_size items are kept
        data = buffer.get_data()
        assert data["value"] == [2, 3, 4]
        assert len(data["value"]) == max_size

    def test_missing_data_field(self, buffer: SequentialBuffer):
        """Test adding data with missing required fields raises KeyError."""
        incomplete_data: StepData = {
            "state": [1.0, 0.0],
            "action": 1,
        }  # Missing 'reward'

        with pytest.raises(
            KeyError, match="Required data 'reward' not found in step_data"
        ):
            buffer.add(incomplete_data)

    def test_get_data_returns_copy(self, buffer: SequentialBuffer):
        """Test that get_data returns a copy that doesn't affect the internal
        state."""
        buffer.add({"state": [1.0], "action": 1, "reward": 0.5})

        # Get data and modify it
        data = buffer.get_data()
        data["state"].append([2.0])

        # Verify internal state is unchanged
        new_data = buffer.get_data()
        assert new_data["state"] == [[1.0]]
        assert len(new_data["state"]) == 1

    def test_save_and_load_state(self, buffer: SequentialBuffer, tmp_path: Path):
        """Test saving and loading the buffer state."""
        # Add some data to the buffer
        buffer.add({"state": [1.0, 0.0], "action": 1, "reward": 0.5})
        buffer.add({"state": [0.0, 1.0], "action": 0, "reward": -0.5})

        # Save state
        save_path = tmp_path / "test_buffer"
        buffer.save_state(save_path)

        # Verify files were created
        for name in buffer.collecting_data_names:
            assert (save_path / f"{name}.pkl").is_file()

        # Create a new buffer and load state
        new_buffer = SequentialBuffer(buffer.collecting_data_names, buffer.max_size)
        new_buffer.load_state(save_path)

        # Check that loaded data matches original
        original_data = buffer.get_data()
        loaded_data = new_buffer.get_data()

        assert loaded_data == original_data

        for name in buffer.collecting_data_names:
            assert loaded_data[name] == original_data[name]

    def test_len(self, buffer: SequentialBuffer):
        """Test the __len__ method returns the correct buffer size."""
        assert len(buffer) == 0

        buffer.add({"state": [1.0, 0.0], "action": 1, "reward": 0.5})
        assert len(buffer) == 1

        buffer.add({"state": [0.0, 1.0], "action": 0, "reward": -0.5})
        assert len(buffer) == 2
