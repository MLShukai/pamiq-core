from collections import deque

import pytest

from pamiq_core.data.buffer import BufferData, StepData
from pamiq_core.data.implementations.sequential_buffer import SequentialBuffer


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
