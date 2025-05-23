import pickle
import random
from pathlib import Path
from typing import Any

import pytest

from pamiq_core.data.impls.random_replacement_buffer import (
    RandomReplacementBuffer,
)


class TestRandomReplacementBuffer:
    """Test suite for RandomReplacementBuffer class."""

    @pytest.fixture
    def buffer(self) -> RandomReplacementBuffer[Any]:
        """Fixture providing a standard RandomReplacementBuffer for tests."""
        return RandomReplacementBuffer(["state", "action", "reward"], 5)

    def test_init(self):
        """Test RandomReplacementBuffer initialization with various
        parameters."""
        # Test with standard parameters
        data_names = ["state", "action", "reward"]
        max_size = 10
        buffer = RandomReplacementBuffer(data_names, max_size)

        assert buffer.max_size == max_size
        assert buffer.collecting_data_names == set(data_names)
        assert buffer._replace_probability == 1.0
        assert buffer._current_size == 0
        assert not buffer.is_full

        # Test with custom replace probability
        buffer = RandomReplacementBuffer(data_names, max_size, replace_probability=0.5)
        assert buffer._replace_probability == 0.5

        # Test with invalid replace probability
        with pytest.raises(ValueError):
            RandomReplacementBuffer(data_names, max_size, replace_probability=-0.1)

        with pytest.raises(ValueError):
            RandomReplacementBuffer(data_names, max_size, replace_probability=1.1)

    def test_init_with_expected_survival_length(self):
        """Test initialization with expected_survival_length parameter."""
        data_names = ["state", "action", "reward"]
        max_size = 10

        # Test with expected_survival_length only
        buffer = RandomReplacementBuffer(
            data_names, max_size, expected_survival_length=20
        )

        # Should compute probability automatically
        assert 0.0 <= buffer._replace_probability <= 1.0
        assert buffer.max_size == max_size
        assert buffer.collecting_data_names == set(data_names)

    def test_init_with_both_parameters_raises_error(self):
        """Test that specifying both replace_probability and
        expected_survival_length raises ValueError."""
        data_names = ["state", "action", "reward"]
        max_size = 10

        with pytest.raises(
            ValueError,
            match="Cannot specify both replace_probability and expected_survival_length",
        ):
            RandomReplacementBuffer(
                data_names,
                max_size,
                replace_probability=0.5,
                expected_survival_length=20,
            )

    def test_init_with_none_parameters(self):
        """Test initialization when both parameters are None (should default to
        1.0)."""
        data_names = ["state", "action", "reward"]
        max_size = 10

        buffer = RandomReplacementBuffer(data_names, max_size)
        assert buffer._replace_probability == 1.0

    @pytest.mark.parametrize(
        "max_size,survival_length",
        [
            (10, 5),
            (10, 10),
            (10, 20),
            (100, 50),
            (100, 200),
            (50, 25),
            (50, 100),
        ],
    )
    def test_compute_replace_probability_from_expected_survival_length(
        self, max_size, survival_length
    ):
        """Test the static method for computing replace probability."""
        probability = RandomReplacementBuffer.compute_replace_probability_from_expected_survival_length(
            max_size, survival_length
        )
        assert (
            0.0 <= probability <= 1.0
        ), f"Probability {probability} out of range for max_size={max_size}, survival_length={survival_length}"

    def test_add_and_get_data(self, buffer: RandomReplacementBuffer[Any]):
        """Test adding data to the buffer and retrieving it."""
        # Sample data
        sample1 = {"state": [1.0, 0.0], "action": 1, "reward": 0.5}
        sample2 = {"state": [0.0, 1.0], "action": 0, "reward": -0.5}

        # Add data
        buffer.add(sample1)

        # Check data retrieval after adding one sample
        data = buffer.get_data()
        assert data["state"] == [[1.0, 0.0]]
        assert data["action"] == [1]
        assert data["reward"] == [0.5]
        assert buffer._current_size == 1

        # Add another sample
        buffer.add(sample2)

        # Check data retrieval after adding second sample
        data = buffer.get_data()
        assert data["state"] == [[1.0, 0.0], [0.0, 1.0]]
        assert data["action"] == [1, 0]
        assert data["reward"] == [0.5, -0.5]
        assert buffer._current_size == 2

    def test_is_full_property(self):
        """Test the is_full property correctly reports buffer fullness."""
        buffer = RandomReplacementBuffer(["value"], 3)

        assert not buffer.is_full

        # Fill the buffer
        for i in range(3):
            buffer.add({"value": i})

        assert buffer.is_full

    def test_replacement_when_full(self, monkeypatch):
        """Test the random replacement behavior when buffer is full."""
        # Create a small buffer
        buffer = RandomReplacementBuffer(["value"], 2)

        # Fill the buffer
        buffer.add({"value": "A"})
        buffer.add({"value": "B"})
        assert buffer.is_full

        # Mock random functions to get deterministic behavior
        monkeypatch.setattr(random, "random", lambda: 0.0)  # Always below probability
        monkeypatch.setattr(
            random, "randint", lambda a, b: 0
        )  # Always replace first element

        # Add another item
        buffer.add({"value": "C"})

        # Check that first element was replaced
        assert buffer.get_data()["value"] == ["C", "B"]

    def test_skip_replacement(self, monkeypatch):
        """Test that replacement is skipped based on probability."""
        # Create a buffer with low replacement probability
        buffer = RandomReplacementBuffer(["value"], 2, replace_probability=0.3)

        # Fill the buffer
        buffer.add({"value": "A"})
        buffer.add({"value": "B"})
        assert buffer.is_full

        # Mock random to always be above the replacement probability
        monkeypatch.setattr(random, "random", lambda: 0.9)

        # Try to add another item
        buffer.add({"value": "C"})

        # Check that no replacement occurred
        assert buffer.get_data()["value"] == ["A", "B"]

    def test_missing_data_field(self, buffer: RandomReplacementBuffer[Any]):
        """Test adding data with missing required fields raises KeyError."""
        incomplete_data = {"state": [1.0, 0.0], "action": 1}  # Missing 'reward'

        with pytest.raises(
            KeyError, match="Required data 'reward' not found in step_data"
        ):
            buffer.add(incomplete_data)

    def test_get_data_returns_copy(self, buffer: RandomReplacementBuffer[Any]):
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

    def test_save_and_load_state(
        self, buffer: RandomReplacementBuffer[Any], tmp_path: Path
    ):
        """Test saving and loading the buffer state."""
        # Add some data to the buffer
        buffer.add({"state": [1.0, 0.0], "action": 1, "reward": 0.5})
        buffer.add({"state": [0.0, 1.0], "action": 0, "reward": -0.5})

        # Save state
        save_path = tmp_path / "test_buffer"
        buffer.save_state(save_path)

        # Verify files were created
        for name in buffer.collecting_data_names:
            assert (save_path / f"{name}.pkl").exists()

        # Create a new buffer and load state
        new_buffer = RandomReplacementBuffer(
            buffer.collecting_data_names, buffer.max_size
        )
        new_buffer.load_state(save_path)

        # Check that loaded data matches original
        original_data = buffer.get_data()
        loaded_data = new_buffer.get_data()

        assert loaded_data == original_data
        assert new_buffer._current_size == buffer._current_size

    def test_load_state_inconsistent_data(
        self, buffer: RandomReplacementBuffer[Any], tmp_path: Path
    ):
        """Test loading state with inconsistent data lengths raises
        ValueError."""
        # Add some data to the buffer
        buffer.add({"state": [1.0], "action": 1, "reward": 0.5})

        # Save state
        save_path = tmp_path / "test_buffer"
        buffer.save_state(save_path)

        # Corrupt one of the saved files
        with open(save_path / "state.pkl", "wb") as f:
            pickle.dump([[1.0], [2.0]], f)  # Different length

        # Create a new buffer and try to load inconsistent state
        new_buffer = RandomReplacementBuffer(
            buffer.collecting_data_names, buffer.max_size
        )
        with pytest.raises(
            ValueError, match="Inconsistent list lengths in loaded data"
        ):
            new_buffer.load_state(save_path)

    def test_save_and_load_state_max_size(
        self, buffer: RandomReplacementBuffer[Any], tmp_path: Path
    ):
        """Test saving and loading the buffer state."""
        # Add some data to the buffer
        buffer.add({"state": [1.0, 0.0], "action": 1, "reward": 0.5})
        buffer.add({"state": [0.0, 1.0], "action": 0, "reward": -0.5})

        # Save state
        save_path = tmp_path / "test_buffer"
        buffer.save_state(save_path)

        # Verify files were created
        for name in buffer.collecting_data_names:
            assert (save_path / f"{name}.pkl").exists()

        # Create a new buffer and load state
        new_buffer = RandomReplacementBuffer(buffer.collecting_data_names, max_size=1)
        new_buffer.load_state(save_path)

        # Check that loaded data matches original
        original_data = buffer.get_data()
        loaded_data = new_buffer.get_data()

        for key in buffer.collecting_data_names:
            assert loaded_data[key] == original_data[key][:1]
        assert new_buffer._current_size == 1

    def test_len(self, buffer: RandomReplacementBuffer[Any]):
        """Test the __len__ method returns the correct buffer size."""
        assert len(buffer) == 0

        buffer.add({"state": [1.0, 0.0], "action": 1, "reward": 0.5})
        assert len(buffer) == 1

        buffer.add({"state": [0.0, 1.0], "action": 0, "reward": -0.5})
        assert len(buffer) == 2
