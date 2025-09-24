import pickle
import random
from pathlib import Path
from typing import Any

import pytest

from pamiq_core.data.impls.random_replacement_buffer import (
    DictRandomReplacementBuffer,
    RandomReplacementBuffer,
)


class TestRandomReplacementBuffer:
    """Test suite for RandomReplacementBuffer class."""

    @pytest.fixture
    def buffer(self) -> RandomReplacementBuffer[int]:
        """Fixture providing a standard RandomReplacementBuffer for tests."""
        return RandomReplacementBuffer(5)

    def test_init(self):
        """Test RandomReplacementBuffer initialization with various
        parameters."""
        # Test with standard parameters
        max_size = 10
        buffer = RandomReplacementBuffer[int](max_size)

        assert buffer.max_size == max_size
        assert buffer._replace_probability == 1.0
        assert len(buffer) == 0
        assert not buffer.is_full

        # Test with custom replace probability
        buffer = RandomReplacementBuffer[int](max_size, replace_probability=0.5)
        assert buffer._replace_probability == 0.5

        # Test with invalid replace probability
        with pytest.raises(ValueError):
            RandomReplacementBuffer[int](max_size, replace_probability=-0.1)

        with pytest.raises(ValueError):
            RandomReplacementBuffer[int](max_size, replace_probability=1.1)

    def test_init_with_expected_survival_length(self):
        """Test initialization with expected_survival_length parameter."""
        max_size = 10

        # Test with expected_survival_length only
        buffer = RandomReplacementBuffer[int](max_size, expected_survival_length=20)

        # Should compute probability automatically
        assert 0.0 <= buffer._replace_probability <= 1.0
        assert buffer.max_size == max_size

    def test_init_with_both_parameters_raises_error(self):
        """Test that specifying both replace_probability and
        expected_survival_length raises ValueError."""
        max_size = 10

        with pytest.raises(
            ValueError,
            match="Cannot specify both replace_probability and expected_survival_length",
        ):
            RandomReplacementBuffer[int](
                max_size,
                replace_probability=0.5,
                expected_survival_length=20,
            )

    def test_init_with_none_parameters(self):
        """Test initialization when both parameters are None (should default to
        1.0)."""
        max_size = 10

        buffer = RandomReplacementBuffer[int](max_size)
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

    def test_add_and_get_data(self, buffer: RandomReplacementBuffer[int]):
        """Test adding data to the buffer and retrieving it."""
        # Add data
        buffer.add(1)

        # Check data retrieval after adding one sample
        data = buffer.get_data()
        assert data == [1]
        assert len(buffer) == 1

        # Add another sample
        buffer.add(2)

        # Check data retrieval after adding second sample
        data = buffer.get_data()
        assert data == [1, 2]
        assert len(buffer) == 2

    def test_is_full_property(self):
        """Test the is_full property correctly reports buffer fullness."""
        buffer = RandomReplacementBuffer[int](3)

        assert not buffer.is_full

        # Fill the buffer
        for i in range(3):
            buffer.add(i)

        assert buffer.is_full

    def test_replacement_when_full(self, monkeypatch):
        """Test the random replacement behavior when buffer is full."""
        # Create a small buffer
        buffer = RandomReplacementBuffer[str](2)

        # Fill the buffer
        buffer.add("A")
        buffer.add("B")
        assert buffer.is_full

        # Mock random functions to get deterministic behavior
        monkeypatch.setattr(random, "random", lambda: 0.0)  # Always below probability
        monkeypatch.setattr(
            random, "randint", lambda a, b: 0
        )  # Always replace first element

        # Add another item
        buffer.add("C")

        # Check that first element was replaced
        assert buffer.get_data() == ["C", "B"]

    def test_skip_replacement(self, monkeypatch):
        """Test that replacement is skipped based on probability."""
        # Create a buffer with low replacement probability
        buffer = RandomReplacementBuffer[str](2, replace_probability=0.3)

        # Fill the buffer
        buffer.add("A")
        buffer.add("B")
        assert buffer.is_full

        # Mock random to always be above the replacement probability
        monkeypatch.setattr(random, "random", lambda: 0.9)

        # Try to add another item
        buffer.add("C")

        # Check that no replacement occurred
        assert buffer.get_data() == ["A", "B"]

    def test_get_data_returns_copy(self, buffer: RandomReplacementBuffer[int]):
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

    def test_save_and_load_state(
        self, buffer: RandomReplacementBuffer[int], tmp_path: Path
    ):
        """Test saving and loading the buffer state."""
        # Add some data to the buffer
        buffer.add(1)
        buffer.add(2)

        # Save state
        save_path = tmp_path / "test_buffer"
        buffer.save_state(save_path)

        # Verify file was created with .pkl extension
        assert save_path.with_suffix(".pkl").exists()

        # Create a new buffer and load state
        new_buffer = RandomReplacementBuffer[int](buffer.max_size)
        new_buffer.load_state(save_path)

        # Check that loaded data matches original
        original_data = buffer.get_data()
        loaded_data = new_buffer.get_data()

        assert loaded_data == original_data
        assert len(new_buffer) == len(buffer)

    def test_save_and_load_state_max_size(
        self, buffer: RandomReplacementBuffer[int], tmp_path: Path
    ):
        """Test saving and loading the buffer state with smaller max_size."""
        # Add some data to the buffer
        buffer.add(1)
        buffer.add(2)

        # Save state
        save_path = tmp_path / "test_buffer"
        buffer.save_state(save_path)

        # Verify file was created with .pkl extension
        assert save_path.with_suffix(".pkl").exists()

        # Create a new buffer with smaller max_size and load state
        new_buffer = RandomReplacementBuffer[int](max_size=1)
        new_buffer.load_state(save_path)

        # Check that loaded data is truncated to new max_size
        loaded_data = new_buffer.get_data()
        assert loaded_data == [1]
        assert len(new_buffer) == 1

    def test_len(self, buffer: RandomReplacementBuffer[int]):
        """Test the __len__ method returns the correct buffer size."""
        assert len(buffer) == 0

        buffer.add(1)
        assert len(buffer) == 1

        buffer.add(2)
        assert len(buffer) == 2


class TestDictRandomReplacementBuffer:
    """Test suite for DictRandomReplacementBuffer class."""

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
        buffer = DictRandomReplacementBuffer[int](["a", "b"], max_size=max_size)
        assert buffer.max_size == expected
        assert len(buffer) == 0

    def test_init_with_empty_keys(self):
        """Test initialization with empty keys is allowed."""
        buffer = DictRandomReplacementBuffer[int]([], max_size=10)
        assert len(buffer) == 0
        assert buffer.get_data() == {}

    def test_init_with_both_probability_params_raises(self):
        """Test that specifying both probability parameters raises error."""
        with pytest.raises(ValueError, match="Cannot specify both"):
            DictRandomReplacementBuffer[int](
                ["a"], max_size=10, replace_probability=0.5, expected_survival_length=20
            )

    @pytest.mark.parametrize(
        "keys,data",
        [
            (["a", "b"], {"a": 1, "b": 2}),
            ([], {}),  # Empty keys and data
        ],
    )
    def test_add_valid_data(self, keys, data):
        """Test adding valid data to buffer."""
        buffer = DictRandomReplacementBuffer[int](keys, max_size=10)
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
        buffer = DictRandomReplacementBuffer[int](keys, max_size=10)
        with pytest.raises(ValueError, match="Data keys.*do not match expected keys"):
            buffer.add(data)

    def test_get_data_structure(self):
        """Test get_data returns correct structure."""
        buffer = DictRandomReplacementBuffer[int](["x", "y"], max_size=3)

        # Empty buffer
        assert buffer.get_data() == {"x": [], "y": []}

        # Add data
        buffer.add({"x": 1, "y": 10})
        buffer.add({"x": 2, "y": 20})

        result = buffer.get_data()
        assert result == {"x": [1, 2], "y": [10, 20]}

    def test_replacement_maintains_size(self):
        """Test buffer size remains constant when full."""
        buffer = DictRandomReplacementBuffer[int](
            ["a"], max_size=2, replace_probability=1.0
        )

        # Fill buffer
        buffer.add({"a": 1})
        buffer.add({"a": 2})
        assert len(buffer) == 2

        # Add more - should replace
        buffer.add({"a": 3})
        assert len(buffer) == 2
        assert len(buffer.get_data()["a"]) == 2

    def test_save_and_load_state(self, tmp_path):
        """Test state persistence."""
        keys = ["a", "b"]
        buffer = DictRandomReplacementBuffer[int](keys, max_size=10)

        # Add data
        buffer.add({"a": 1, "b": 2})
        buffer.add({"a": 3, "b": 4})

        # Save
        path = tmp_path / "state"
        buffer.save_state(path)

        # Load into new buffer
        new_buffer = DictRandomReplacementBuffer[int](keys, max_size=10)
        new_buffer.load_state(path)

        assert new_buffer.get_data() == buffer.get_data()
        assert len(new_buffer) == len(buffer)

    def test_add_creates_shallow_copy(self):
        """Test that add creates a shallow copy of the input dictionary."""
        buffer = DictRandomReplacementBuffer[int](["x", "y"], max_size=5)

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
        buffer = DictRandomReplacementBuffer[int](keys, max_size=10)

        # Check keys property returns same keys
        assert buffer.keys == keys

        # Check keys property returns a copy
        returned_keys = buffer.keys
        returned_keys.add("d")
        assert buffer.keys == keys  # Should be unchanged
