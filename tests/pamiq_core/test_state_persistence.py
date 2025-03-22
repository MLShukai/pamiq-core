from datetime import datetime
from pathlib import Path

import pytest

from pamiq_core.state_persistence import (
    PersistentStateMixin,
    StateStore,
    load_pickle,
    save_pickle,
)


class TestStateStore:
    state_1 = PersistentStateMixin()
    state_2 = PersistentStateMixin()

    def test_register(self, tmp_path):
        store = StateStore(states_dir=tmp_path)
        store.register("state_1", self.state_1)

        assert store._registered_states == {"state_1": self.state_1}

        store.register("state_2", self.state_2)

        assert store._registered_states == {
            "state_1": self.state_1,
            "state_2": self.state_2,
        }

    def test_register_name_already_used_error(self, tmp_path):
        store = StateStore(states_dir=tmp_path)
        store.register("same_name", self.state_1)

        # should raise KeyError:
        with pytest.raises(KeyError):
            store.register("same_name", self.state_2)

    def test_save_state(self, tmp_path, mocker):
        # prepare mock objects
        mock_state_1 = mocker.Mock(spec=PersistentStateMixin)
        mock_state_2 = mocker.Mock(spec=PersistentStateMixin)

        # configure StateStore object
        store = StateStore(states_dir=tmp_path)
        store.register("mock_state_1", mock_state_1)
        store.register("mock_state_2", mock_state_2)

        # Mock store.datetime.now so that tests do not depend on the current time
        fixed_test_time = datetime(2025, 2, 27, 12, 0, 0)

        mock_dt = mocker.Mock(datetime)
        mock_dt.now.return_value = fixed_test_time
        mocker.patch("pamiq_core.state_persistence.datetime", mock_dt)

        state_path = store.save_state()

        assert state_path.exists()  # test: folder is created
        assert state_path == Path(tmp_path / "2025-02-27_12-00-00,000000.state")
        mock_state_1.save_state.assert_called_once_with(state_path / "mock_state_1")
        mock_state_2.save_state.assert_called_once_with(state_path / "mock_state_2")

        # expect error in `Path.mkdir`:
        with pytest.raises(FileExistsError):
            store.save_state()

    def test_load_state(self, tmp_path, mocker):
        # prepare mock objects
        mock_state_1 = mocker.Mock(spec=PersistentStateMixin)
        mock_state_2 = mocker.Mock(spec=PersistentStateMixin)

        # configure StateStore object
        store = StateStore(states_dir=tmp_path)
        store.register("mock_state_1", mock_state_1)
        store.register("mock_state_2", mock_state_2)

        # test for exceptional case
        with pytest.raises(FileNotFoundError):
            store.load_state(tmp_path / "non_existent_folder")

        # test for normal case
        store.load_state(tmp_path)

        mock_state_1.load_state.assert_called_once_with(tmp_path / "mock_state_1")
        mock_state_2.load_state.assert_called_once_with(tmp_path / "mock_state_2")


class TestPickleFunctions:
    @pytest.fixture
    def temp_file(self, tmp_path):
        """Fixture to provide a temporary file path."""
        return tmp_path / "test.pkl"

    @pytest.fixture
    def sample_data(self):
        """Fixture to provide sample data for testing."""
        return {"name": "test", "values": [1, 2, 3], "nested": {"a": 1, "b": 2}}

    def test_save_and_load_pickle(self, temp_file, sample_data):
        """Test saving and loading an object with pickle."""
        save_pickle(sample_data, temp_file)

        # Verify file exists
        assert temp_file.is_file()

        # Load and verify data
        loaded_data = load_pickle(temp_file)
        assert loaded_data == sample_data

    def test_save_pickle_with_string_path(self, temp_file, sample_data):
        """Test saving pickle using a string path."""
        save_pickle(sample_data, str(temp_file))

        # Verify file exists
        assert temp_file.is_file()

        # Load and verify data
        loaded_data = load_pickle(temp_file)
        assert loaded_data == sample_data

    def test_load_pickle_invalid_path(self, tmp_path):
        """Test loading from non-existent path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_pickle(tmp_path / "non_existent_file.pkl")
