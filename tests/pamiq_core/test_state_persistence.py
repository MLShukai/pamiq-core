import time
from datetime import datetime
from pathlib import Path

import pytest
from pytest_mock import MockerFixture

from pamiq_core.state_persistence import (
    LatestStatesKeeper,
    PersistentStateMixin,
    StateStore,
    load_pickle,
    save_pickle,
)
from tests.helpers import check_log_message


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
        loaded_data = load_pickle(str(temp_file))
        assert loaded_data == sample_data

    def test_load_pickle_invalid_path(self, tmp_path):
        """Test loading from non-existent path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_pickle(tmp_path / "non_existent_file.pkl")


class TestLatestStatesKeeper:
    """Test suite for LatestStatesKeeper class."""

    @pytest.fixture
    def states_dir(self, tmp_path: Path) -> Path:
        """Create a temporary directory for test state directories."""
        return tmp_path / "states"

    @pytest.fixture
    def setup_test_states(self, states_dir: Path) -> list[Path]:
        """Set up test state directories with different modification times."""
        states_dir.mkdir(exist_ok=True)

        # Create test state directories
        state_dirs = []
        for i in range(5):
            state_path = states_dir / f"test_{i}.state"
            state_path.mkdir()
            # Create a file to make directory non-empty
            (state_path / "test_file.txt").write_text(f"Test content {i}")
            state_dirs.append(state_path)
            # Sleep briefly to ensure different modification times
            time.sleep(0.01)

        return state_dirs

    def test_init_creates_dir_if_not_exists(self, tmp_path: Path) -> None:
        """Test that init creates directory if it doesn't exist."""
        states_dir = tmp_path / "nonexistent"
        LatestStatesKeeper(states_dir)

        assert states_dir.exists()
        assert states_dir.is_dir()

    def test_cleanup_removes_oldest_directories(
        self, states_dir: Path, setup_test_states: list[Path]
    ) -> None:
        """Test that cleanup removes the oldest directories."""
        keeper = LatestStatesKeeper(states_dir, max_keep=3)

        # Get before state
        state_dirs_before = list(states_dir.glob("*.state"))
        assert len(state_dirs_before) == 5

        # Clean up
        removed = keeper.cleanup()

        # Get after state
        state_dirs_after = list(states_dir.glob("*.state"))

        # Verify
        assert len(state_dirs_after) == 3
        assert len(removed) == 2

        # Verify the oldest were removed (first two created)
        assert setup_test_states[0] in removed
        assert setup_test_states[1] in removed

        # Verify the newest remain
        assert setup_test_states[2] in state_dirs_after
        assert setup_test_states[3] in state_dirs_after
        assert setup_test_states[4] in state_dirs_after

    def test_cleanup_with_max_keep_zero(
        self, states_dir: Path, setup_test_states: list[Path]
    ) -> None:
        """Test that cleanup all when max_keep is negative."""
        keeper = LatestStatesKeeper(states_dir, max_keep=0)
        removed = keeper.cleanup()

        # All directories should be removed
        assert len(removed) == 5
        assert len(list(states_dir.glob("*.state"))) == 0

    def test_cleanup_with_max_keep_negative(
        self, states_dir: Path, setup_test_states: list[Path]
    ) -> None:
        """Test that cleanup do nothing when max_keep is negative."""
        keeper = LatestStatesKeeper(states_dir, max_keep=-1)
        removed = keeper.cleanup()

        # All directories should exists.
        assert len(removed) == 0
        assert len(list(states_dir.glob("*.state"))) == 5

    def test_start_and_stop_background_thread(
        self, states_dir: Path, mocker: MockerFixture, caplog
    ) -> None:
        """Test that background thread starts and stops correctly."""
        # Mock the cleanup method to avoid actual execution
        mock_cleanup = mocker.patch.object(LatestStatesKeeper, "cleanup")

        # Create keeper
        keeper = LatestStatesKeeper(states_dir, cleanup_interval=0.001)

        # Start the keeper in background
        keeper.start(background=True)

        time.sleep(0.01)
        # Stop the keeper
        keeper.stop()

        mock_cleanup.assert_called_with()
        assert mock_cleanup.call_count > 1
        check_log_message(
            "Started background state cleanup thread. Max keep: 10", "INFO", caplog
        )

    def test_start_foreground(self, states_dir: Path, mocker: MockerFixture) -> None:
        """Test that start in foreground directly calls cleanup."""
        # Mock the _cleanup method
        mock_cleanup = mocker.patch.object(LatestStatesKeeper, "_cleanup")

        # Create keeper
        keeper = LatestStatesKeeper(states_dir)

        # Start in foreground
        keeper.start(background=False)

        # Verify cleanup was called directly
        mock_cleanup.assert_called_once()
