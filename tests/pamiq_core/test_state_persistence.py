import time
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import override

import pytest
from pytest_mock import MockerFixture

from pamiq_core.state_persistence import (
    LatestStatesKeeper,
    PeriodicSaveCondition,
    PersistentStateMixin,
    StatesKeeper,
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


class TestStatesKeeper:
    """Test suite for StatesKeeper abstract base class."""

    class ImplStatesKeeper(StatesKeeper):
        """Test implementation of StatesKeeper."""

        def __init__(self, paths_to_remove: list[Path] | None = None) -> None:
            super().__init__()
            self._paths_to_remove = paths_to_remove or []
            self.appended_paths: list[Path] = []

        @override
        def append(self, path: Path) -> None:
            """Track appended paths for testing."""
            self.appended_paths.append(path)

        @override
        def select_removal_states(self) -> Iterable[Path]:
            return self._paths_to_remove

    @pytest.fixture
    def states_dir(self, tmp_path: Path) -> Path:
        """Create a temporary directory for test state directories."""
        return tmp_path / "states"

    def test_cleanup_removes_selected_paths(self, states_dir: Path) -> None:
        """Test that cleanup removes directories selected by
        select_removal_states."""
        states_dir.mkdir(exist_ok=True)

        # Create test state directories
        state1 = states_dir / "state1"
        state2 = states_dir / "state2"
        state3 = states_dir / "state3"

        for state_dir in [state1, state2, state3]:
            state_dir.mkdir()
            (state_dir / "test.txt").write_text("test")

        # Create keeper that will remove state1 and state2
        keeper = self.ImplStatesKeeper([state1, state2])

        # Track all states
        keeper.append(state1)
        keeper.append(state2)
        keeper.append(state3)

        # Cleanup should remove state1 and state2
        removed = keeper.cleanup()

        # Verify removal
        assert not state1.exists()
        assert not state2.exists()
        assert state3.exists()

        # Verify the correct paths were removed
        assert set(removed) == {state1, state2}

    def test_cleanup_logs_removed_paths(self, states_dir: Path, caplog) -> None:
        """Test that cleanup logs removed paths."""
        states_dir.mkdir(exist_ok=True)

        state_dir = states_dir / "state_to_remove"
        state_dir.mkdir()

        keeper = self.ImplStatesKeeper([state_dir])
        keeper.cleanup()

        check_log_message(f"Removed: '{state_dir}'", "INFO", caplog)


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
        LatestStatesKeeper(states_dir, 10)

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

    def test_state_tracking_with_append(self, states_dir: Path) -> None:
        """Test that LatestStatesKeeper properly tracks appended states."""
        states_dir.mkdir(exist_ok=True)
        keeper = LatestStatesKeeper(states_dir, max_keep=2)

        # Create and append state paths
        state1 = states_dir / "state1.state"
        state2 = states_dir / "state2.state"
        state3 = states_dir / "state3.state"

        for state in [state1, state2, state3]:
            state.mkdir()
            time.sleep(0.01)  # Ensure different timestamps
            keeper.append(state)

        # Cleanup should remove the oldest (state1)
        removed = keeper.cleanup()

        assert len(removed) == 1
        assert state1 in removed
        assert not state1.exists()
        assert state2.exists()
        assert state3.exists()


class TestPeriodicSaveCondition:
    """Test the PeriodicSaveCondition class."""

    def test_initial_state_returns_false(self) -> None:
        """Test that condition returns False initially."""
        condition = PeriodicSaveCondition(interval=1.0)
        assert condition() is False

    def test_returns_true_after_interval(self) -> None:
        """Test that condition returns True after interval has elapsed."""
        import time

        # Use very short interval for testing
        interval = 0.01
        condition = PeriodicSaveCondition(interval=interval)

        # First call should return False
        assert condition() is False

        # Wait for interval to elapse
        time.sleep(interval * 1.5)

        # Should now return True
        assert condition() is True

    def test_multiple_intervals(self) -> None:
        """Test behavior across multiple intervals."""
        import time

        interval = 0.01
        condition = PeriodicSaveCondition(interval=interval)

        # Initial state
        assert condition() is False

        # First interval
        time.sleep(interval * 1.5)
        assert condition() is True
        assert condition() is False

        # Second interval
        time.sleep(interval * 1.5)
        assert condition() is True
        assert condition() is False
