import shutil
from datetime import datetime, tzinfo
from pathlib import Path

import pytest

from pamiq_core.state_persistence import PersistentStateMixin, StateStore


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
        parent_mock = mocker.Mock()
        mock_state_1 = mocker.Mock(spec=PersistentStateMixin)
        mock_state_2 = mocker.Mock(spec=PersistentStateMixin)
        parent_mock.attach_mock(mock_state_1, "mock_state_1")
        parent_mock.attach_mock(mock_state_2, "mock_state_2")

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
        assert parent_mock.mock_calls == [
            mocker.call.mock_state_1.save_state(state_path / "mock_state_1"),
            mocker.call.mock_state_2.save_state(state_path / "mock_state_2"),
        ]  # test: `save_state()` is called for each registered state

        # expect error in `Path.mkdir`:
        with pytest.raises(FileExistsError):
            store.save_state()

    def test_load_state(self, tmp_path, mocker):
        # prepare mock objects
        parent_mock = mocker.Mock()
        mock_state_1 = mocker.Mock(spec=PersistentStateMixin)
        mock_state_2 = mocker.Mock(spec=PersistentStateMixin)
        parent_mock.attach_mock(mock_state_1, "mock_state_1")
        parent_mock.attach_mock(mock_state_2, "mock_state_2")

        # configure StateStore object
        store = StateStore(states_dir=tmp_path)
        store.register("mock_state_1", mock_state_1)
        store.register("mock_state_2", mock_state_2)

        # test for exceptional case
        mocker.patch("pamiq_core.state_persistence.Path.exists", return_value=False)
        with pytest.raises(FileNotFoundError):
            store.load_state(tmp_path / "non_existent_folder")

        # test for normal case
        mocker.patch("pamiq_core.state_persistence.Path.exists", return_value=True)
        store.load_state(tmp_path)
        assert parent_mock.mock_calls == [
            mocker.call.mock_state_1.load_state(tmp_path / "mock_state_1"),
            mocker.call.mock_state_2.load_state(tmp_path / "mock_state_2"),
        ]  # test: `load_state()`` is called for each registered state


