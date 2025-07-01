from pathlib import Path

import pytest

from pamiq_core.data.container import DataCollectorsDict, DataUsersDict
from pamiq_core.data.interface import DataCollector, DataUser
from pamiq_core.state_persistence import PersistentStateMixin

from .helpers import MockDataBuffer


class TestDataUsersDictAndCollectorsDict:
    @pytest.fixture
    def buffers(self) -> dict[str, MockDataBuffer[int]]:
        return {
            "main": MockDataBuffer[int](100),
            "sub": MockDataBuffer[int](100),
        }

    @pytest.fixture
    def users_dict(self, buffers: dict[str, MockDataBuffer[int]]) -> DataUsersDict:
        return DataUsersDict.from_data_buffers(buffers)

    @pytest.fixture
    def collectors_dict(self, users_dict: DataUsersDict) -> DataCollectorsDict:
        return users_dict.data_collectors_dict

    def test_setitem(
        self, users_dict: DataUsersDict, collectors_dict: DataCollectorsDict
    ):
        assert users_dict["main"]._collector is collectors_dict["main"]
        assert users_dict["sub"]._collector is collectors_dict["sub"]

    # DataUsersDict initialization tests
    def test_from_data_buffers_dict(self, buffers: dict[str, MockDataBuffer[int]]):
        users_dict = DataUsersDict.from_data_buffers(buffers)
        assert len(users_dict) == 2
        assert all(isinstance(v, DataUser) for v in users_dict.values())
        assert set(users_dict.keys()) == {"main", "sub"}

        collectors_dict = users_dict.data_collectors_dict
        assert len(collectors_dict) == 2
        assert all(isinstance(v, DataCollector) for v in collectors_dict.values())
        assert set(collectors_dict.keys()) == {"main", "sub"}

    def test_from_data_buffers_kwargs(self, buffers: dict[str, MockDataBuffer[int]]):
        users_dict = DataUsersDict.from_data_buffers(**buffers)
        assert len(users_dict) == 2
        assert all(isinstance(v, DataUser) for v in users_dict.values())
        assert set(users_dict.keys()) == {"main", "sub"}

        collectors_dict = users_dict.data_collectors_dict
        assert len(collectors_dict) == 2
        assert all(isinstance(v, DataCollector) for v in collectors_dict.values())
        assert set(collectors_dict.keys()) == {"main", "sub"}

    def test_from_data_buffers_mixed(self, buffers: dict[str, MockDataBuffer[int]]):
        buffer3 = MockDataBuffer[int](100)
        users_dict = DataUsersDict.from_data_buffers(buffers, extra=buffer3)
        assert len(users_dict) == 3
        assert all(isinstance(v, DataUser) for v in users_dict.values())
        assert set(users_dict.keys()) == {"main", "sub", "extra"}

        collectors_dict = users_dict.data_collectors_dict
        assert len(collectors_dict) == 3
        assert all(isinstance(v, DataCollector) for v in collectors_dict.values())
        assert set(collectors_dict.keys()) == {"main", "sub", "extra"}

    # DataUsersDict and DataCollectorsDict interaction tests
    def test_user_and_collector_synchronization(
        self,
    ):
        users_dict = DataUsersDict()
        user = DataUser(MockDataBuffer[int](100))
        users_dict["test"] = user

        assert "test" in users_dict
        assert isinstance(users_dict["test"], DataUser)
        assert "test" in users_dict.data_collectors_dict
        assert isinstance(users_dict.data_collectors_dict["test"], DataCollector)

    # DataCollectorsDict functionality tests
    def test_collector_acquire_success(self, collectors_dict: DataCollectorsDict):
        collector = collectors_dict.acquire("main")
        assert isinstance(collector, DataCollector)
        assert collector == collectors_dict["main"]

    def test_collector_acquire_already_acquired(
        self, collectors_dict: DataCollectorsDict
    ):
        collectors_dict.acquire("main")
        with pytest.raises(KeyError, match="'main' is already acquired"):
            collectors_dict.acquire("main")

    def test_collector_acquire_not_found(self, collectors_dict: DataCollectorsDict):
        with pytest.raises(KeyError, match="'unknown' not found"):
            collectors_dict.acquire("unknown")

    def test_data_users_dict_inheritance(self):
        """Test if DataUsersDict inherits from PersistentStateMixin."""
        assert issubclass(DataUsersDict, PersistentStateMixin)

    def test_data_users_dict_save_and_loadstate(
        self, users_dict: DataUsersDict, mocker, tmp_path: Path
    ):
        """Test save_state and load_State method of DataUsersDict."""

        # Create spy objects for each user's save_state method
        mock_save_states = {}
        for name, user in users_dict.items():
            mock_save_states[name] = mocker.spy(user, "save_state")

        # Create spy objects for each user's load_state method
        mock_load_states = {}
        for name, user in users_dict.items():
            mock_load_states[name] = mocker.spy(user, "load_state")

        # Call save_state
        test_path = tmp_path / "test"
        users_dict.save_state(test_path)

        # Verify mkdir
        assert test_path.is_dir()

        # Verify each user's save_state was called with correct path
        for name, mock_save_state in mock_save_states.items():
            mock_save_state.assert_called_once_with(test_path / name)

        # Call load_state
        users_dict.load_state(test_path)

        # Verify each user's load_state was called with correct path
        for name, mock_load_state in mock_load_states.items():
            mock_load_state.assert_called_once_with(test_path / name)
