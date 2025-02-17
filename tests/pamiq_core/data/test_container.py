import pytest

from pamiq_core.data.container import DataCollectorsDict, DataUsersDict
from pamiq_core.data.interface import DataCollector, DataUser

from .helpers import MockDataBuffer


class TestDataUsersDictAndCollectorsDict:
    @pytest.fixture
    def buffers(self) -> dict[str, MockDataBuffer]:
        return {
            "main": MockDataBuffer(["state"], 100),
            "sub": MockDataBuffer(["action"], 100),
        }

    @pytest.fixture
    def users_dict(self, buffers: dict[str, MockDataBuffer]) -> DataUsersDict:
        return DataUsersDict.from_data_buffers(buffers)

    @pytest.fixture
    def collectors_dict(self, users_dict: DataUsersDict) -> DataCollectorsDict:
        return users_dict.data_collectors_dict

    # DataUsersDict initialization tests
    def test_from_data_buffers_dict(self, buffers: dict[str, MockDataBuffer]):
        users_dict = DataUsersDict.from_data_buffers(buffers)
        assert len(users_dict) == 2
        assert all(isinstance(v, DataUser) for v in users_dict.values())
        assert set(users_dict.keys()) == {"main", "sub"}

        collectors_dict = users_dict.data_collectors_dict
        assert len(collectors_dict) == 2
        assert all(isinstance(v, DataCollector) for v in collectors_dict.values())
        assert set(collectors_dict.keys()) == {"main", "sub"}

    def test_from_data_buffers_kwargs(self, buffers: dict[str, MockDataBuffer]):
        users_dict = DataUsersDict.from_data_buffers(**buffers)
        assert len(users_dict) == 2
        assert all(isinstance(v, DataUser) for v in users_dict.values())
        assert set(users_dict.keys()) == {"main", "sub"}

        collectors_dict = users_dict.data_collectors_dict
        assert len(collectors_dict) == 2
        assert all(isinstance(v, DataCollector) for v in collectors_dict.values())
        assert set(collectors_dict.keys()) == {"main", "sub"}

    def test_from_data_buffers_mixed(self, buffers: dict[str, MockDataBuffer]):
        buffer3 = MockDataBuffer(["reward"], 100)
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
        user = DataUser(MockDataBuffer(["reward"], 100))
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
