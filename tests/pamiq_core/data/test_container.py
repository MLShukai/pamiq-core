import pytest

from pamiq_core.data.container import DataCollectorsDict, DataUsersDict
from pamiq_core.data.interface import DataCollector, DataUser

from .helpers import MockDataBuffer


class TestDataUsersDict:
    @pytest.fixture
    def buffers(self) -> dict[str, MockDataBuffer]:
        return {
            "main": MockDataBuffer(["state"], 100),
            "sub": MockDataBuffer(["action"], 100),
        }

    def test_from_data_buffers_dict(self, buffers: dict[str, MockDataBuffer]):
        users_dict = DataUsersDict.from_data_buffers(buffers)
        assert len(users_dict) == 2
        assert all(isinstance(v, DataUser) for v in users_dict.values())
        assert set(users_dict.keys()) == {"main", "sub"}

    def test_from_data_buffers_kwargs(self, buffers: dict[str, MockDataBuffer]):
        users_dict = DataUsersDict.from_data_buffers(**buffers)
        assert len(users_dict) == 2
        assert all(isinstance(v, DataUser) for v in users_dict.values())
        assert set(users_dict.keys()) == {"main", "sub"}

    def test_from_data_buffers_mixed(self, buffers: dict[str, MockDataBuffer]):
        buffer3 = MockDataBuffer(["reward"], 100)
        users_dict = DataUsersDict.from_data_buffers(buffers, extra=buffer3)
        assert len(users_dict) == 3
        assert all(isinstance(v, DataUser) for v in users_dict.values())
        assert set(users_dict.keys()) == {"main", "sub", "extra"}

    def test_get_data_collectors(self, buffers: dict[str, MockDataBuffer]):
        users_dict = DataUsersDict.from_data_buffers(buffers)
        collectors_dict = users_dict.get_data_collectors()

        assert isinstance(collectors_dict, DataCollectorsDict)
        assert len(collectors_dict) == 2
        assert all(isinstance(v, DataCollector) for v in collectors_dict.values())
        assert set(collectors_dict.keys()) == {"main", "sub"}


class TestDataCollectorsDict:
    @pytest.fixture
    def collectors_dict(self) -> DataCollectorsDict:
        buffer1 = MockDataBuffer(["state"], 100)
        buffer2 = MockDataBuffer(["action"], 100)
        users_dict = DataUsersDict.from_data_buffers(main=buffer1, sub=buffer2)
        return users_dict.get_data_collectors()

    def test_acquire_success(self, collectors_dict: DataCollectorsDict):
        collector = collectors_dict.acquire("main")
        assert isinstance(collector, DataCollector)
        assert collector == collectors_dict["main"]

    def test_acquire_already_acquired(self, collectors_dict: DataCollectorsDict):
        collectors_dict.acquire("main")
        with pytest.raises(KeyError, match="'main' is already acquired"):
            collectors_dict.acquire("main")

    def test_acquire_not_found(self, collectors_dict: DataCollectorsDict):
        with pytest.raises(KeyError, match="'unknown' not found"):
            collectors_dict.acquire("unknown")
