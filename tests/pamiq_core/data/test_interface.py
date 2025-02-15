import pytest

from pamiq_core.data.buffer import BufferData, DataBuffer, StepData
from pamiq_core.data.interface import DataUser, TimestampingQueuesDict


class TestTimestampingQueuesDict:
    """Test suite for TimestampingQueuesDict class."""

    QUEUE_NAMES = ["state", "action", "reward"]
    MAX_LENGTH = 5
    SAMPLE_DATA = {"state": [1.0, 2.0, 3.0], "action": [0, 1], "reward": 1.0}

    @pytest.fixture
    def queues_dict(self) -> TimestampingQueuesDict:
        """Fixture providing TimestampingQueuesDict instance."""
        return TimestampingQueuesDict(self.QUEUE_NAMES, self.MAX_LENGTH)

    def test_init(self, queues_dict: TimestampingQueuesDict):
        """Test initialization of TimestampingQueuesDict."""
        for queue in queues_dict._queues.values():
            assert queue.maxlen == self.MAX_LENGTH
        assert queues_dict._timestamps.maxlen == self.MAX_LENGTH
        assert len(queues_dict) == 0

    def test_append(self, queues_dict: TimestampingQueuesDict, mocker):
        """Test append method."""
        mock_time = mocker.patch("time.time")
        mock_time.return_value = 123.456

        queues_dict.append(self.SAMPLE_DATA)

        assert len(queues_dict) == 1
        for key, queue in queues_dict._queues.items():
            assert list(queue) == [self.SAMPLE_DATA[key]]
        assert list(queues_dict._timestamps) == [123.456]

    def test_popleft(self, queues_dict: TimestampingQueuesDict, mocker):
        """Test popleft method."""
        mock_time = mocker.patch("time.time")
        mock_time.return_value = 123.456

        queues_dict.append(self.SAMPLE_DATA)
        data, timestamp = queues_dict.popleft()

        assert data == self.SAMPLE_DATA
        assert timestamp == 123.456
        assert len(queues_dict) == 0

    def test_len(self, queues_dict: TimestampingQueuesDict):
        """Test __len__ method."""
        assert len(queues_dict) == 0

        queues_dict.append(self.SAMPLE_DATA)
        assert len(queues_dict) == 1

        queues_dict.append(self.SAMPLE_DATA)
        assert len(queues_dict) == 2

    def test_max_length(self, queues_dict: TimestampingQueuesDict):
        """Test maximum length constraint."""
        for _ in range(self.MAX_LENGTH + 2):
            queues_dict.append(self.SAMPLE_DATA)

        assert len(queues_dict) == self.MAX_LENGTH
        for queue in queues_dict._queues.values():
            assert len(queue) == self.MAX_LENGTH

    def test_empty_popleft(self, queues_dict: TimestampingQueuesDict):
        """Test popleft from empty queues raises IndexError."""
        with pytest.raises(IndexError):
            queues_dict.popleft()


class MockDataBuffer(DataBuffer):
    """Mock buffer implementation for testing."""

    def __init__(self, collecting_data_names: list[str], max_size: int) -> None:
        super().__init__(collecting_data_names, max_size)
        self.data: list[StepData] = []

    def add(self, step_data: StepData) -> None:
        if len(self.data) < self.max_size:
            self.data.append(step_data)

    def get_data(self) -> BufferData:
        return {
            name: [d[name] for d in self.data] for name in self._collecting_data_names
        }


class TestDataUserAndCollector:
    """Integration test suite for DataUser and DataCollector."""

    COLLECTING_NAMES = ["state", "action", "reward"]
    MAX_SIZE = 5
    SAMPLE_DATA = {"state": [1.0, 2.0], "action": 1, "reward": 0.5}

    @pytest.fixture
    def buffer(self) -> MockDataBuffer:
        """Fixture providing mock buffer instance."""
        return MockDataBuffer(self.COLLECTING_NAMES, self.MAX_SIZE)

    @pytest.fixture
    def data_user(self, buffer: MockDataBuffer) -> DataUser[MockDataBuffer]:
        """Fixture providing DataUser instance."""
        return DataUser(buffer)

    def test_basic_collection_and_update(
        self, data_user: DataUser[MockDataBuffer], mocker
    ):
        """Test basic data collection and update process."""
        mock_time = mocker.patch("time.time")
        mock_time.return_value = 100.0

        data_user._collector.collect(self.SAMPLE_DATA)
        assert len(data_user._buffer.data) == 0

        data_user.update()
        assert len(data_user._buffer.data) == 1
        assert data_user._buffer.data[0] == self.SAMPLE_DATA
        assert list(data_user._timestamps) == [100.0]

    def test_timestamp_counting(self, data_user: DataUser[MockDataBuffer], mocker):
        """Test counting of data points added since a timestamp."""
        mock_time = mocker.patch("time.time")

        timestamps = [100.0, 101.0, 102.0, 103.0]
        for t in timestamps:
            mock_time.return_value = t
            data_user._collector.collect(self.SAMPLE_DATA)
            data_user.update()

        assert data_user.count_data_added_since(99.0) == 4
        assert data_user.count_data_added_since(100.5) == 3
        assert data_user.count_data_added_since(102.5) == 1
        assert data_user.count_data_added_since(104.0) == 0

    def test_max_size_constraint(self, data_user: DataUser[MockDataBuffer], mocker):
        """Test maximum size constraint is respected."""
        mock_time = mocker.patch("time.time")

        for i in range(self.MAX_SIZE + 2):
            mock_time.return_value = 100.0 + i
            data_user._collector.collect(self.SAMPLE_DATA)
            data_user.update()

        assert len(data_user._timestamps) == self.MAX_SIZE
        assert len(data_user._buffer.data) == self.MAX_SIZE
