import pytest

from pamiq_core.data.buffer import BufferData, DataBuffer, StepData
from pamiq_core.data.interface import DataUser, TimestampingQueuesDict


class TestTimestampingQueuesDict:
    """Test suite for TimestampingQueuesDict class."""

    @pytest.fixture
    def queue_names(self) -> list[str]:
        """Fixture providing sample queue names."""
        return ["state", "action", "reward"]

    @pytest.fixture
    def max_len(self) -> int:
        """Fixture providing sample maximum queue length."""
        return 5

    @pytest.fixture
    def queues_dict(
        self, queue_names: list[str], max_len: int
    ) -> TimestampingQueuesDict:
        """Fixture providing TimestampingQueuesDict instance."""
        return TimestampingQueuesDict(queue_names, max_len)

    @pytest.fixture
    def sample_data(self) -> StepData:
        """Fixture providing sample step data."""
        return {"state": [1.0, 2.0, 3.0], "action": [0, 1], "reward": 1.0}

    def test_init(self, queues_dict: TimestampingQueuesDict, max_len: int):
        """Test initialization of TimestampingQueuesDict."""
        for queue in queues_dict._queues.values():
            assert queue.maxlen == max_len
        assert queues_dict._timestamps.maxlen == max_len
        assert len(queues_dict) == 0

    def test_append(
        self, queues_dict: TimestampingQueuesDict, sample_data: StepData, mocker
    ):
        """Test append method."""
        mock_time = mocker.patch("time.time")
        mock_time.return_value = 123.456

        queues_dict.append(sample_data)

        assert len(queues_dict) == 1
        for key, queue in queues_dict._queues.items():
            assert list(queue) == [sample_data[key]]
        assert list(queues_dict._timestamps) == [123.456]

    def test_popleft(
        self, queues_dict: TimestampingQueuesDict, sample_data: StepData, mocker
    ):
        """Test popleft method."""
        mock_time = mocker.patch("time.time")
        mock_time.return_value = 123.456

        queues_dict.append(sample_data)
        data, timestamp = queues_dict.popleft()

        assert data == sample_data
        assert timestamp == 123.456
        assert len(queues_dict) == 0

    def test_len(self, queues_dict: TimestampingQueuesDict, sample_data: StepData):
        """Test __len__ method."""
        assert len(queues_dict) == 0

        queues_dict.append(sample_data)
        assert len(queues_dict) == 1

        queues_dict.append(sample_data)
        assert len(queues_dict) == 2

    def test_max_length(
        self, queues_dict: TimestampingQueuesDict, sample_data: StepData, max_len: int
    ):
        """Test maximum length constraint."""
        for _ in range(max_len + 2):
            queues_dict.append(sample_data)

        assert len(queues_dict) == max_len
        for queue in queues_dict._queues.values():
            assert len(queue) == max_len

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

    @pytest.fixture
    def collecting_names(self) -> list[str]:
        """Fixture providing sample data field names."""
        return ["state", "action", "reward"]

    @pytest.fixture
    def max_size(self) -> int:
        """Fixture providing sample maximum buffer size."""
        return 5

    @pytest.fixture
    def buffer(self, collecting_names: list[str], max_size: int) -> MockDataBuffer:
        """Fixture providing mock buffer instance."""
        return MockDataBuffer(collecting_names, max_size)

    @pytest.fixture
    def data_user(self, buffer: MockDataBuffer) -> DataUser[MockDataBuffer]:
        """Fixture providing DataUser instance."""
        return DataUser(buffer)

    @pytest.fixture
    def sample_data(self) -> StepData:
        """Fixture providing sample step data."""
        return {"state": [1.0, 2.0], "action": 1, "reward": 0.5}

    def test_basic_collection_and_update(
        self, data_user: DataUser[MockDataBuffer], sample_data: StepData, mocker
    ):
        """Test basic data collection and update process."""
        mock_time = mocker.patch("time.time")
        mock_time.return_value = 100.0

        # Collect data
        data_user._collector.collect(sample_data)
        assert len(data_user._buffer.data) == 0  # Not yet updated

        # Update buffer
        data_user.update()
        assert len(data_user._buffer.data) == 1
        assert data_user._buffer.data[0] == sample_data
        assert list(data_user._timestamps) == [100.0]

    def test_timestamp_counting(
        self, data_user: DataUser[MockDataBuffer], sample_data: StepData, mocker
    ):
        """Test counting of data points added since a timestamp."""
        mock_time = mocker.patch("time.time")

        # Add data at different timestamps
        timestamps = [100.0, 101.0, 102.0, 103.0]
        for t in timestamps:
            mock_time.return_value = t
            data_user._collector.collect(sample_data)
            data_user.update()

        # Test counting
        assert data_user.count_data_added_since(99.0) == 4
        assert data_user.count_data_added_since(100.5) == 3
        assert data_user.count_data_added_since(102.5) == 1
        assert data_user.count_data_added_since(104.0) == 0

    def test_max_size_constraint(
        self,
        data_user: DataUser[MockDataBuffer],
        sample_data: StepData,
        max_size: int,
        mocker,
    ):
        """Test maximum size constraint is respected."""
        mock_time = mocker.patch("time.time")

        # Add more data than max_size
        for i in range(max_size + 2):
            mock_time.return_value = 100.0 + i
            data_user._collector.collect(sample_data)
            data_user.update()

        assert len(data_user._timestamps) == max_size
        assert len(data_user._buffer.data) == max_size
