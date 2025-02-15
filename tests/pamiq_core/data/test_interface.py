import pytest

from pamiq_core.data.interface import StepData, TimestampingQueuesDict


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
