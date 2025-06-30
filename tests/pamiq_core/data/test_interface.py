from pathlib import Path

import pytest

from pamiq_core.data.interface import DataUser, TimestampingQueue
from pamiq_core.state_persistence import PersistentStateMixin

from .helpers import MockDataBuffer


class TestTimestampingQueue:
    """Test suite for TimestampingQueue class."""

    MAX_LENGTH = 5
    SAMPLE_DATA = 42

    @pytest.fixture
    def queue(self) -> TimestampingQueue[int]:
        """Fixture providing TimestampingQueue instance."""
        return TimestampingQueue[int](self.MAX_LENGTH)

    def test_append_and_popleft(self, queue: TimestampingQueue[int], mocker):
        """Test append and popleft methods using only public interfaces."""
        mock_time = mocker.patch("pamiq_core.time.time")
        mock_time.return_value = 123.456

        # Verify empty initially
        assert len(queue) == 0

        # Test append
        queue.append(self.SAMPLE_DATA)
        assert len(queue) == 1

        # Test popleft - verify we get the expected data and timestamp
        data, timestamp = queue.popleft()
        assert data == self.SAMPLE_DATA
        assert timestamp == 123.456
        assert len(queue) == 0

    def test_append_multiple_items(self, queue: TimestampingQueue[int]):
        """Test appending multiple items and length tracking."""
        # Append multiple items
        for i in range(3):
            queue.append(i)
            assert len(queue) == i + 1

        # Verify popleft returns them in order
        for i in range(3):
            data, _ = queue.popleft()
            assert data == i

        # Verify empty after popping all items
        assert len(queue) == 0

    def test_max_length_constraint(self, queue: TimestampingQueue[int]):
        """Test maximum length constraint using only public interfaces."""
        # Fill beyond max capacity
        for i in range(self.MAX_LENGTH + 2):
            queue.append(i)

        # Verify length is capped at MAX_LENGTH
        assert len(queue) == self.MAX_LENGTH

        # Verify we can pop exactly MAX_LENGTH items
        for _ in range(self.MAX_LENGTH):
            queue.popleft()

        # Verify empty after popping all items
        assert len(queue) == 0

    def test_empty_popleft(self, queue: TimestampingQueue[int]):
        """Test popleft from empty queue raises IndexError."""
        with pytest.raises(IndexError):
            queue.popleft()


class TestDataUserAndCollector:
    """Integration test suite for DataUser and DataCollector."""

    MAX_SIZE = 5
    SAMPLE_DATA = 42

    @pytest.fixture
    def buffer(self) -> MockDataBuffer[int]:
        """Fixture providing mock buffer instance."""
        return MockDataBuffer[int](self.MAX_SIZE)

    @pytest.fixture
    def data_user(self, buffer: MockDataBuffer[int]) -> DataUser[list[int]]:
        """Fixture providing DataUser instance."""
        return DataUser(buffer)

    def test_basic_collection_and_update(
        self, data_user: DataUser[list[int]], buffer: MockDataBuffer[int], mocker
    ):
        """Test the collection and update workflow between collector and
        user."""
        mock_time = mocker.patch("pamiq_core.time.time")
        mock_time.return_value = 100.0

        # Access collector through the intended interface for this pair
        collector = data_user._collector
        collector.collect(self.SAMPLE_DATA)

        # Verify data is in collector's queue but not yet in buffer
        assert len(buffer.data) == 0

        # Update to move data from collector to buffer
        data_user.update()

        # Verify data was transferred to buffer
        assert len(buffer.data) == 1
        assert buffer.data[0] == self.SAMPLE_DATA

    def test_timestamp_counting(self, data_user: DataUser[list[int]], mocker):
        """Test counting of data points added since a timestamp."""
        mock_time = mocker.patch("pamiq_core.time.time")
        collector = data_user._collector

        timestamps = [100.0, 101.0, 102.0, 103.0]
        for i, t in enumerate(timestamps):
            mock_time.return_value = t
            collector.collect(i)
            data_user.update()

        # Test the public interface for timestamp-based counting
        assert data_user.count_data_added_since(99.0) == 4
        assert data_user.count_data_added_since(100.5) == 3
        assert data_user.count_data_added_since(102.5) == 1
        assert data_user.count_data_added_since(104.0) == 0

    def test_max_size_constraint(
        self, data_user: DataUser[list[int]], buffer: MockDataBuffer[int], mocker
    ):
        """Test maximum size constraint is respected."""
        mock_time = mocker.patch("pamiq_core.time.time")
        collector = data_user._collector

        # Add more data than MAX_SIZE
        for i in range(self.MAX_SIZE + 2):
            mock_time.return_value = 100.0 + i
            collector.collect(i)
            data_user.update()

        # Verify size is capped at MAX_SIZE
        assert len(buffer.data) == self.MAX_SIZE

    def test_public_interface_for_data_access(
        self, data_user: DataUser[list[int]], mocker
    ):
        """Test the public interfaces for accessing collected data."""
        mock_time = mocker.patch("pamiq_core.time.time")
        mock_time.return_value = 100.0

        # Collect and update data
        data_user._collector.collect(self.SAMPLE_DATA)
        data_user.update()

        # Test public get_data() interface
        buffer_data = data_user.get_data()
        assert isinstance(buffer_data, list)
        assert len(buffer_data) == 1
        assert buffer_data[0] == self.SAMPLE_DATA

        # Test public len() interface
        assert len(data_user) == 1

    def test_implict_update(self, data_user: DataUser[list[int]], mocker):
        """Test the public interfaces for accessing collected data."""
        mock_time = mocker.patch("pamiq_core.time.time")
        mock_time.return_value = 100.0

        # Collect and update data
        data_user._collector.collect(self.SAMPLE_DATA)

        # Test public get_data() interface
        buffer_data = data_user.get_data()
        assert isinstance(buffer_data, list)
        assert len(buffer_data) == 1
        assert buffer_data[0] == self.SAMPLE_DATA

    def test_save_and_load_state(
        self,
        data_user: DataUser[list[int]],
        mocker,
        buffer: MockDataBuffer[int],
        tmp_path: Path,
    ):
        """Test save_state and load_state methods of DataUser."""
        # Mock the buffer's save_state and load_state methods
        mock_buffer_save = mocker.spy(buffer, "save_state")
        mock_buffer_load = mocker.spy(buffer, "load_state")

        # Mock update method to verify it's called during save_state
        mock_update = mocker.spy(data_user, "update")

        # Call save_state
        test_path = tmp_path / "data"
        data_user.save_state(test_path)

        # Verify mkdir
        assert test_path.is_dir()
        # Verify update was called
        mock_update.assert_called_once()
        # Verify buffer.save_state was called with the same path
        mock_buffer_save.assert_called_once_with(test_path / "buffer")
        # Verify save timestamps
        assert (test_path / "timestamps.pkl").is_file()

        # Call load_state
        prev_timestamps = data_user._timestamps
        data_user.load_state(test_path)

        # Verify buffer.load_state was called with the same path
        mock_buffer_load.assert_called_once_with(test_path / "buffer")

        # Verify loading timestamps
        assert data_user._timestamps is not prev_timestamps
        assert data_user._timestamps == prev_timestamps

    def test_persistence_inheritance(self):
        """Test DataUser inherits from PersistentStateMixin for state
        persistence."""
        assert issubclass(DataUser, PersistentStateMixin)
