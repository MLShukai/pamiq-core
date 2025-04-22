import pytest

from pamiq_core.thread.threads.training import TrainingThread
from pamiq_core.trainer import Trainer, TrainersDict
from tests.helpers import check_log_message


class TestTrainingThread:
    @pytest.fixture
    def mock_trainer1(self, mocker):
        """Fixture providing a mock trainer that returns True from run()."""
        mock = mocker.Mock(spec=Trainer)
        mock.run.return_value = True
        return mock

    @pytest.fixture
    def mock_trainer2(self, mocker):
        """Fixture providing a mock trainer that returns False from run()."""
        mock = mocker.Mock(spec=Trainer)
        mock.run.return_value = False
        return mock

    @pytest.fixture
    def trainers_dict(self, mock_trainer1, mock_trainer2):
        """Fixture providing a TrainersDict with two mock trainers."""
        return TrainersDict({"trainer1": mock_trainer1, "trainer2": mock_trainer2})

    @pytest.fixture
    def training_thread(self, trainers_dict, mocker):
        """Fixture providing a TrainingThread instance with mocked
        controller."""
        thread = TrainingThread(trainers_dict)

        # Mock controller to avoid controller dependency in tests
        mock_controller = mocker.Mock()
        thread.attach_controller(mock_controller)

        return thread

    def test_on_tick_empty_trainers(self, mocker):
        """Test on_tick behavior with empty trainers dict."""
        empty_trainers = TrainersDict({})
        thread = TrainingThread(empty_trainers)

        # Mock controller
        mock_controller = mocker.Mock()
        thread.attach_controller(mock_controller)

        thread.on_tick()
        # No other operations should happen with empty trainers

    def test_on_tick_with_trainers(
        self, training_thread, mock_trainer1, mock_trainer2, caplog
    ):
        """Test on_tick runs trainers and logs results."""
        # First tick should run the first trainer
        training_thread.on_tick()

        # Verify first trainer was run
        mock_trainer1.run.assert_called_once()
        mock_trainer2.run.assert_not_called()

        # Verify log message for first trainer (which returns True)
        check_log_message(
            expected_log_message=r"Trained trainer1 in .+? seconds",
            log_level="INFO",
            caplog=caplog,
        )

        # Reset log and mocks
        caplog.clear()
        mock_trainer1.reset_mock()

        # Second tick should run the second trainer
        training_thread.on_tick()

        # Verify second trainer was run
        mock_trainer1.run.assert_not_called()
        mock_trainer2.run.assert_called_once()

        # Second trainer returns False, so no log message should be present
        assert not any(
            "Trained trainer2" in record.message for record in caplog.records
        )

        # Reset mocks
        mock_trainer2.reset_mock()

        # Third tick should cycle back to the first trainer
        training_thread.on_tick()

        # Verify first trainer was run again
        mock_trainer1.run.assert_called_once()
        mock_trainer2.run.assert_not_called()

    def test_on_paused(self, training_thread, trainers_dict, mocker):
        """Test on_paused propagates to trainers."""
        spy_on_resumed = mocker.spy(trainers_dict, "on_paused")
        training_thread.on_paused()
        # Verify trainers.on_paused was called
        spy_on_resumed.assert_called_once()

    def test_on_resumed(self, training_thread, trainers_dict, mocker):
        """Test on_resumed propagates to trainers."""
        spy_on_resumed = mocker.spy(trainers_dict, "on_resumed")
        training_thread.on_resumed()

        # Verify trainers.on_resumed was called
        spy_on_resumed.assert_called_once()
