from pathlib import Path
from typing import override

import pytest
from pytest_mock import MockerFixture

from pamiq_core.data import DataCollector, DataUser, DataUsersDict
from pamiq_core.model import InferenceModel, TrainingModel, TrainingModelsDict
from pamiq_core.state_persistence import PersistentStateMixin
from pamiq_core.thread import ThreadEventMixin
from pamiq_core.trainer import Trainer


class TrainerImpl(Trainer):
    @override
    def on_data_users_attached(self):
        super().on_data_users_attached()
        self.user = self.get_data_user("data")

    @override
    def on_training_models_attached(self):
        super().on_training_models_attached()
        self.model = self.get_training_model("model")

    @override
    def train(self) -> None:
        pass


class TestTrainer:
    def test_trainer_subclasses(self):
        assert issubclass(Trainer, PersistentStateMixin)
        assert issubclass(Trainer, ThreadEventMixin)

    def test_abstract_methods(self):
        assert Trainer.__abstractmethods__ == frozenset({"train"})

    @pytest.fixture
    def mock_model(self, mocker: MockerFixture) -> TrainingModel:
        model = mocker.Mock(TrainingModel)
        model.inference_model = mocker.Mock(InferenceModel)
        model.has_inference_model = True
        model.inference_thread_only = False
        return model

    @pytest.fixture
    def mock_user(self, mocker: MockerFixture) -> DataUser:
        user = mocker.MagicMock(DataUser)
        # Configure mock for trainability tests
        user.__len__.return_value = 100  # Mock buffer size
        user.count_data_added_since.return_value = 10  # Mock new data count
        user._collector = mocker.MagicMock(DataCollector)
        return user

    @pytest.fixture
    def training_models_dict(self, mock_model) -> TrainingModelsDict:
        return TrainingModelsDict({"model": mock_model})

    @pytest.fixture
    def data_users_dict(self, mock_user) -> DataUsersDict:
        return DataUsersDict(
            {
                "data": mock_user,
            }
        )

    @pytest.fixture
    def trainer(self) -> TrainerImpl:
        return TrainerImpl()

    @pytest.fixture
    def conditional_trainer(self) -> TrainerImpl:
        """Fixture providing a trainer with training conditions."""
        return TrainerImpl(
            training_condition_data_user="data",
            min_buffer_size=50,
            min_new_data_count=5,
        )

    @pytest.fixture
    def trainer_attached(
        self,
        trainer: TrainerImpl,
        training_models_dict: TrainingModelsDict,
        data_users_dict: DataUsersDict,
    ) -> TrainerImpl:
        trainer.attach_training_models(training_models=training_models_dict)
        trainer.attach_data_users(data_users=data_users_dict)
        return trainer

    @pytest.fixture
    def conditional_trainer_attached(
        self,
        conditional_trainer: TrainerImpl,
        training_models_dict: TrainingModelsDict,
        data_users_dict: DataUsersDict,
    ) -> TrainerImpl:
        conditional_trainer.attach_training_models(training_models=training_models_dict)
        conditional_trainer.attach_data_users(data_users=data_users_dict)
        return conditional_trainer

    def test_attach_training_models(
        self,
        trainer: TrainerImpl,
        training_models_dict: TrainingModelsDict,
        mock_model,
    ) -> None:
        trainer.attach_training_models(training_models_dict)
        assert trainer.model == mock_model

    def test_attach_data_users(
        self, trainer: TrainerImpl, data_users_dict: DataUsersDict, mock_user
    ) -> None:
        trainer.attach_data_users(data_users_dict)
        assert trainer.user == mock_user

    def test_get_training_model(
        self, trainer_attached: TrainerImpl, mock_model
    ) -> None:
        assert trainer_attached.get_training_model("model") == mock_model

    def test_get_data_user(self, trainer_attached: TrainerImpl, mock_user) -> None:
        assert trainer_attached.get_data_user("data") == mock_user

    def test_is_trainable(self, trainer_attached: TrainerImpl) -> None:
        assert trainer_attached.is_trainable() is True

    def test_sync_models(self, trainer_attached: TrainerImpl, mock_model) -> None:
        trainer_attached.sync_models()
        mock_model.sync.assert_called_once_with()

    def test_run(self, trainer_attached: TrainerImpl, mocker: MockerFixture) -> None:
        """Test that run() executes training and returns True."""
        mock_setup = mocker.spy(trainer_attached, "setup")
        mock_train = mocker.spy(trainer_attached, "train")
        mock_sync_models = mocker.spy(trainer_attached, "sync_models")
        mock_teardown = mocker.spy(trainer_attached, "teardown")

        result = trainer_attached.run()

        assert result is True
        mock_setup.assert_called_once_with()
        mock_train.assert_called_once_with()
        mock_sync_models.assert_called_once_with()
        mock_teardown.assert_called_once_with()

    def test_is_trainable_with_condition_sufficient_data(
        self,
        conditional_trainer_attached: TrainerImpl,
        mock_user,
        mocker: MockerFixture,
    ) -> None:
        """Test is_trainable returns True when conditions are met."""
        # Configure mock for sufficient data
        mock_user.__len__.return_value = 100  # > min_buffer_size (50)
        mock_user.count_data_added_since.return_value = 10  # > min_new_data_count (5)

        # Mock current time for deterministic testing
        mock_time = mocker.patch("pamiq_core.time.time")
        mock_time.return_value = 1000.0

        assert conditional_trainer_attached.is_trainable() is True

        # Verify data user was updated and checked properly
        mock_user.update.assert_called_once()
        mock_user.count_data_added_since.assert_called_once_with(float("-inf"))

        # Verify previous training time was updated
        conditional_trainer_attached.is_trainable()
        mock_user.count_data_added_since.assert_called_with(1000.0)

    def test_is_trainable_with_condition_insufficient_buffer_size(
        self, conditional_trainer_attached: TrainerImpl, mock_user
    ) -> None:
        """Test is_trainable returns False when buffer size is insufficient."""
        # Configure mock for insufficient buffer size
        mock_user.__len__.return_value = 30  # < min_buffer_size (50)
        mock_user.count_data_added_since.return_value = 10  # > min_new_data_count (5)

        assert conditional_trainer_attached.is_trainable() is False

    def test_is_trainable_with_condition_insufficient_new_data(
        self,
        conditional_trainer_attached: TrainerImpl,
        mock_user,
        mocker: MockerFixture,
    ) -> None:
        """Test is_trainable returns False when new data count is
        insufficient."""
        # Set an initial previous training time

        # Configure mock for insufficient new data
        mock_user.__len__.return_value = 100  # > min_buffer_size (50)
        mock_user.count_data_added_since.return_value = 3  # < min_new_data_count (5)

        assert conditional_trainer_attached.is_trainable() is False

    def test_run_skips_when_not_trainable(
        self,
        conditional_trainer_attached: TrainerImpl,
        mock_user,
        mocker: MockerFixture,
    ) -> None:
        """Test that run() skips execution when not trainable."""
        # Configure mock to make trainer not trainable
        mock_user.__len__.return_value = 10  # < min_buffer_size (50)

        # Spy on the methods that should be skipped
        mock_setup = mocker.spy(conditional_trainer_attached, "setup")
        mock_train = mocker.spy(conditional_trainer_attached, "train")

        result = conditional_trainer_attached.run()
        # Verify run returns False when training is skipped
        assert result is False

        # Verify none of the training steps were executed
        mock_setup.assert_not_called()
        mock_train.assert_not_called()

    def test_save_and_load_state(self, trainer: TrainerImpl, tmp_path: Path) -> None:
        """Test save_state and load_state methods."""
        test_path = tmp_path / "trainer"
        state_path = test_path / "previous_training_time"
        trainer.save_state(test_path)

        assert state_path.is_file()
        assert state_path.read_text("utf-8") == "-inf"

        trainer._previous_training_time = 0
        trainer.load_state(test_path)
        assert trainer._previous_training_time == float("-inf")
