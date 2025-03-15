from typing import override

import pytest
from pytest_mock import MockerFixture

from pamiq_core.data import DataUser, DataUsersDict
from pamiq_core.model import InferenceModel, TrainingModel, TrainingModelsDict
from pamiq_core.state_persistence import PersistentStateMixin
from pamiq_core.trainer import Trainer


class TrainerImpl(Trainer):
    @override
    def on_data_users_dict_attached(self):
        super().on_data_users_dict_attached()
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
        return mocker.Mock(DataUser)

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
    def trainer_attached(
        self,
        trainer: TrainerImpl,
        training_models_dict: TrainingModelsDict,
        data_users_dict: DataUsersDict,
    ) -> TrainerImpl:
        trainer.attach_training_models_dict(training_models_dict=training_models_dict)
        trainer.attach_data_users_dict(data_users_dict=data_users_dict)
        return trainer

    def test_attach_training_models_dict(
        self,
        trainer: TrainerImpl,
        training_models_dict: TrainingModelsDict,
        mock_model,
    ) -> None:
        trainer.attach_training_models_dict(training_models_dict)
        assert trainer.model == mock_model

    def test_attach_data_users_dict(
        self, trainer: TrainerImpl, data_users_dict: DataUsersDict, mock_user
    ) -> None:
        trainer.attach_data_users_dict(data_users_dict)
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
        mock_setup = mocker.spy(trainer_attached, "setup")
        mock_train = mocker.spy(trainer_attached, "train")
        mock_sync_models = mocker.spy(trainer_attached, "sync_models")
        mock_teardown = mocker.spy(trainer_attached, "teardown")

        trainer_attached.run()

        mock_setup.assert_called_once_with()
        mock_train.assert_called_once_with()
        mock_sync_models.assert_called_once_with()
        mock_teardown.assert_called_once_with()
