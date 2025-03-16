from typing import override

import pytest
from pytest_mock import MockerFixture

from pamiq_core.data import DataUser, DataUsersDict
from pamiq_core.model import InferenceModel, TrainingModel, TrainingModelsDict
from pamiq_core.state_persistence import PersistentStateMixin
from pamiq_core.trainer import Trainer, TrainersDict


class DummyTrainer(Trainer):
    @override
    def train(self) -> None:
        pass


class TestTrainersDict:
    def test_trainers_dict_subclasses(self):
        assert issubclass(TrainersDict, PersistentStateMixin)

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
        return DataUsersDict({"data": mock_user})

    @pytest.fixture
    def trainer(self) -> Trainer:
        return DummyTrainer()

    @pytest.fixture
    def trainer_attached(
        self,
        trainer: Trainer,
        training_models_dict: TrainingModelsDict,
        data_users_dict: DataUsersDict,
    ) -> Trainer:
        trainer.attach_training_models_dict(training_models_dict=training_models_dict)
        trainer.attach_data_users_dict(data_users_dict=data_users_dict)
        return trainer

    @pytest.fixture
    def trainers_dict(self) -> TrainersDict:
        return TrainersDict()

    @pytest.fixture
    def trainers_dict_attached(self, trainer_attached: Trainer) -> TrainersDict:
        return TrainersDict({"trainer": trainer_attached})

    def test_get_trainers_list(
        self, trainers_dict_attached: TrainersDict, trainer_attached: Trainer
    ) -> None:
        assert trainers_dict_attached.get_trainers_list() == [trainer_attached]

    # def test_get_trainable_trainer(self) -> None:

    # def test_attach_training_models_dict(self, training_models_dict: TrainingModelsDict) -> None:

    # def test_attach_data_users_dict(self, data_users_dict: DataUsersDict) -> None:
