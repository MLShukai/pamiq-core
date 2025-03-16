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
    def trainer(
        self,
        training_models_dict: TrainingModelsDict,
        data_users_dict: DataUsersDict,
    ) -> Trainer:
        trainer = DummyTrainer()
        trainer.attach_training_models_dict(training_models_dict=training_models_dict)
        trainer.attach_data_users_dict(data_users_dict=data_users_dict)
        return trainer

    @pytest.fixture
    def trainers_dict(self, trainer: Trainer) -> TrainersDict:
        return TrainersDict({"trainer": trainer})

    def test_get_trainers_list(
        self, trainers_dict: TrainersDict, trainer: Trainer
    ) -> None:
        assert trainers_dict.get_trainers_list() == [trainer]

    def test_get_trainable_trainer(
        self, trainers_dict: TrainersDict, trainer: Trainer
    ) -> None:
        assert trainers_dict.get_trainable_trainer() is trainer

    def test_attach_training_models_dict(
        self, mocker: MockerFixture, trainers_dict: TrainersDict
    ) -> None:
        new_training_models_dict = mocker.Mock(TrainingModelsDict)
        trainers_dict.attach_training_models_dict(new_training_models_dict)
        assert (
            trainers_dict["trainer"]._training_models_dict is new_training_models_dict
        )

    def test_attach_data_users_dict(
        self, mocker: MockerFixture, trainers_dict: TrainersDict
    ) -> None:
        new_data_users_dict = mocker.Mock(DataUsersDict)
        trainers_dict.attach_data_users_dict(new_data_users_dict)
        assert trainers_dict["trainer"]._data_users_dict is new_data_users_dict
