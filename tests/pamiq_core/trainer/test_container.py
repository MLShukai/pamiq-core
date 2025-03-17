from typing import override

import pytest
from pytest_mock import MockerFixture

from pamiq_core.data import DataUser, DataUsersDict
from pamiq_core.model import InferenceModel, TrainingModel, TrainingModelsDict
from pamiq_core.state_persistence import PersistentStateMixin
from pamiq_core.trainer import Trainer, TrainersDict



class TestTrainersDict:
    def test_trainers_dict_subclasses(self):
        assert issubclass(TrainersDict, PersistentStateMixin)

    @pytest.fixture
    def trainer(self, mocker: MockerFixture) -> Trainer:
        trainer = DummyTrainer()
        trainer.attach_training_models_dict(
            training_models_dict=mocker.Mock(TrainingModelsDict)
        )
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
        training_models_dict = mocker.Mock(TrainingModelsDict)
        trainers_dict.attach_training_models_dict(training_models_dict)
        for mock_trainer in [...]:
            mock_trainer.attach_training_models_dict.assert_called_once_with(training_models_dict)

    def test_attach_data_users_dict(
        self, mocker: MockerFixture, trainers_dict: TrainersDict
    ) -> None:
        new_data_users_dict = mocker.Mock(DataUsersDict)
        trainers_dict.attach_data_users_dict(new_data_users_dict)
        assert trainers_dict["trainer"]._data_users_dict is new_data_users_dict
