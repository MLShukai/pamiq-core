from collections import OrderedDict
from pathlib import Path

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
    def trainable_trainer_1(self, mocker: MockerFixture) -> Trainer:
        mock = mocker.Mock(Trainer)
        mock.is_trainable.return_value = True
        return mock

    @pytest.fixture
    def trainable_trainer_2(self, mocker: MockerFixture) -> Trainer:
        mock = mocker.Mock(Trainer)
        mock.is_trainable.return_value = True
        return mock

    @pytest.fixture
    def untrainable_trainer(self, mocker: MockerFixture) -> Trainer:
        mock = mocker.Mock(Trainer)
        mock.is_trainable.return_value = False
        return mock

    @pytest.fixture
    def trainers(
        self,
        trainable_trainer_1: Trainer,
        trainable_trainer_2: Trainer,
        untrainable_trainer: Trainer,
    ) -> dict[str, Trainer]:
        return {
            "trainable_1": trainable_trainer_1,
            "trainable_2": trainable_trainer_2,
            "untrainable": untrainable_trainer,
        }

    @pytest.fixture
    def trainers_dict(self, trainers: dict[str, Trainer]) -> TrainersDict:
        return TrainersDict(trainers)

    def test_get_trainers_list(
        self, trainers_dict: TrainersDict, trainers: dict[str, Trainer]
    ) -> None:
        assert trainers_dict.get_trainers_list() == list(trainers.values())

    def test_get_trainable_trainer(
        self, trainers_dict: TrainersDict, trainers: Trainer
    ) -> None:
        assert trainers_dict.get_trainable_trainer() == trainers["trainable_1"]
        assert trainers_dict.get_trainable_trainer() == trainers["trainable_2"]
        # Check if it loops back to the first element.
        assert trainers_dict.get_trainable_trainer() == trainers["trainable_1"]

    def test_get_trainable_trainer_with_just_an_untrainable(
        self, untrainable_trainer: Trainer
    ) -> None:
        trainers_dict = TrainersDict({"untrainable": untrainable_trainer})
        assert trainers_dict.get_trainable_trainer() is None

    def test_attach_training_models_dict(
        self,
        mocker: MockerFixture,
        trainers_dict: TrainersDict,
        trainers: dict[str, Trainer],
    ) -> None:
        training_models_dict = mocker.Mock(TrainingModelsDict)
        trainers_dict.attach_training_models_dict(training_models_dict)
        for trainer in trainers.values():
            trainer.attach_training_models_dict.assert_called_once_with(
                training_models_dict
            )

    def test_attach_data_users_dict(
        self,
        mocker: MockerFixture,
        trainers_dict: TrainersDict,
        trainers: dict[str, Trainer],
    ) -> None:
        data_users_dict = mocker.Mock(DataUsersDict)
        trainers_dict.attach_data_users_dict(data_users_dict)
        for trainer in trainers.values():
            trainer.attach_data_users_dict.assert_called_once_with(
                data_users_dict
            )

    def test_save_state(
        self, trainers_dict: TrainersDict, trainers: dict[str, Trainer]
    ) -> None:
        path = Path("test/")
        trainers_dict.save_state(path)
        assert path.is_dir()
        for trainer in trainers.values():
            trainer.save_state.assert_called_once_with(path=path)

    def test_load_state(
        self, trainers_dict: TrainersDict, trainers: dict[str, Trainer]
    ) -> None:
        path = Path("test/")
        trainers_dict.load_state(path=path)
        for trainer in trainers.values():
            trainer.load_state.assert_called_once_with(path=path)
