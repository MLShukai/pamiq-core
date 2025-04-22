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

    def test_attach_training_models(
        self,
        mocker: MockerFixture,
        trainers_dict: TrainersDict,
        trainers: dict[str, Trainer],
    ) -> None:
        training_models_dict = mocker.Mock(TrainingModelsDict)
        trainers_dict.attach_training_models(training_models_dict)
        for trainer in trainers.values():
            trainer.attach_training_models.assert_called_once_with(training_models_dict)

    def test_attach_data_users(
        self,
        mocker: MockerFixture,
        trainers_dict: TrainersDict,
        trainers: dict[str, Trainer],
    ) -> None:
        data_users_dict = mocker.Mock(DataUsersDict)
        trainers_dict.attach_data_users(data_users_dict)
        for trainer in trainers.values():
            trainer.attach_data_users.assert_called_once_with(data_users_dict)

    def test_save_state(
        self, trainers_dict: TrainersDict, trainers: dict[str, Trainer], tmp_path: Path
    ) -> None:
        path = tmp_path / "test/"
        trainers_dict.save_state(path)
        assert path.is_dir()
        for name, trainer in trainers.items():
            trainer.save_state.assert_called_once_with(path / name)

    def test_load_state(
        self, trainers_dict: TrainersDict, trainers: dict[str, Trainer], tmp_path: Path
    ) -> None:
        path = tmp_path / "test/"
        trainers_dict.load_state(path)
        for name, trainer in trainers.items():
            trainer.load_state.assert_called_once_with(path / name)

    def test_on_paused(
        self,
        trainers_dict: TrainersDict,
        trainers: dict[str, Trainer],
    ) -> None:
        """Test that on_paused calls on_paused on all trainers."""
        # Call on_paused
        trainers_dict.on_paused()

        # Verify each trainer's on_paused was called
        for trainer in trainers.values():
            trainer.on_paused.assert_called_once_with()

    def test_on_resumed(
        self,
        trainers_dict: TrainersDict,
        trainers: dict[str, Trainer],
    ) -> None:
        """Test that on_resumed calls on_resumed on all trainers."""
        # Call on_resumed
        trainers_dict.on_resumed()

        # Verify each trainer's on_resumed was called
        for trainer in trainers.values():
            trainer.on_resumed.assert_called_once_with()
