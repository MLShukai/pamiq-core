from abc import ABC, abstractmethod
from typing import Any

import pytest

from pamiq_core.data import DataUser, DataUsersDict
from pamiq_core.model import (
    TrainingModel,
    TrainingModelsDict,
)

from ._dummy_impls import (
    DummyDataBuffer,
    DummyInferenceModel,
    DummyTrainer,
    DummyTrainingModel,
)


class TestTrainer:
    @pytest.fixture
    def training_models_dict(self) -> TrainingModelsDict:
        return TrainingModelsDict(
            {
                "model_A": DummyTrainingModel(
                    has_inference_model=True,
                    inference_thread_only=True,
                ),
                "model_B": DummyTrainingModel(
                    has_inference_model=True,
                    inference_thread_only=False,
                ),
                "model_C": DummyTrainingModel(
                    has_inference_model=False,
                    inference_thread_only=False,
                ),
            }
        )

    @pytest.fixture
    def data_users_dict(self) -> DataUsersDict:
        return DataUsersDict.from_data_buffers(
            {
                "dummy_visual": DummyDataBuffer(["dummy_image"], 100),
                "dummy_auditory": DummyDataBuffer(["dummy_audio"], 100),
            }
        )

    @pytest.fixture
    def trainer(
        self, training_models_dict: TrainingModelsDict, data_users_dict: DataUsersDict
    ) -> DummyTrainer:
        trainer = DummyTrainer()
        trainer.attach_training_models_dict(training_models_dict=training_models_dict)
        trainer.attach_data_users_dict(data_users_dict=data_users_dict)
        return trainer

    def test_attach_training_models_dict(
        self, trainer: DummyTrainer, training_models_dict: TrainingModelsDict
    ) -> None:
        assert trainer._training_models_dict is training_models_dict

    def test_on_training_models_attached(self, trainer: DummyTrainer) -> None:
        assert trainer.training_models_attached

    def test_attach_data_users_dict(
        self, trainer: DummyTrainer, data_users_dict: DataUsersDict
    ) -> None:
        assert trainer._data_users_dict is data_users_dict

    def test_on_data_users_dict_attached(self, trainer: DummyTrainer) -> None:
        assert trainer.data_users_dict_attached

    def test_inference_models_dict(
        self, trainer: DummyTrainer, training_models_dict: TrainingModelsDict
    ) -> None:
        assert (
            trainer.inference_models_dict is training_models_dict.inference_models_dict
        )

    def test_get_training_model(
        self, trainer: DummyTrainer, training_models_dict: TrainingModelsDict
    ) -> None:
        # It is not need to test about model_A, since model_A is inference thread only.
        assert (
            trainer.get_training_model(name="model_B")
            is training_models_dict["model_B"]
        )
        assert (
            trainer.get_training_model(name="model_C")
            is training_models_dict["model_C"]
        )
        # Check if the names of the retrieved models are correctly kept
        assert trainer._retrieved_model_names == {"model_B", "model_C"}

    def test_get_data_user(
        self, trainer: DummyTrainer, data_users_dict: DataUsersDict
    ) -> None:
        assert (
            trainer.get_data_user(name="dummy_visual")
            is data_users_dict["dummy_visual"]
        )
        assert (
            trainer.get_data_user(name="dummy_auditory")
            is data_users_dict["dummy_auditory"]
        )

    def test_sync_models(
        self, trainer: DummyTrainer, data_users_dict: DataUsersDict
    ) -> None:
        trainer.sync_models()
        # model_A is inference thread only and model_C don't have a inference_model.
        # So sync is performed with only model_B.
        trainer.get_training_model(
            "model_B"
        )._dummy_param == trainer.get_training_model(
            "model_B"
        ).inference_model._dummy_param
