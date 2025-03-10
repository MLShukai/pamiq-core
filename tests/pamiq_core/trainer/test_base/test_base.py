from abc import ABC, abstractmethod
from typing import Any

import pytest

from pamiq_core.data import DataUser, DataUsersDict
from pamiq_core.model import (
    InferenceModel,
    InferenceModelsDict,
    TrainingModel,
    TrainingModelsDict,
)

from ._dummy_impls import DummyInferenceModel, DummyTrainer, DummyTrainingModel


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
    def data_user_dict(self) -> DataUsersDict:
        return DataUsersDict()

    @pytest.fixture
    def trainer(self, training_models_dict: TrainingModelsDict) -> DummyTrainer:
        return DummyTrainer()
