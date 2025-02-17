import dataclasses
import random
from collections import deque
from collections.abc import Iterable
from typing import Any, override

import pytest

from pamiq_core.model import (
    InferenceModel,
    InferenceModelsDict,
    TrainingModel,
    TrainingModelsDict,
)


class DummyInferenceModel(InferenceModel):
    def __init__(self, model_id: int) -> None:
        self.model_id = model_id

    @override
    def infer(self) -> None:
        pass


class DummyTrainingModel(TrainingModel):
    def __init__(
        self, model_id: int, has_inference_model: bool, inference_only: bool
    ) -> None:
        super().__init__(
            has_inference_model=has_inference_model, inference_only=inference_only
        )
        self.model_id = model_id

    @override
    def _create_inference_model(self) -> InferenceModel:
        return DummyInferenceModel(self.model_id)

    @override
    def forward(self) -> None:
        pass


@dataclasses.dataclass
class DataclassForTest:
    has_inference_model: bool
    inference_only: bool
    key: str
    model_id: int


class TestTrainingModelsDict:
    @pytest.fixture
    def dataclasses_for_test(self) -> list[DataclassForTest]:
        return [
            DataclassForTest(
                has_inference_model=True,
                inference_only=True,
                key="test_model_99999",
                model_id=99999,
            ),
            DataclassForTest(
                has_inference_model=True,
                inference_only=False,
                key="test_model_2048",
                model_id=4096,
            ),
            DataclassForTest(
                has_inference_model=False,
                inference_only=False,
                key="test_model_123456789",
                model_id=3333,
            ),
        ]

    @pytest.fixture
    def training_models_dict(
        self, dataclasses_for_test: list[DataclassForTest]
    ) -> TrainingModelsDict:
        training_models_dict = TrainingModelsDict()
        for dataclass in dataclasses_for_test:
            training_models_dict[dataclass.key] = DummyTrainingModel(
                model_id=dataclass.model_id,
                has_inference_model=dataclass.has_inference_model,
                inference_only=dataclass.inference_only,
            )
        return training_models_dict

    def test_inference_models_dict(
        self,
        training_models_dict: TrainingModelsDict,
        dataclasses_for_test: list[DataclassForTest],
    ) -> None:
        inference_models_dict = training_models_dict.inference_models_dict
        # For each key, check if the correct inference model is registered or not.
        for dataclass in dataclasses_for_test:
            if dataclass.has_inference_model:
                inference_model = inference_models_dict[dataclass.key]
                assert isinstance(inference_model, DummyInferenceModel)
                assert inference_model.model_id == dataclass.model_id
        # Check if all keys are correct.
        expected_keys: list[str] = [
            dataclass.key
            for dataclass in dataclasses_for_test
            if dataclass.has_inference_model
        ]
        assert list(inference_models_dict.keys()) == expected_keys

    def test_getitem(
        self,
        training_models_dict: TrainingModelsDict,
        dataclasses_for_test: list[DataclassForTest],
    ) -> None:
        for dataclass in dataclasses_for_test:
            if dataclass.inference_only:
                with pytest.raises(KeyError):
                    training_models_dict[dataclass.key]
            else:
                # For each key, check if the correct training model is registered or not.
                training_model = training_models_dict[dataclass.key]
                assert isinstance(training_model, DummyTrainingModel)
                assert training_model.model_id == dataclass.model_id
