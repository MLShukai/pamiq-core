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


class TestTrainingModelsDict:
    def test_inference_models_dict(self) -> None:
        # Make training_models_dict.
        training_models_dict = TrainingModelsDict({})
        expected_keys = []
        expected_ids = []
        for i in range(10):
            key = f"model_{i}"
            has_inference_model, inference_only = random.choice(
                [(True, True), (True, False), (False, False)]
            )
            training_models_dict[key] = DummyTrainingModel(
                model_id=i,
                has_inference_model=has_inference_model,
                inference_only=inference_only,
            )
            if has_inference_model and not inference_only:
                expected_keys.append(key)
                expected_ids.append(i)
        # Get inference_models_dict.
        inference_models_dict = training_models_dict.inference_models_dict
        # Check if all keys are correct.
        assert list(inference_models_dict.keys()) == expected_keys
        # For each key, check if the correct inference model is registered or not.
        for expected_key, expected_id in zip(expected_keys, expected_ids):
            inference_model = inference_models_dict[expected_key]
            assert isinstance(inference_model, DummyInferenceModel)
            assert inference_model.model_id == expected_id
