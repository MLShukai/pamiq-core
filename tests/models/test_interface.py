from collections import deque
from collections.abc import Iterable
from typing import Any, override

import pytest

from pamiq_core.model import InferenceModel, TrainingModel


class DummyInferenceModel(InferenceModel):
    @override
    def infer(self, input: list[int]) -> int:
        return sum(input)


class DummyTrainingModel(TrainingModel):
    @override
    def _create_inference_model(self) -> InferenceModel:
        return DummyInferenceModel()

    @override
    def forward(self, input: list[str]) -> str:
        return "".join(input)


class TestInferenceModel:
    @pytest.mark.parametrize("input", [[1, 10, 100]])
    def test_call(self, input: list[int]) -> None:
        dummy_inference_model = DummyInferenceModel()
        output = dummy_inference_model(input)
        expected_output = sum(input)
        assert output == expected_output


@pytest.mark.parametrize("has_inference", [True, False])
@pytest.mark.parametrize("inference_only", [True, False])
class TestTrainingModel:
    @pytest.fixture
    def dummy_training_model(
        self, has_inference: bool, inference_only: bool
    ) -> TrainingModel:
        if inference_only and not has_inference:
            with pytest.raises(ValueError):
                DummyTrainingModel(
                    has_inference=has_inference, inference_only=inference_only
                )
            pytest.skip()
        return DummyTrainingModel(
            has_inference=has_inference, inference_only=inference_only
        )

    def test_inference_model(
        self, dummy_training_model: TrainingModel, has_inference: bool
    ) -> None:
        if has_inference:
            dummy_inference_model = dummy_training_model.inference_model
            assert isinstance(dummy_inference_model, InferenceModel)
        else:
            with pytest.raises(RuntimeError):
                dummy_training_model.inference_model

    @pytest.mark.parametrize(
        "input",
        [["The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"]],
    )
    def test_call(self, dummy_training_model: TrainingModel, input: list[str]) -> None:
        output = dummy_training_model(input)
        expected_output = "".join(input)
        assert output == expected_output
