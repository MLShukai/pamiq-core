from typing import override

import pytest

from pamiq_core.model import InferenceModel, TrainingModel
from pamiq_core.state_persistence import PersistentStateMixin


class InferenceModelImpl(InferenceModel):
    param: int = 1234

    @override
    def infer(self, input: list[int]) -> int:
        return sum(input)


class TrainingModelImpl(TrainingModel[InferenceModelImpl]):
    param: int = 9999

    @override
    def _create_inference_model(self) -> InferenceModelImpl:
        return InferenceModelImpl()

    @override
    def forward(self, input: list[str]) -> str:
        return "".join(input)

    @override
    def sync_impl(self, inference_model: InferenceModelImpl) -> None:
        inference_model.param = self.param


class TestInferenceModel:
    @pytest.mark.parametrize("method", ["infer"])
    def test_abstractmethods(self, method):
        assert method in InferenceModel.__abstractmethods__

    @pytest.mark.parametrize("input", [[1, 10, 100]])
    def test_call(self, input: list[int]) -> None:
        inference_model = InferenceModelImpl()
        output = inference_model(input)
        expected_output = sum(input)
        assert output == expected_output


class TestTrainingModel:
    @pytest.fixture
    def training_model(self) -> TrainingModel:
        return TrainingModelImpl()

    @pytest.fixture
    def training_model_no_inference(self):
        return TrainingModelImpl(has_inference_model=False)

    @pytest.fixture
    def training_model_inference_only(self):
        return TrainingModelImpl(inference_thread_only=True)

    def test_inconsistent_inference_model_option(self):
        with pytest.raises(ValueError):
            TrainingModelImpl(
                has_inference_model=False,
                inference_thread_only=True,
            )

    @pytest.mark.parametrize(
        "method", ["_create_inference_model", "forward", "sync_impl"]
    )
    def test_abstractmethods(self, method):
        assert method in TrainingModel.__abstractmethods__

    def test_model_subclass(self):
        assert issubclass(TrainingModel, PersistentStateMixin)

    def test_inference_model(
        self, training_model, training_model_inference_only, training_model_no_inference
    ) -> None:
        for model in [training_model, training_model_inference_only]:
            assert isinstance(model.inference_model, InferenceModelImpl)

        with pytest.raises(RuntimeError):
            training_model_no_inference.inference_model

    def test_call(self, training_model: TrainingModel) -> None:
        input = ["The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"]
        output = training_model(input)
        expected_output = training_model.forward(input)
        assert output == expected_output

    def test_sync(
        self,
        training_model: TrainingModelImpl,
    ) -> None:
        inference_model = training_model.inference_model
        training_model.param += 1
        assert training_model.param != inference_model.param
        training_model.sync()
        assert training_model.param == inference_model.param
