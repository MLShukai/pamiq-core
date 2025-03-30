import pytest
from pytest_mock import MockerFixture

from pamiq_core.model import (
    InferenceModel,
    InferenceModelsDict,
    TrainingModel,
    TrainingModelsDict,
)


class TestTrainingModelsDict:
    def create_model(
        self,
        mocker: MockerFixture,
        has_inference_model: bool = True,
        inference_thread_only: bool = False,
    ):
        model = mocker.Mock(TrainingModel)
        model.inference_model = mocker.Mock(InferenceModel)
        model.has_inference_model = has_inference_model
        model.inference_thread_only = inference_thread_only
        return model

    @pytest.fixture
    def default_model(self, mocker):
        return self.create_model(mocker)

    @pytest.fixture
    def no_inference_model(self, mocker):
        return self.create_model(mocker, has_inference_model=False)

    @pytest.fixture
    def inference_only_model(self, mocker):
        return self.create_model(mocker, inference_thread_only=True)

    @pytest.fixture
    def training_models_dict(
        self, default_model, no_inference_model, inference_only_model
    ) -> TrainingModelsDict:
        return TrainingModelsDict(
            {
                "model": default_model,
                "no_inference": no_inference_model,
                "inference_only": inference_only_model,
            }
        )

    def test_inference_models_dict(
        self,
        training_models_dict: TrainingModelsDict,
        default_model,
        inference_only_model,
    ) -> None:
        inference_models_dict = training_models_dict.inference_models_dict
        assert isinstance(inference_models_dict, InferenceModelsDict)
        assert inference_models_dict["model"] is default_model.inference_model
        assert (
            inference_models_dict["inference_only"]
            is inference_only_model.inference_model
        )
        assert "no_inference" not in inference_models_dict

    def test_getitem(
        self,
        training_models_dict: TrainingModelsDict,
        default_model,
        no_inference_model,
    ) -> None:
        assert training_models_dict["model"] is default_model
        assert training_models_dict["no_inference"] is no_inference_model

        with pytest.raises(
            KeyError, match="model 'inference_only' is inference thread only."
        ):
            training_models_dict["inference_only"]
