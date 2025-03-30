import pytest

from pamiq_core.model import (
    InferenceModelsDict,
    TrainingModelsDict,
)

from .helpers import InferenceModelImpl, TrainingModelImpl


class TestTrainingModelsDict:
    @pytest.fixture
    def training_models_dict(self) -> TrainingModelsDict:
        return TrainingModelsDict(
            {
                "model": TrainingModelImpl(),
                "no_inference": TrainingModelImpl(has_inference_model=False),
                "inference_only": TrainingModelImpl(inference_thread_only=True),
            }
        )

    def test_inference_models_dict(
        self,
        training_models_dict: TrainingModelsDict,
    ) -> None:
        inference_models_dict = training_models_dict.inference_models_dict
        assert isinstance(inference_models_dict, InferenceModelsDict)
        assert isinstance(inference_models_dict["model"], InferenceModelImpl)
        assert isinstance(inference_models_dict["inference_only"], InferenceModelImpl)
        assert "no_inference" not in inference_models_dict

    def test_getitem(
        self,
        training_models_dict: TrainingModelsDict,
    ) -> None:
        assert isinstance(training_models_dict["model"], TrainingModelImpl)
        assert isinstance(training_models_dict["no_inference"], TrainingModelImpl)

        with pytest.raises(
            KeyError, match="model 'inference_only' is inference thread only."
        ):
            training_models_dict["inference_only"]
