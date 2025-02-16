from collections import UserDict

from .interface import InferenceModel, TrainingModel


class InferenceModelsDict(UserDict[str, InferenceModel]): ...


class TrainingModelsDict(UserDict[str, TrainingModel]):
    _inference_models_dict: InferenceModelsDict | None = None

    @property
    def inference_models_dict(self) -> InferenceModelsDict:
        if self._inference_models_dict is None:
            self._inference_models_dict = self._create_inferences()
        return self._inference_models_dict

    def _create_inferences(self) -> InferenceModelsDict:
        return InferenceModelsDict(
            {
                model_name: training_model.inference_model
                for model_name, training_model in self.items()
                if training_model.has_inference_model
                and not training_model.inference_only
            }
        )
