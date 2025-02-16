from collections import UserDict

from .interface import InferenceModel, TrainingModel


class InferenceModelsDict(UserDict[str, InferenceModel]):
    """Wrapper class for models to infer."""

    ...


class TrainingModelsDict(UserDict[str, TrainingModel]):
    """Wrapper class to train model."""

    _inference_models_dict: InferenceModelsDict | None = None

    @property
    def inference_models_dict(self) -> InferenceModelsDict:
        """Make InferenceModelsDict."""
        if self._inference_models_dict is None:
            self._inference_models_dict = self._create_inferences()
        return self._inference_models_dict

    def _create_inferences(self) -> InferenceModelsDict:
        """Select training models that has inference_model and isn't inference
        only and return them as InferenceModelsDict.

        Returns: InferenceModelsDict
        """
        return InferenceModelsDict(
            {
                model_name: training_model.inference_model
                for model_name, training_model in self.items()
                if training_model.has_inference_model
                and not training_model.inference_only
            }
        )
