from collections import UserDict
from typing import Any, override

from .interface import InferenceModel, TrainingModel


class InferenceModelsDict(UserDict[str, InferenceModel]):
    """Wrapper class for models to infer."""


class TrainingModelsDict(UserDict[str, TrainingModel]):
    """Wrapper class to train model."""

    @override
    def __init__(self, *args: Any, **kwds: Any) -> None:
        """Initialize."""
        self._inference_models_dict = InferenceModelsDict()
        super().__init__(*args, **kwds)

    @property
    def inference_models_dict(self) -> InferenceModelsDict:  # Define getter only
        """Getter of self._inference_models_dict."""
        return self._inference_models_dict

    @override
    def __getitem__(self, key: str) -> TrainingModel:
        """Select training model by a key.

        If it is inference only, raise KeyError.
        """
        model = super().__getitem__(key)
        if model.inference_only:
            raise KeyError(f"model '{key}' is inference only.")
        return model

    @override
    def __setitem__(self, key: str, model: TrainingModel) -> None:
        """Register a key and a training model to this user dict.

        If the training model has inference model, set the key and it to
        self._inference_models_dict.
        """
        super().__setitem__(key, model)
        if model.has_inference_model:
            self._inference_models_dict[key] = model.inference_model
