from collections import UserDict
from pathlib import Path
from typing import Any, override

from pamiq_core.state_persistence import PersistentStateMixin

from .interface import InferenceModel, TrainingModel


class InferenceModelsDict(UserDict[str, InferenceModel]):
    """Dictionary mapping keys to inference models."""


class TrainingModelsDict(UserDict[str, TrainingModel[Any]], PersistentStateMixin):
    """Dictionary mapping keys to training models.

    This class stores training models and manages their associated
    inference models. It ensures that inference thread only models
    cannot be accessed directly through this dictionary.
    """

    @override
    def __init__(self, *args: Any, **kwds: Any) -> None:
        """Initialize a new TrainingModelsDict.

        Creates an empty InferenceModelsDict to store associated inference models.

        Args:
            *args: Positional arguments passed to UserDict constructor.
            **kwds: Keyword arguments passed to UserDict constructor.
        """
        self._inference_models_dict = InferenceModelsDict()
        super().__init__(*args, **kwds)

    @property
    def inference_models_dict(self) -> InferenceModelsDict:
        """Provides read-only access to the inference models dictionary."""
        return self._inference_models_dict

    @override
    def __getitem__(self, key: str) -> TrainingModel[Any]:
        """Retrieve a training model by key.

        Args:
            key: The identifier of the training model to retrieve.

        Returns:
            The requested training model.

        Raises:
            KeyError: If the model is inference thread only or the key doesn't exist.
        """
        model = super().__getitem__(key)
        if model.inference_thread_only:
            raise KeyError(f"model '{key}' is inference thread only.")
        return model

    @override
    def __setitem__(self, key: str, model: TrainingModel[Any]) -> None:
        """Add or update a training model in the dictionary.

        If the training model has an associated inference model, that model
        is also stored in the inference_models_dict with the same key.

        Args:
            key: The identifier for the model.
            model: The training model to store.
        """
        super().__setitem__(key, model)
        if model.has_inference_model:
            self._inference_models_dict[key] = model.inference_model

    @override
    def save_state(self, path: Path) -> None:
        """Save the state of all training models in the dictionary.

        Creates a directory at the given path and saves each model's state
        in a subdirectory named after its key in this dictionary.

        Args:
            path: Directory path where the states should be saved
        """
        path.mkdir()
        for name, model in self.data.items():
            model.save_state(path / name)

    @override
    def load_state(self, path: Path) -> None:
        """Load the state of all training models from the given path.

        Loads each model's state from a subdirectory named after its key in this
        dictionary, then synchronizes the changes to their associated inference models.

        Args:
            path: Directory path from where the states should be loaded
        """
        for name, model in self.data.items():
            model.load_state(path / name)
            model.sync()
