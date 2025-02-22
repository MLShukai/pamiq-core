from abc import ABC, abstractmethod
from typing import Any

from pamiq_core.data import DataUser, DataUsersDict
from pamiq_core.model import (
    InferenceModel,
    InferenceModelsDict,
    TrainingModel,
    TrainingModelsDict,
)


class Trainer(ABC):
    _training_models_dict: TrainingModelsDict
    _data_users_dict: DataUsersDict

    def __init__(self) -> None:
        """Initialize."""
        super().__init__()
        self._synchronoized_model_names: set[str] = set()

    @property
    def _inference_models_dict(self) -> InferenceModelsDict:
        """Get inference models from _training_models_dict."""
        return self._training_models_dict.inference_models_dict

    def attach_training_models_dict(
        self, training_models_dict: TrainingModelsDict
    ) -> None:
        """Attaches TrainingModelsDict to this trainer."""
        self._training_models_dict = TrainingModelsDict
        self.on_training_models_attached()

    def on_training_models_attached(self) -> None:
        """Callback method for when `model_wrappers_dict` is attached to the
        trainer.

        Override this method to retrieve models.

        Use :meth:`get_training_model` to retrieve the model that will be trained.
        Use :meth:`get_frozen_model` to obtain the model used exclusively for performing inference during training.
        """
        pass

    def attach_data_users_dict(self, data_users_dict: DataUsersDict) -> None:
        """Attaches DataUsersDict dictionary."""
        self._data_users_dict = data_users_dict
        self.on_data_users_dict_attached()

    def get_training_model(self, name: str) -> TrainingModel:
        """Retrieves the training model.

        If the specified model includes an inference model, it is
        automatically synchronized after training.
        """
        model = self._training_models_dict[name]
        self._synchronized_model_names.add(name)
        return model

    def get_data_user(self, name: str) -> DataUser[Any]:
        """Retrieves the data user."""
        return self._data_users_dict[name]

    def is_trainable(self) -> bool:
        """Determines if the training can be executed.

        This method checks if the training process is currently
        feasible. If it returns `False`, the training procedure is
        skipped. Subclasses should override this method to implement
        custom logic for determining trainability status.
        """
        return True

    def setup(self) -> None:
        """Setup procedure before training starts."""
        pass

    @abstractmethod
    def train(self) -> None:
        """Train models.

        Please build the models, optimizers, dataset, and other components in this method.
        This method is called repeatedly.

        After this method, :meth:`synchronize` to be called.
        """

    def synchronize(self):
        """Synchronizes params of trained models to inference models."""
        for name in self._synchronized_model_names:
            self._sync_model(
                self._training_models_dict[name],
                self._training_models_dict[name].inference_models,
            )

    @abstractmethod
    def _sync_model(
        self, training_model: TrainingModel, inference_model: InferenceModel
    ) -> None:
        """Synchronizes params of trained model to inference model.

        Args:
            training_model: Trained model.
            inference_model: Inference model.
        """
        pass

    def teardown(self) -> None:
        """Teardown procedure after training."""
        pass

    def run(self) -> None:
        """Runs the training process."""
        self.setup()
        self.train()
        self.synchronize()
        self.teardown()
