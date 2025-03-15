from abc import ABC, abstractmethod
from typing import Any

from pamiq_core.data import DataUser, DataUsersDict
from pamiq_core.model import (
    InferenceModelsDict,
    TrainingModel,
    TrainingModelsDict,
)


class Trainer(ABC):
    """Abstract base trainer class.

    The `run` method is called repeatedly in the training thread.

    Override the following methods:
        - `on_training_models_attached`: Callback method for when `model_wrappers_dict` is attached to the trainer.
        - `on_data_users_dict_attached`: Callback method when `data_users_dict` is attached to the trainer.
        - `is_trainable`: Return whether the training can be executed.
        - `setup`: To setup before training starts.
        - `train`: The training process.
        - `teardown`: To teardown after training.

    Models and data buffers become available after the thread has started.
    """

    _training_models_dict: TrainingModelsDict
    _data_users_dict: DataUsersDict

    def __init__(self) -> None:
        """Initialize."""
        super().__init__()
        self._retrieved_model_names: set[str] = set()


    def attach_training_models_dict(
        self, training_models_dict: TrainingModelsDict
    ) -> None:
        """Attaches TrainingModelsDict to this trainer."""
        self._training_models_dict = training_models_dict
        self.on_training_models_attached()

    def on_training_models_attached(self) -> None:
        """Callback method for when `model_wrappers_dict` is attached to the
        trainer.

        Use :meth:`get_training_model` to retrieve the model that will be trained.
        """
        pass

    def attach_data_users_dict(self, data_users_dict: DataUsersDict) -> None:
        """Attaches DataUsersDict dictionary."""
        self._data_users_dict = data_users_dict
        self.on_data_users_dict_attached()

    def on_data_users_dict_attached(self) -> None:
        """Callback method when `data_users_dict` is attached to the trainer.

        Use :meth:`get_data_user` to obtain the data user class for dataset.
        """
        pass

    def get_training_model(self, name: str) -> TrainingModel[Any]:
        """Retrieves the training model.

        If the specified model includes an inference model, it is
        automatically synchronized after training.
        """
        model = self._training_models_dict[name]
        self._retrieved_model_names.add(name)
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

        After this method, :meth:`sync_models` to be called.
        """

    def sync_models(self):
        """Synchronizes params of trained models to inference models."""
        for name in self._retrieved_model_names:
            self._training_models_dict[name].sync()

    def teardown(self) -> None:
        """Teardown procedure after training."""
        pass

    def run(self) -> None:
        """Runs the training process."""
        self.setup()
        self.train()
        self.sync_models()
        self.teardown()
