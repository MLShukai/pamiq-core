from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, override

from pamiq_core import time
from pamiq_core.data import DataUser, DataUsersDict
from pamiq_core.model import TrainingModel, TrainingModelsDict
from pamiq_core.state_persistence import PersistentStateMixin
from pamiq_core.thread import ThreadEventMixin


class Trainer(ABC, PersistentStateMixin, ThreadEventMixin):
    """Abstract base trainer class.

    The `run` method is called repeatedly in the training thread.

    Override the following methods:
        - `on_training_models_attached`: Callback method for when training models are attached to the trainer.
        - `on_data_users_attached`: Callback method when data_users are attached to the trainer.
        - `is_trainable`: Return whether the training can be executed.
        - `setup`: To setup before training starts.
        - `train`: The training process.
        - `teardown`: To teardown after training.

    Models and data buffers become available after the thread has started.
    """

    _training_models: TrainingModelsDict
    _data_users: DataUsersDict

    def __init__(
        self,
        training_condition_data_user: str | None = None,
        min_buffer_size: int = 0,
        min_new_data_count: int = 0,
    ) -> None:
        """Initialize a trainer.

        Args:
            training_condition_data_user: Name of the data user to check for trainability.
                If None, trainer is always trainable.
            min_buffer_size: Minimum total data points required in the buffer.
            min_new_data_count: Minimum number of new data points required since last training.
        """
        super().__init__()
        self._retrieved_model_names: set[str] = set()
        self._training_condition_data_user = training_condition_data_user
        self._min_buffer_size = min_buffer_size
        self._min_new_data_count = min_new_data_count
        self._previous_training_time = float("-inf")

    def attach_training_models(self, training_models: TrainingModelsDict) -> None:
        """Attaches TrainingModelsDict to this trainer."""
        self._training_models = training_models
        self.on_training_models_attached()

    def on_training_models_attached(self) -> None:
        """Callback method for when training models are attached to the
        trainer.

        Use :meth:`get_training_model` to retrieve the model that will be trained.
        """
        pass

    def attach_data_users(self, data_users: DataUsersDict) -> None:
        """Attaches DataUsersDict to this trainer."""
        self._data_users = data_users
        self.on_data_users_attached()

    def on_data_users_attached(self) -> None:
        """Callback method when data users are attached to the trainer.

        Use :meth:`get_data_user` to obtain the data user class for dataset.
        """
        pass

    def get_training_model(self, name: str) -> TrainingModel[Any]:
        """Retrieves the training model.

        If the specified model includes an inference model, it is
        automatically synchronized after training.
        """
        model = self._training_models[name]
        self._retrieved_model_names.add(name)
        return model

    def get_data_user(self, name: str) -> DataUser[Any]:
        """Retrieves the data user."""
        return self._data_users[name]

    def is_trainable(self) -> bool:
        """Determines if the training can be executed.

        Checks if training can proceed based on data availability when
        a training condition data user is specified.

        Returns:
            True if training can be executed, False otherwise.
        """
        # If no data user is specified for condition checking, always trainable
        if self._training_condition_data_user is None:
            return True

        data_user = self.get_data_user(self._training_condition_data_user)
        data_user.update()

        trainable = (
            len(data_user) >= self._min_buffer_size
            and data_user.count_data_added_since(self._previous_training_time)
            >= self._min_new_data_count
        )

        if trainable:
            self._previous_training_time = time.time()

        return trainable

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

    def sync_models(self) -> None:
        """Synchronizes params of trained models to inference models."""
        for name in self._retrieved_model_names:
            self._training_models[name].sync()

    def teardown(self) -> None:
        """Teardown procedure after training."""
        pass

    def run(self) -> bool:
        """Runs the training process if the trainer is trainable.

        Returns:
            bool: True if training was executed, False if skipped due to conditions not met.
        """
        if not self.is_trainable():
            return False

        self.setup()
        self.train()
        self.sync_models()
        self.teardown()
        return True

    @override
    def save_state(self, path: Path) -> None:
        """Save the trainer state to the specified path.

        Args:
            path: Directory path where to save the trainer state.
        """
        path.mkdir()
        (path / "previous_training_time").write_text(
            str(self._previous_training_time), encoding="utf-8"
        )

    @override
    def load_state(self, path: Path) -> None:
        """Load the trainer state from the specified path.

        Args:
            path: Directory path from where to load the trainer state.
        """
        self._previous_training_time = float(
            (path / "previous_training_time").read_text("utf-8")
        )
