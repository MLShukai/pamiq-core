from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Any

from pamiq_core.data import DataUser, DataUsersDict
from pamiq_core.model import (
    TrainingModel,
    TrainingModelsDict,
)
from pamiq_core.state_persistence import PersistentStateMixin
from pamiq_core.trainer import Trainer


class TrainersDict(OrderedDict[str, Trainer], PersistentStateMixin):
    def __init__(self, *args: Any, **kwds: Any) -> None:
        """Initialize."""
        super().__init__(*args, **kwds)
        self._current_index = 0

    def get_trainers_list(self) -> list[Trainer]:
        """Retrieve trainers as a list."""
        return list(self.values())

    def get_trainable_trainer(self) -> Trainer | None:
        """Retrieve trainable trainer in order one by one.

        Returns:
            If chosen trainer is trainable, returns trainer. Else None.
        """
        trainers = self.get_trainers_list()
        for i in range(len(trainers)):
            self._current_index = (self._current_index + i) % len(self)
            trainer = trainers[self._current_index]
            if trainer.is_trainable():
                return trainer
        return None

    def attach_training_models_dict(
        self, training_models_dict: TrainingModelsDict
    ) -> None:
        """Add the training_models_dict to all trainers.

        Args:
            training_models_dict: TrainingModelsDict to be added to each trainer.
        """

    def attach_data_users_dict(self, data_users_dict: DataUsersDict) -> None:
        """Add the data_users_dict to all trainers.

        Args:
            data_users_dict: DataUsersDict to be added to each trainer.
        """
