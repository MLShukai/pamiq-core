from collections import OrderedDict
from pathlib import Path
from typing import Any, override

from pamiq_core.data import DataUsersDict
from pamiq_core.model import TrainingModelsDict
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
        for _ in range(len(trainers)):
            trainer = trainers[self._current_index]
            self._current_index = (self._current_index + 1) % len(self)
            if trainer.is_trainable():
                return trainer
        return None

    def attach_training_models_dict(
        self, training_models_dict: TrainingModelsDict
    ) -> None:
        """Attach the training_models_dict to all trainers.

        Args:
            training_models_dict: TrainingModelsDict to be added to each trainer.
        """
        for trainer in self.values():
            trainer.attach_training_models_dict(training_models_dict)

    def attach_data_users_dict(self, data_users_dict: DataUsersDict) -> None:
        """Attach the data_users_dict to all trainers.

        Args:
            data_users_dict: DataUsersDict to be added to each trainer.
        """
        for trainer in self.values():
            trainer.attach_data_users_dict(data_users_dict)

    @override
    def save_state(self, path: Path) -> None:
        """Save states of each trainer to the path."""
        path.mkdir()
        for name, trainer in self.items():
            trainer.save_state(path / name)

    @override
    def load_state(self, path: Path) -> None:
        """Load states of each trainer from the path."""
        for name, trainer in self.items():
            trainer.load_state(path / name)
