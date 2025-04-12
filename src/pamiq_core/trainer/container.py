from collections import OrderedDict
from pathlib import Path
from typing import override

from pamiq_core.data import DataUsersDict
from pamiq_core.model import TrainingModelsDict
from pamiq_core.state_persistence import PersistentStateMixin
from pamiq_core.thread import ThreadEventMixin
from pamiq_core.trainer import Trainer


class TrainersDict(OrderedDict[str, Trainer], PersistentStateMixin, ThreadEventMixin):
    """A container class for trainers."""

    def attach_training_models(self, training_models: TrainingModelsDict) -> None:
        """Attach the training_models to all trainers.

        Args:
            training_models: TrainingModelsDict to be added to each trainer.
        """
        for trainer in self.values():
            trainer.attach_training_models(training_models)

    def attach_data_users(self, data_users: DataUsersDict) -> None:
        """Attach the data_users to all trainers.

        Args:
            data_users: DataUsersDict to be added to each trainer.
        """
        for trainer in self.values():
            trainer.attach_data_users(data_users)

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

    @override
    def on_paused(self) -> None:
        """Propagate pause event to all trainers."""
        super().on_paused()
        for trainer in self.values():
            trainer.on_paused()

    @override
    def on_resumed(self) -> None:
        """Propagate resume event to all trainers."""
        super().on_resumed()
        for trainer in self.values():
            trainer.on_resumed()
