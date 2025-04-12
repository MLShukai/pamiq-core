from functools import cached_property
from typing import override

from pamiq_core import time
from pamiq_core.trainer import Trainer, TrainersDict

from ..thread_types import ThreadTypes
from .base import BackgroundThread


class TrainingThread(BackgroundThread):
    """Background thread for running model training.

    This thread iterates through a collection of trainers, executing
    each trainer's run method in sequence. It cycles through all
    trainers continuously while active.
    """

    THREAD_TYPE = ThreadTypes.TRAINING

    @override
    def __init__(self, trainers: TrainersDict) -> None:
        """Initialize the training thread.

        Args:
            trainers: Dictionary of trainers to be executed by this thread.
        """
        super().__init__()
        self._trainers = trainers
        self._current_trainer_index = 0

    @cached_property
    def _trainers_items(self) -> list[tuple[str, Trainer]]:
        """Get trainers items as a list.

        Returns:
            List of (name, trainer) tuples.
        """
        return list(self._trainers.items())

    @override
    def on_tick(self) -> None:
        """Execute one tick of the training loop.

        Runs the next trainer in the sequence and logs if training was
        performed.
        """
        super().on_tick()
        if len(self._trainers) == 0:
            return
        name, trainer = self._trainers_items[self._current_trainer_index]
        start = time.fixed_time()
        if trainer.run():
            self._logger.info(
                f"Trained {name} in {time.fixed_time() - start:.2f} seconds"
            )
        self._current_trainer_index = (self._current_trainer_index + 1) % len(
            self._trainers
        )

    @override
    def on_paused(self) -> None:
        """Handle thread pause event."""
        super().on_paused()
        self._trainers.on_paused()

    @override
    def on_resumed(self) -> None:
        """Handle thread resume event."""
        super().on_resumed()
        self._trainers.on_resumed()
