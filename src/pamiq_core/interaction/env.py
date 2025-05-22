from abc import ABC, abstractmethod

from pamiq_core.state_persistence import PersistentStateMixin
from pamiq_core.thread import ThreadEventMixin

from .event_mixin import InteractionEventMixin


class Environment[ObsType, ActType](
    ABC, InteractionEventMixin, PersistentStateMixin, ThreadEventMixin
):
    """Base environment class for agent interaction."""

    @abstractmethod
    def observe(self) -> ObsType:
        """Gets and returns observation from the environment.

        Returns:
            Observation from the environment.
        """
        pass

    @abstractmethod
    def affect(self, action: ActType) -> None:
        """Applies an action to the environment.

        Args:
            action: Action to apply to the environment.
        """
        pass
