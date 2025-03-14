from abc import ABC, abstractmethod

from pamiq_core.state_persistence import PersistentStateMixin

from .event_mixin import InteractionEventMixin


class Agent[ObsType, ActType](ABC, InteractionEventMixin, PersistentStateMixin):
    """Base agent class for decision making.

    An agent receives observations from an environment and decides on
    actions to take in response. This abstract class defines the
    interface that all agent implementations must follow.
    """

    @abstractmethod
    def step(self, observation: ObsType) -> ActType:
        """Processes an observation and determines the next action.

        Args:
            observation: The current observation from the environment.

        Returns:
            The action to take in response to the observation.
        """
        ...
