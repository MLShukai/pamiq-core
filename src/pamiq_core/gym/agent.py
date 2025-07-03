from abc import abstractmethod
from dataclasses import asdict
from typing import Any, override

from pamiq_core.interaction import Agent

from .types import EnvReset, EnvStep, GymAction, GymObs


class GymAgent[O, A](Agent[GymObs[O], GymAction[A]]):
    """Base class for agents that interact with Gymnasium environments.

    This abstract class provides the interface for agents to handle
    Gymnasium environment observations and produce actions.

    Set `self.need_reset=True` to reset the environment after current step.

    Example:
        ```python
        class MyCartPoleAgent(GymAgent[np.ndarray, int]):
            def on_reset(self, obs, info):
                # Return initial action
                return 0

            def on_step(self, obs, reward, truncated, terminated, info):
                # Simple policy: move right if pole is tilting right
                return 1 if obs[2] > 0 else 0
        ```
    """

    need_reset: bool = False

    @override
    def setup(self) -> None:
        """Initialize the agent and reset the need_reset flag."""
        super().setup()
        self.need_reset = False

    @abstractmethod
    def on_reset(self, obs: O, info: dict[str, Any]) -> A:
        """Handle environment reset and return initial action.

        Args:
            obs: Initial observation from the environment
            info: Additional information from the environment

        Returns:
            The initial action to take
        """
        pass

    @abstractmethod
    def on_step(
        self,
        obs: O,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ) -> A:
        """Process a step observation and return next action.

        Args:
            obs: Current observation from the environment
            reward: Reward received from the previous action
            truncated: Whether the episode was truncated
            truncated: Whether the episode was truncated before completion
            terminated: Whether the episode terminated successfully
            info: Additional information from the environment

        Returns:
            The next action to take
        """
        pass

    def _on_reset(self, obs: O, info: dict[str, Any]) -> A:
        """Internal reset handler that manages the need_reset flag."""
        self.need_reset = False
        return self.on_reset(obs, info)

    @override
    def step(self, observation: GymObs[O]) -> GymAction[A]:
        """Process observation and return action wrapped with reset flag.

        Handles different observation types:
        - EnvReset: Initial observation after reset
        - EnvStep: Regular step observation
        - tuple: Combined step and reset (episode end)

        Args:
            observation: Current observation from the environment

        Returns:
            Action wrapped with need_reset flag
        """
        match observation:
            case EnvReset():
                action = self._on_reset(**asdict(observation))
            case EnvStep():
                action = self.on_step(**asdict(observation))
            case tuple():
                # Process final step then reset
                self.on_step(**asdict(observation[0]))
                action = self._on_reset(**asdict(observation[1]))
        return GymAction(action, self.need_reset)
