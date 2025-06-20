from typing import Any, override

import gymnasium as gym

from pamiq_core.interaction import Environment

from .types import EnvReset, EnvStep, GymAction, GymObs


class GymEnvironment[O, A](Environment[GymObs[O], GymAction[A]]):
    """Wrapper for Gymnasium environments to work with PAMIQ Core.

    This class adapts Gymnasium environments to the PAMIQ Core Environment
    interface, handling observation and action conversions between the two
    frameworks.

    Example:
        ```python
        # Create from environment ID
        env = GymEnvironment("CartPole-v1")

        # Or use existing Gym environment
        gym_env = gym.make("CartPole-v1")
        env = GymEnvironment(gym_env)
        ```
    """

    _obs: GymObs[O]

    def __init__(self, env: gym.Env[O, A] | str, **gym_make_kwds: Any) -> None:
        """Initialize the GymEnvironment wrapper.

        Args:
            env: Either a Gymnasium environment instance or a string ID to create one
            **gym_make_kwds: Additional keyword arguments passed to gym.make() if env is a string
        """
        super().__init__()
        if isinstance(env, str):
            # Create environment from registered ID
            self.env: gym.Env[O, A] = gym.make(env, **gym_make_kwds)  # pyright: ignore[reportUnknownMemberType, ]
        else:
            # Use provided environment instance
            self.env = env

    @override
    def setup(self) -> None:
        """Set up the environment by resetting it to initial state.

        This method is called during environment initialization and
        stores the initial observation from the reset.
        """
        super().setup()
        # Reset environment and wrap observation in EnvReset type
        self._obs = EnvReset(*self.env.reset())

    @override
    def observe(self) -> GymObs[O]:
        """Get the current observation from the environment.

        Returns:
            The current observation, which can be:
            - EnvReset: After environment reset
            - EnvStep: After a step
            - tuple[EnvStep, EnvReset]: When episode ends and new one begins
        """
        return self._obs

    @override
    def affect(self, action: GymAction[A]) -> None:
        """Apply an action to the environment and update the observation.

        Args:
            action: The action to apply, containing the actual action value
                   and a flag indicating if reset is needed

        The observation is updated based on the step result and whether
        the episode has ended or a reset was requested.
        """
        # Execute action in the environment
        out = self.env.step(action.action)
        obs = EnvStep(out[0], float(out[1]), out[2], out[3], out[4])
        # Check if episode ended or agent requested reset
        if obs.done or action.need_reset:
            # Package both the final step and the reset observation
            obs = (obs, EnvReset(*self.env.reset()))
        self._obs = obs

    def __del__(self) -> None:
        """Clean up resources by closing the Gymnasium environment."""
        if hasattr(self, "env"):
            self.env.close()
