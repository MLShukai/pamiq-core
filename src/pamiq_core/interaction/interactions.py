from pathlib import Path
from typing import Self, override

from pamiq_core.state_persistence import PersistentStateMixin
from pamiq_core.thread import ThreadEventMixin

from .agent import Agent
from .env import Environment
from .event_mixin import InteractionEventMixin
from .interval_adjustors import IntervalAdjustor, SleepIntervalAdjustor


class Interaction[ObsType, ActType](
    InteractionEventMixin, PersistentStateMixin, ThreadEventMixin
):
    """Class that combines an agent and environment to create an interaction
    loop.

    This class manages the interaction between an agent and its
    environment, implementing a basic observe-decide-act loop. It also
    handles state persistence and lifecycle management for both
    components.
    """

    def __init__(
        self, agent: Agent[ObsType, ActType], environment: Environment[ObsType, ActType]
    ) -> None:
        """Initialize interaction with an agent and environment.

        Args:
            agent: The agent that makes decisions based on observations.
            environment: The environment that provides observations and receives actions.
        """
        self.agent = agent
        self.environment = environment

    def step(self) -> None:
        """Execute one step of the agent-environment interaction loop.

        Gets an observation from the environment, passes it to the agent
        to get an action, and then applies that action to the
        environment.
        """
        obs = self.environment.observe()
        action = self.agent.step(obs)
        self.environment.affect(action)

    @override
    def setup(self) -> None:
        """Initialize the interaction by setting up agent and environment.

        Calls the setup methods of both the agent and environment.
        """
        super().setup()
        self.agent.setup()
        self.environment.setup()

    @override
    def teardown(self) -> None:
        """Clean up the interaction by tearing down agent and environment.

        Calls the teardown methods of both the agent and environment.
        """
        super().teardown()
        self.agent.teardown()
        self.environment.teardown()

    @override
    def save_state(self, path: Path) -> None:
        """Save the current state of the interaction to the specified path.

        Creates a directory at the given path and saves the states of both
        the agent and environment in subdirectories.

        Args:
            path: Directory path where to save the interaction state.
        """
        path.mkdir()
        self.agent.save_state(path / "agent")
        self.environment.save_state(path / "environment")

    @override
    def load_state(self, path: Path) -> None:
        """Load the interaction state from the specified path.

        Loads the states of both the agent and environment from subdirectories
        at the given path.

        Args:
            path: Directory path from where to load the interaction state.
        """
        self.agent.load_state(path / "agent")
        self.environment.load_state(path / "environment")

    @override
    def on_paused(self) -> None:
        """The method to be called when the thread is paused."""
        super().on_paused()
        self.agent.on_paused()
        self.environment.on_paused()

    @override
    def on_resumed(self) -> None:
        """The method to be called when the thread is resumed."""
        super().on_resumed()
        self.agent.on_resumed()
        self.environment.on_resumed()


class FixedIntervalInteraction[ObsType, ActType](Interaction[ObsType, ActType]):
    """Interaction class that executes steps at fixed time intervals.

    This class extends the base Interaction to maintain a consistent
    timing between steps using an IntervalAdjustor. It ensures the
    agent-environment interaction loop runs at a specified frequency
    regardless of how long individual steps take to compute.
    """

    @override
    def __init__(
        self,
        agent: Agent[ObsType, ActType],
        environment: Environment[ObsType, ActType],
        adjustor: IntervalAdjustor,
    ) -> None:
        """Initialize the fixed interval interaction.

        Args:
            agent: The agent that makes decisions based on observations.
            environment: The environment that provides observations and receives actions.
            adjustor: The interval adjustor that maintains consistent timing between steps.
        """
        super().__init__(agent, environment)
        self._adjustor = adjustor

    @override
    def setup(self) -> None:
        """Initialize the interaction and reset the interval adjustor."""
        super().setup()
        self._adjustor.reset()

    @override
    def step(self) -> None:
        """Execute one step of the interaction and adjust timing."""
        super().step()
        self._adjustor.adjust()

    @classmethod
    def with_sleep_adjustor(
        cls,
        agent: Agent[ObsType, ActType],
        environment: Environment[ObsType, ActType],
        interval: float,
        offset: float = 0.0,
    ) -> Self:
        """Create a FixedIntervalInteraction with a SleepIntervalAdjustor.

        Args:
            agent: The agent that makes decisions based on observations.
            environment: The environment that provides observations and receives actions.
            interval: The desired time between each step in seconds.
            offset: Optional initial time offset to adjust for system-specific timing differences.
                Defaults to 0.0.

        Returns:
            A new FixedIntervalInteraction instance configured with a SleepIntervalAdjustor.
        """
        return cls(agent, environment, SleepIntervalAdjustor(interval, offset))
