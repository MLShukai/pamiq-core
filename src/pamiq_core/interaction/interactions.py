from pathlib import Path
from typing import override

from pamiq_core.state_persistence import PersistentStateMixin

from .agent import Agent
from .env import Environment
from .event_mixin import InteractionEventMixin


class Interaction[ObsType, ActType](InteractionEventMixin, PersistentStateMixin):
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
