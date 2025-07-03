"""Minimal example demonstrating how to launch a pamiq-core system.

This example shows the basic structure needed to run a system with pamiq-core:
1. Create an Agent and Environment
2. Set up an Interaction between them
3. Launch the system with a proper configuration
"""

import logging
from tempfile import TemporaryDirectory
from typing import override

from pamiq_core import Agent, Environment, Interaction, LaunchConfig, launch


def setup_logging() -> None:
    """Configure logging for the demo."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )


class MinimalAgent(Agent[None, None]):
    """Minimal agent implementation that does nothing.

    This demonstrates the minimal structure needed for an Agent in
    pamiq-core. For real applications, implement meaningful observation
    processing and action generation.
    """

    @override
    def step(self, observation: None) -> None:
        """Process observation and return action.

        Args:
            observation: Input from the environment (None in this example)

        Returns:
            Action to be sent to the environment (None in this example)
        """
        return None


class MinimalEnvironment(Environment[None, None]):
    """Minimal environment implementation that does nothing.

    This demonstrates the minimal structure needed for an Environment in
    pamiq-core. For real applications, implement meaningful observation
    generation and action handling.
    """

    @override
    def observe(self) -> None:
        """Generate an observation from the environment.

        Returns:
            Observation data (None in this example)
        """
        return None

    @override
    def affect(self, action: None) -> None:
        """Apply an action to the environment.

        Args:
            action: Action to apply to the environment (None in this example)
        """
        pass


def main() -> None:
    """Run a minimal pamiq-core system.

    Creates a system with null agent and environment, then launches it
    with temporary state storage and a local web API endpoint.
    """
    setup_logging()

    with TemporaryDirectory() as tmp_dir:
        launch(
            interaction=Interaction(MinimalAgent(), MinimalEnvironment()),
            models={},
            buffers={},
            trainers={},
            config=LaunchConfig(
                states_dir=tmp_dir, web_api_address=("localhost", 8391)
            ),
        )


if __name__ == "__main__":
    main()
