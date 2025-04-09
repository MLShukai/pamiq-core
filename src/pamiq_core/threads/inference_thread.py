from typing import override

from pamiq_core.interaction import Interaction

from .base import BackgroundThread
from .thread_types import ThreadTypes


class InferenceThread(BackgroundThread):
    """Thread for model inference running in background.

    This thread executes the interaction loop between an agent and
    environment in the background, handling the setup, step, and
    teardown lifecycle. It correctly propagates pause/resume events to
    the underlying interaction.
    """

    THREAD_TYPE = ThreadTypes.INFERENCE

    @override
    def __init__[Obs, Act](self, interaction: Interaction[Obs, Act]) -> None:
        """Initialize the inference thread.

        Args:
            interaction: The interaction object that manages the agent-environment loop.
        """
        super().__init__()
        self._interaction = interaction

    @override
    def on_start(self) -> None:
        """Execute setup procedures when the thread starts.

        Calls the interaction's setup method to initialize the agent-
        environment loop.
        """
        super().on_start()
        self._interaction.setup()

    @override
    def on_tick(self) -> None:
        """Execute a single step of the interaction loop.

        Called repeatedly during the thread's main execution loop.
        """
        super().on_tick()
        self._interaction.step()

    @override
    def on_finally(self) -> None:
        """Execute teardown procedures when the thread is about to exit.

        Ensures the interaction is properly cleaned up even if the
        thread exits unexpectedly.
        """
        super().on_finally()
        self._interaction.teardown()

    @override
    def on_paused(self) -> None:
        """Handle thread pause event.

        Propagates the pause event to the interaction to ensure
        coordinated pausing.
        """
        super().on_paused()
        self._interaction.on_paused()

    @override
    def on_resumed(self) -> None:
        """Handle thread resume event.

        Propagates the resume event to the interaction to ensure
        coordinated resuming.
        """
        super().on_resumed()
        self._interaction.on_resumed()
