import statistics
from typing import override

from pamiq_core import time
from pamiq_core.interaction import Interaction
from pamiq_core.utils.schedulers import TimeIntervalScheduler

from ..thread_types import ThreadTypes
from .base import BackgroundThread


class InferenceThread(BackgroundThread):
    """Thread for model inference running in background.

    This thread executes the interaction loop between an agent and
    environment in the background, handling the setup, step, and
    teardown lifecycle. It correctly propagates pause/resume events to
    the underlying interaction.
    """

    THREAD_TYPE = ThreadTypes.INFERENCE

    @override
    def __init__[Obs, Act](
        self,
        interaction: Interaction[Obs, Act],
        log_tick_time_statistics_interval: float = 60.0,
    ) -> None:
        """Initialize the inference thread.

        Args:
            interaction: The interaction object that manages the agent-environment loop.
            log_tick_time_statistics_interval: Interval in seconds for logging step time statistics.
        """
        super().__init__()
        self._interaction = interaction
        self._tick_times: list[float] = []
        self._tick_start: float | None = None
        self._log_tick_time_scheduler = TimeIntervalScheduler(
            log_tick_time_statistics_interval, self.log_tick_time_statistics
        )

    def log_tick_time_statistics(self) -> None:
        if len(self._tick_times) == 0:
            return
        mean = statistics.mean(self._tick_times)
        stdev = statistics.stdev(self._tick_times)
        self._logger.info(
            f"Step time: {mean:.3e} ± {stdev:.3e} [s] in {len(self._tick_times)} steps."
        )
        self._tick_times.clear()

    @override
    def on_start(self) -> None:
        """Execute setup procedures when the thread starts."""
        super().on_start()
        self._interaction.setup()

    @override
    def on_tick(self) -> None:
        """Execute a single step of the interaction loop."""
        super().on_tick()
        self._interaction.step()

        if self._tick_start is not None:
            self._tick_times.append(time.fixed_time() - self._tick_start)
        self._tick_start = time.fixed_time()
        self._log_tick_time_scheduler.update()

    @override
    def on_finally(self) -> None:
        """Execute teardown procedures when the thread is about to exit."""
        super().on_finally()
        self._interaction.teardown()

    @override
    def on_paused(self) -> None:
        """Handle thread pause event."""
        super().on_paused()
        self._interaction.on_paused()

    @override
    def on_resumed(self) -> None:
        """Handle thread resume event."""
        super().on_resumed()
        self._interaction.on_resumed()
