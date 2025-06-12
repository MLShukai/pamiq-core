from __future__ import annotations

from enum import Enum, auto

from pamiq_core.thread import ReadOnlyController, ThreadStatusesMonitor


class SystemStatus(Enum):
    """Enum representing the system's operational status."""

    ACTIVE = auto()
    PAUSING = auto()
    PAUSED = auto()
    RESUMING = auto()
    SHUTTING_DOWN = auto()
    OFFLINE = auto()

    @property
    def status_name(self) -> str:
        """Returns the lowercase string representation of the status.

        Returns:
            The lowercase string name of the status.
        """
        return _STATUS_NAMES[self]


_STATUS_NAMES = {
    SystemStatus.ACTIVE: "active",
    SystemStatus.PAUSING: "pausing",
    SystemStatus.PAUSED: "paused",
    SystemStatus.RESUMING: "resuming",
    SystemStatus.SHUTTING_DOWN: "shutting down",
    SystemStatus.OFFLINE: "offline",
}


class SystemStatusProvider:
    """Provides the current status of the system based on thread controller and
    thread statuses.

    This class acts as a bridge between the thread control mechanism and
    the UI/console, determining the overall system status by examining
    the controller and thread states.
    """

    def __init__(
        self,
        controller: ReadOnlyController,
        thread_statuses_monitor: ThreadStatusesMonitor,
    ) -> None:
        """Initialize the SystemStatusProvider.

        Args:
            controller: A read-only interface to the thread controller
            thread_statuses_monitor: Monitor for observing and querying thread statuses
        """
        self._controller = controller
        self._thread_statuses_monitor = thread_statuses_monitor

    def get_current_status(self) -> SystemStatus:
        """Determine the current status of the system.

        The status is determined based on the following rules:
        - SHUTTING_DOWN: if the controller is in shutdown state
        - PAUSED: if the controller is paused and all threads are paused
        - PAUSING: if the controller is paused but not all threads are paused yet
        - RESUMING: if the controller is resumed but some threads are still paused
        - ACTIVE: if the controller is resumed and no threads are paused

        Returns:
            The current system status
        """
        if self._controller.is_shutdown():
            return SystemStatus.SHUTTING_DOWN

        if self._controller.is_pause():
            if self._thread_statuses_monitor.check_all_threads_paused():
                return SystemStatus.PAUSED
            else:
                return SystemStatus.PAUSING
        elif self._controller.is_resume():
            if self._thread_statuses_monitor.check_any_threads_paused():
                return SystemStatus.RESUMING

        return SystemStatus.ACTIVE
