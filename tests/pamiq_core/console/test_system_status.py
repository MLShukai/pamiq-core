import pytest

from pamiq_core.console.system_status import SystemStatus, SystemStatusProvider
from pamiq_core.threads import (
    ThreadController,
    ThreadStatus,
    ThreadStatusesHandler,
    ThreadTypes,
)


class TestSystemStatusProvider:
    """A test class for SystemStatusProvider."""

    @pytest.fixture()
    def thread_controller(self) -> ThreadController:
        """Fixture for a ThreadController instance."""
        return ThreadController()

    @pytest.fixture()
    def inference_thread_status(self) -> ThreadStatus:
        """Fixture for a ThreadStatus instance for inference thread."""
        return ThreadStatus()

    @pytest.fixture()
    def training_thread_status(self) -> ThreadStatus:
        """Fixture for a ThreadStatus instance for training thread."""
        return ThreadStatus()

    @pytest.fixture()
    def thread_statuses_handler(
        self,
        inference_thread_status: ThreadStatus,
        training_thread_status: ThreadStatus,
    ) -> ThreadStatusesHandler:
        """Fixture for a ThreadStatusesHandler instance with real thread
        statuses."""
        return ThreadStatusesHandler(
            {
                ThreadTypes.INFERENCE: inference_thread_status.read_only,
                ThreadTypes.TRAINING: training_thread_status.read_only,
            }
        )

    @pytest.fixture()
    def status_provider(
        self,
        thread_controller: ThreadController,
        thread_statuses_handler: ThreadStatusesHandler,
    ) -> SystemStatusProvider:
        """Fixture for a SystemStatusProvider instance with real objects."""
        return SystemStatusProvider(
            thread_controller.read_only, thread_statuses_handler
        )

    def test_status_shutting_down(
        self,
        status_provider: SystemStatusProvider,
        thread_controller: ThreadController,
        inference_thread_status: ThreadStatus,
        training_thread_status: ThreadStatus,
    ) -> None:
        """Test that status is SHUTTING_DOWN when controller is shutdown."""
        # Arrange
        thread_controller.shutdown()

        # Act & Assert
        assert status_provider.get_current_status() is SystemStatus.SHUTTING_DOWN

        # Status should be SHUTTING_DOWN regardless of thread states
        inference_thread_status.pause()
        training_thread_status.pause()
        assert status_provider.get_current_status() is SystemStatus.SHUTTING_DOWN

        inference_thread_status.resume()
        training_thread_status.resume()
        assert status_provider.get_current_status() is SystemStatus.SHUTTING_DOWN

    def test_status_paused(
        self,
        status_provider: SystemStatusProvider,
        thread_controller: ThreadController,
        inference_thread_status: ThreadStatus,
        training_thread_status: ThreadStatus,
    ) -> None:
        """Test that status is PAUSED when controller is paused and all threads
        are paused."""
        # Arrange
        thread_controller.pause()
        inference_thread_status.pause()
        training_thread_status.pause()

        # Act & Assert
        assert status_provider.get_current_status() is SystemStatus.PAUSED

    def test_status_pausing(
        self,
        status_provider: SystemStatusProvider,
        thread_controller: ThreadController,
        inference_thread_status: ThreadStatus,
        training_thread_status: ThreadStatus,
    ) -> None:
        """Test that status is PAUSING when controller is paused but not all
        threads are paused."""
        # Arrange - only one thread paused
        thread_controller.pause()
        inference_thread_status.pause()
        training_thread_status.resume()

        # Act & Assert
        assert status_provider.get_current_status() is SystemStatus.PAUSING

        # Also PAUSING when no threads are paused
        inference_thread_status.resume()
        assert status_provider.get_current_status() is SystemStatus.PAUSING

    def test_status_resuming(
        self,
        status_provider: SystemStatusProvider,
        thread_controller: ThreadController,
        inference_thread_status: ThreadStatus,
        training_thread_status: ThreadStatus,
    ) -> None:
        """Test that status is RESUMING when controller is resumed but some
        threads are still paused."""
        # Arrange - controller resumed but one thread still paused
        thread_controller.resume()
        inference_thread_status.pause()
        training_thread_status.resume()

        # Act & Assert
        assert status_provider.get_current_status() is SystemStatus.RESUMING

        # Also RESUMING when all threads paused but controller resumed
        training_thread_status.pause()
        assert status_provider.get_current_status() is SystemStatus.RESUMING

    def test_status_active(
        self,
        status_provider: SystemStatusProvider,
        thread_controller: ThreadController,
        inference_thread_status: ThreadStatus,
        training_thread_status: ThreadStatus,
    ) -> None:
        """Test that status is ACTIVE when controller is resumed and no threads
        are paused."""
        # Arrange
        thread_controller.resume()
        inference_thread_status.resume()
        training_thread_status.resume()

        # Act & Assert
        assert status_provider.get_current_status() is SystemStatus.ACTIVE

    def test_transition_from_pausing_to_paused(
        self,
        status_provider: SystemStatusProvider,
        thread_controller: ThreadController,
        inference_thread_status: ThreadStatus,
        training_thread_status: ThreadStatus,
    ) -> None:
        """Test transition from PAUSING to PAUSED when threads finish
        pausing."""
        # Arrange - pausing state
        thread_controller.pause()
        inference_thread_status.resume()
        training_thread_status.resume()
        assert status_provider.get_current_status() is SystemStatus.PAUSING

        # Act - transition to all threads paused
        inference_thread_status.pause()
        training_thread_status.pause()

        # Assert - now paused
        assert status_provider.get_current_status() is SystemStatus.PAUSED

    def test_transition_from_resuming_to_active(
        self,
        status_provider: SystemStatusProvider,
        thread_controller: ThreadController,
        inference_thread_status: ThreadStatus,
        training_thread_status: ThreadStatus,
    ) -> None:
        """Test transition from RESUMING to ACTIVE when threads finish
        resuming."""
        # Arrange - resuming state
        thread_controller.resume()
        inference_thread_status.pause()
        training_thread_status.pause()
        assert status_provider.get_current_status() is SystemStatus.RESUMING

        # Act - transition to all threads resumed
        inference_thread_status.resume()
        training_thread_status.resume()

        # Assert - now active
        assert status_provider.get_current_status() is SystemStatus.ACTIVE

    def test_transition_to_shutting_down(
        self,
        status_provider: SystemStatusProvider,
        thread_controller: ThreadController,
        inference_thread_status: ThreadStatus,
        training_thread_status: ThreadStatus,
    ) -> None:
        """Test transition to SHUTTING_DOWN from any state."""
        # From ACTIVE
        thread_controller.resume()
        inference_thread_status.resume()
        training_thread_status.resume()
        assert status_provider.get_current_status() is SystemStatus.ACTIVE

        thread_controller.shutdown()
        assert status_provider.get_current_status() is SystemStatus.SHUTTING_DOWN

        # Reset and try from PAUSED
        thread_controller.activate()
        thread_controller.pause()
        inference_thread_status.pause()
        training_thread_status.pause()
        assert status_provider.get_current_status() is SystemStatus.PAUSED

        thread_controller.shutdown()
        assert status_provider.get_current_status() is SystemStatus.SHUTTING_DOWN
