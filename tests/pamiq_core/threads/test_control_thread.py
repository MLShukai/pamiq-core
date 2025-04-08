import pytest
from pytest_mock import MockerFixture

from pamiq_core.threads import (
    ControlThread,
    ReadOnlyController,
    ReadOnlyThreadStatus,
    ThreadController,
    ThreadStatus,
    ThreadStatusesMonitor,
    ThreadTypes,
)
from tests.helpers import check_log_message


class TestControlThread:
    """Test suite for the ControlThread class."""

    @pytest.fixture
    def control_thread(self) -> ControlThread:
        """Fixture for a basic ControlThread instance."""
        return ControlThread(
            timeout_for_all_threads_pause=0.1, max_attempts_to_pause_all_threads=2
        )

    @pytest.fixture
    def inference_thread_status(self) -> ThreadStatus:
        """Fixture for an inference thread status."""
        return ThreadStatus()

    @pytest.fixture
    def training_thread_status(self) -> ThreadStatus:
        """Fixture for a training thread status."""
        return ThreadStatus()

    @pytest.fixture
    def thread_statuses(
        self,
        inference_thread_status: ThreadStatus,
        training_thread_status: ThreadStatus,
    ) -> dict[ThreadTypes, ReadOnlyThreadStatus]:
        """Fixture for a dictionary of thread statuses."""
        return {
            ThreadTypes.INFERENCE: inference_thread_status.read_only,
            ThreadTypes.TRAINING: training_thread_status.read_only,
        }

    @pytest.fixture
    def control_thread_with_statuses(
        self,
        control_thread: ControlThread,
        thread_statuses: dict[ThreadTypes, ReadOnlyThreadStatus],
    ) -> ControlThread:
        """Fixture for a ControlThread with attached thread statuses."""
        control_thread.attach_thread_statuses(thread_statuses)
        return control_thread

    def test_init_default_state(self, control_thread: ControlThread) -> None:
        """Test that ControlThread initializes with the correct default
        state."""
        # Verify controller is available
        assert isinstance(control_thread.controller, ReadOnlyController)

        # Verify initial running state
        assert control_thread.is_running() is True

    def test_controller_property(self, control_thread: ControlThread) -> None:
        """Test that the controller property returns a valid
        ReadOnlyController."""
        controller = control_thread.controller
        assert isinstance(controller, ReadOnlyController)

    def test_try_pause_success(
        self,
        control_thread_with_statuses: ControlThread,
        mocker: MockerFixture,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test successful thread pausing."""
        # Mock the time.pause method
        mock_time_pause = mocker.patch("pamiq_core.time.pause")

        # Set up for successful pause - prepare threads to pause when signaled
        mocker.patch.object(
            ThreadStatusesMonitor, "wait_for_all_threads_pause", return_value=True
        )

        # Attempt to pause
        result = control_thread_with_statuses.try_pause()

        # Verify results
        assert result is True

        # Verify time was paused
        mock_time_pause.assert_called_once()

        # Verify controller is in pause state
        assert control_thread_with_statuses.controller.is_pause() is True

        # Verify log messages
        check_log_message("Trying to pause...", "INFO", caplog)
        check_log_message(
            "Success to pause the all background threads.", "INFO", caplog
        )

    def test_try_pause_already_paused(
        self,
        control_thread_with_statuses: ControlThread,
        mocker: MockerFixture,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test try_pause when system is already paused."""
        # Set up controller to return True for is_pause
        control_thread_with_statuses._controller.pause()

        # Spy on ThreadStatusesMonitor.wait_for_all_threads_pause
        spy_wait = mocker.patch.object(
            ThreadStatusesMonitor, "wait_for_all_threads_pause"
        )

        # Call try_pause
        result = control_thread_with_statuses.try_pause()

        # Verify results
        assert result is True

        # Verify wait_for_all_threads_pause was not called
        spy_wait.assert_not_called()

        # Verify log message
        check_log_message("System has already been paused.", "INFO", caplog)

    def test_try_pause_failure(
        self,
        control_thread_with_statuses: ControlThread,
        mocker: MockerFixture,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test unsuccessful thread pausing."""
        # Mock the wait_for_all_threads_pause method to return False
        mocker.patch.object(
            ThreadStatusesMonitor, "wait_for_all_threads_pause", return_value=False
        )

        # Mock time.pause to check it's not called
        mock_time_pause = mocker.patch("pamiq_core.time.pause")

        # Attempt to pause
        result = control_thread_with_statuses.try_pause()

        # Verify results
        assert result is False

        # Verify time was not paused
        mock_time_pause.assert_not_called()

        # Verify log messages
        check_log_message("Trying to pause...", "INFO", caplog)
        check_log_message(
            "Failed to pause the background threads in timeout 0.1 seconds.",
            "WARNING",
            caplog,
        )
        check_log_message("Attempting retry 1 / 2 ...", "WARNING", caplog)
        check_log_message("Attempting retry 2 / 2 ...", "WARNING", caplog)
        check_log_message("Failed to pause... ", "ERROR", caplog)

    def test_resume(
        self,
        control_thread_with_statuses: ControlThread,
        mocker: MockerFixture,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test resuming threads."""
        # Mock time.resume
        mock_time_resume = mocker.patch("pamiq_core.time.resume")

        # Spy on the internal controller's resume method
        spy_controller_resume = mocker.spy(ThreadController, "resume")

        # Call resume
        control_thread_with_statuses.resume()

        # Verify time was resumed
        mock_time_resume.assert_called_once()

        # Verify controller resume was called
        assert spy_controller_resume.call_count == 1

        # Verify log message
        check_log_message("Resuming...", "INFO", caplog)

    def test_shutdown(
        self, control_thread: ControlThread, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test shutting down the control thread."""
        # Verify initial state
        assert control_thread.is_running() is True

        # Call shutdown
        control_thread.shutdown()

        # Verify state after shutdown
        assert control_thread.is_running() is False
        assert control_thread.controller.is_shutdown() is True

        # Verify log message
        check_log_message("Shutting down...", "INFO", caplog)

    def test_on_tick_with_no_exceptions(
        self, control_thread_with_statuses: ControlThread, mocker: MockerFixture
    ) -> None:
        """Test on_tick when no exceptions are detected."""
        # Mock check_exception_raised to return False
        mocker.patch.object(
            ThreadStatusesMonitor, "check_exception_raised", return_value=False
        )

        # Spy on shutdown
        spy_shutdown = mocker.spy(control_thread_with_statuses, "shutdown")

        # Call on_tick
        control_thread_with_statuses.on_tick()

        # Verify shutdown was not called
        spy_shutdown.assert_not_called()

        # Verify thread still running
        assert control_thread_with_statuses.is_running() is True

    def test_on_tick_with_exceptions(
        self,
        control_thread_with_statuses: ControlThread,
        mocker: MockerFixture,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test on_tick when exceptions are detected."""
        # Mock check_exception_raised to return True
        mocker.patch.object(
            ThreadStatusesMonitor, "check_exception_raised", return_value=True
        )

        # Call on_tick
        control_thread_with_statuses.on_tick()

        # Verify thread is no longer running
        assert control_thread_with_statuses.is_running() is False

        # Verify controller is in shutdown state
        assert control_thread_with_statuses.controller.is_shutdown() is True

        # Verify log messages
        check_log_message(
            "An exception occurred. The system will terminate immediately.",
            "ERROR",
            caplog,
        )
        check_log_message("Shutting down...", "INFO", caplog)

    def test_on_finally(
        self, control_thread: ControlThread, mocker: MockerFixture
    ) -> None:
        """Test that on_finally calls shutdown."""
        # Spy on shutdown
        spy_shutdown = mocker.spy(control_thread, "shutdown")

        # Call on_finally
        control_thread.on_finally()

        # Verify shutdown was called
        spy_shutdown.assert_called_once()

    def test_on_paused(
        self, control_thread: ControlThread, mocker: MockerFixture
    ) -> None:
        """Test that on_paused calls time.pause."""
        # Mock time.pause
        mock_time_pause = mocker.patch("pamiq_core.time.pause")

        # Call on_paused
        control_thread.on_paused()

        # Verify time was paused
        mock_time_pause.assert_called_once()

    def test_on_resumed(
        self, control_thread: ControlThread, mocker: MockerFixture
    ) -> None:
        """Test that on_resumed calls time.resume."""
        # Mock time.resume
        mock_time_resume = mocker.patch("pamiq_core.time.resume")

        # Call on_resumed
        control_thread.on_resumed()

        # Verify time was resumed
        mock_time_resume.assert_called_once()
