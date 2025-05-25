import threading
import time

import pytest

from pamiq_core.thread import (
    ControllerCommandHandler,
    ReadOnlyController,
    ReadOnlyThreadStatus,
    ThreadController,
    ThreadStatus,
    ThreadStatusesMonitor,
    ThreadTypes,
)
from tests.helpers import (
    check_log_message,
    skip_if_platform_is_darwin,
    skip_if_platform_is_windows,
)


class TestThreadController:
    """A test class for ThreadController."""

    @pytest.fixture()
    def thread_controller(self) -> ThreadController:
        return ThreadController()

    def test_read_only_property(self, thread_controller: ThreadController) -> None:
        """Test that the read_only property returns a valid
        ReadOnlyController."""
        assert isinstance(thread_controller.read_only, ReadOnlyController)

    def test_initial_state(self, thread_controller: ThreadController) -> None:
        assert thread_controller.is_resume() is True
        assert thread_controller.is_active() is True

    def test_resume_and_related_predicate_methods(
        self, thread_controller: ThreadController
    ) -> None:
        thread_controller.resume()

        assert thread_controller.is_resume() is True
        assert thread_controller.is_pause() is False

    def test_resume_when_shutdown(self, thread_controller: ThreadController) -> None:
        thread_controller.shutdown()

        with pytest.raises(
            RuntimeError, match="ThreadController must be activated before resume()."
        ):
            thread_controller.resume()

    def test_pause_and_related_predicate_methods(
        self, thread_controller: ThreadController
    ) -> None:
        thread_controller.pause()

        assert thread_controller.is_resume() is False
        assert thread_controller.is_pause() is True

    def test_pause_when_shutdown(self, thread_controller: ThreadController) -> None:
        thread_controller.shutdown()

        with pytest.raises(
            RuntimeError, match="ThreadController must be activated before pause()."
        ):
            thread_controller.pause()

    def test_shutdown_and_related_predicate_methods_when_resume(
        self, thread_controller: ThreadController
    ) -> None:
        thread_controller.shutdown()

        assert thread_controller.is_shutdown() is True
        assert thread_controller.is_active() is False

    def test_shutdown_and_related_predicate_methods_when_pause(
        self, thread_controller: ThreadController
    ) -> None:
        thread_controller.pause()
        thread_controller.shutdown()

        assert thread_controller.is_shutdown() is True
        assert thread_controller.is_active() is False
        assert (
            thread_controller.is_resume() is True
        )  # `resume()` must be applied in `shutdown()`

    def test_activate_and_related_predicate_methods(
        self, thread_controller: ThreadController
    ) -> None:
        thread_controller.activate()

        assert thread_controller.is_shutdown() is False
        assert thread_controller.is_active() is True

    def test_shutdown_when_already_shutdown(
        self, thread_controller: ThreadController
    ) -> None:
        thread_controller.shutdown()

        # Test that `resume()` in `shutdown()` does not raise an error
        # when already shutdown.
        thread_controller.shutdown()

    def test_wait_for_resume_when_already_resumed(
        self, thread_controller: ThreadController
    ) -> None:
        # immediately return True if already resumed
        thread_controller.resume()
        start = time.perf_counter()
        assert thread_controller.wait_for_resume(timeout=0.1) is True
        assert time.perf_counter() - start < 1e-3

    @skip_if_platform_is_windows()
    @skip_if_platform_is_darwin()
    def test_wait_for_resume_when_already_paused(
        self, thread_controller: ThreadController
    ) -> None:
        # wait timeout and return False if paused
        thread_controller.pause()
        start = time.perf_counter()
        assert thread_controller.wait_for_resume(0.1) is False
        assert 0.1 <= time.perf_counter() - start < 0.2

    @skip_if_platform_is_windows()
    @skip_if_platform_is_darwin()
    def test_wait_for_resume_when_resumed_after_waiting(
        self, thread_controller: ThreadController
    ) -> None:
        # immediately return True if resumed after waiting
        thread_controller.pause()
        threading.Timer(0.1, thread_controller.resume).start()
        start = time.perf_counter()
        assert thread_controller.wait_for_resume(0.5) is True
        assert 0.1 <= time.perf_counter() - start < 0.2


class TestReadOnlyController:
    """A test class for ReadOnlyController."""

    def test_exposed_methods(self) -> None:
        thread_controller = ThreadController()
        read_only_controller = ReadOnlyController(thread_controller)

        assert read_only_controller.is_resume == thread_controller.is_resume
        assert read_only_controller.is_pause == thread_controller.is_pause
        assert read_only_controller.is_shutdown == thread_controller.is_shutdown
        assert read_only_controller.is_active == thread_controller.is_active
        assert read_only_controller.wait_for_resume == thread_controller.wait_for_resume


class TestControllerCommandHandler:
    """A test class for ControllerCommandHandler."""

    @pytest.fixture()
    def thread_controller(self) -> ThreadController:
        return ThreadController()

    @pytest.fixture()
    def read_only_controller(self, thread_controller) -> ReadOnlyController:
        return ReadOnlyController(thread_controller)

    @pytest.fixture()
    def handler(self, read_only_controller) -> ControllerCommandHandler:
        return ControllerCommandHandler(read_only_controller)

    def test_stop_if_pause_when_already_resumed(
        self,
        thread_controller: ThreadController,
        handler: ControllerCommandHandler,
        mocker,
    ) -> None:
        # prepare mock objects
        on_paused_callback_mock = mocker.spy(handler, "on_paused")
        on_resumed_callback_mock = mocker.spy(handler, "on_resumed")

        # immediately return if already resumed
        thread_controller.resume()
        start = time.perf_counter()
        handler.stop_if_pause()
        assert time.perf_counter() - start < 1e-3

        # callbacks are not called
        on_paused_callback_mock.assert_not_called()
        on_resumed_callback_mock.assert_not_called()

    @skip_if_platform_is_windows()
    @skip_if_platform_is_darwin()
    def test_stop_if_pause_pause_to_resume(
        self,
        thread_controller: ThreadController,
        handler: ControllerCommandHandler,
        mocker,
    ) -> None:
        # prepare mock objects
        on_paused_callback_mock = mocker.spy(handler, "on_paused")
        on_resumed_callback_mock = mocker.spy(handler, "on_resumed")

        # immediately return if resumed after waiting
        thread_controller.pause()
        threading.Timer(0.1, thread_controller.resume).start()
        start = time.perf_counter()
        handler.stop_if_pause()
        assert 0.1 <= time.perf_counter() - start < 0.2

        # callbacks are called
        on_paused_callback_mock.assert_called_once_with()
        on_resumed_callback_mock.assert_called_once_with()

    def test_stop_if_pause_when_already_shutdown(
        self, thread_controller: ThreadController, handler: ControllerCommandHandler
    ) -> None:
        # no test for callbacks since callbacks are only related to pause/resume
        # immediately return if already shutdown
        thread_controller.shutdown()
        start = time.perf_counter()
        handler.stop_if_pause()
        assert time.perf_counter() - start < 1e-3

    @skip_if_platform_is_windows()
    @skip_if_platform_is_darwin()
    def test_stop_if_pause_pause_to_shutdown(
        self, thread_controller: ThreadController, handler: ControllerCommandHandler
    ) -> None:
        # no test for callbacks since callbacks are only related to pause/resume
        # immediately return if shutdown after waiting
        thread_controller.pause()
        threading.Timer(0.1, thread_controller.shutdown).start()
        start = time.perf_counter()
        handler.stop_if_pause()
        assert 0.1 <= time.perf_counter() - start < 0.2

    def test_manage_loop_when_already_resumed(
        self, thread_controller: ThreadController, handler: ControllerCommandHandler
    ) -> None:
        # immediately return True if already resumed
        thread_controller.resume()
        start = time.perf_counter()
        assert handler.manage_loop() is True
        assert time.perf_counter() - start < 1e-3

    @skip_if_platform_is_windows()
    @skip_if_platform_is_darwin()
    def test_manage_loop_pause_to_resume(
        self, thread_controller: ThreadController, handler: ControllerCommandHandler
    ) -> None:
        # immediately return True if resumed after waiting
        thread_controller.pause()
        threading.Timer(0.1, thread_controller.resume).start()
        start = time.perf_counter()
        assert handler.manage_loop() is True
        assert 0.1 <= time.perf_counter() - start < 0.2

    def test_manage_loop_when_already_shutdown(
        self, thread_controller: ThreadController, handler: ControllerCommandHandler
    ) -> None:
        # immediately return False if already shutdown
        thread_controller.shutdown()
        start = time.perf_counter()
        assert handler.manage_loop() is False
        assert time.perf_counter() - start < 1e-3

    @skip_if_platform_is_windows()
    @skip_if_platform_is_darwin()
    def test_manage_loop_pause_to_shutdown(
        self, thread_controller: ThreadController, handler: ControllerCommandHandler
    ) -> None:
        # immediately return False if shutdown after waiting
        thread_controller.pause()
        threading.Timer(0.1, thread_controller.shutdown).start()
        start = time.perf_counter()
        assert handler.manage_loop() is False
        assert 0.1 <= time.perf_counter() - start < 0.2

    def test_manage_loop_with_pause_resume_shutdown(
        self, thread_controller: ThreadController, handler: ControllerCommandHandler
    ) -> None:
        counter = 0

        def inifinity_count():
            nonlocal counter
            while handler.manage_loop():
                counter += 1
                time.sleep(0.001)

        # increment occur if active & resume
        thread_controller.resume()
        thread = threading.Thread(target=inifinity_count)
        thread.start()
        time.sleep(0.01)
        assert counter > 0

        # increment does not occur if paused
        prev_count = counter
        thread_controller.pause()
        time.sleep(0.01)
        assert counter == prev_count

        # increment does not occur if shutdown (from when thread is paused)
        thread_controller.shutdown()
        time.sleep(0.01)
        thread.join()  # ensure the thread has finished
        assert counter == prev_count  # check that the loop has exited immediately

        # increment does not occur if shutdown (from when thread is resumed)
        thread_controller.activate()
        thread_controller.resume()
        thread = threading.Thread(target=inifinity_count)  # restart the thread
        thread.start()
        time.sleep(0.01)
        prev_count = counter
        thread_controller.shutdown()
        time.sleep(0.01)
        thread.join()  # ensure the thread has finished
        assert counter == prev_count  # check that the loop has exited immediately


class TestThreadStatus:
    """A test class for ThreadStatus."""

    @pytest.fixture()
    def thread_status(self) -> ThreadStatus:
        """Fixture for thread status."""
        return ThreadStatus()

    def test_read_only_property(self, thread_status: ThreadStatus) -> None:
        """Test that the read_only property returns a valid
        ReadOnlyThreadStatus."""
        assert isinstance(thread_status.read_only, ReadOnlyThreadStatus)

    def test_initial_state(self, thread_status: ThreadStatus) -> None:
        """Test initial state of thread status."""
        assert thread_status.is_pause() is False
        assert thread_status.is_resume() is True
        assert thread_status.is_exception_raised() is False

    def test_pause_and_related_predicate_methods(
        self, thread_status: ThreadStatus
    ) -> None:
        """Test pause and related predicate methods."""
        thread_status.pause()

        assert thread_status.is_pause() is True
        assert thread_status.is_resume() is False

    def test_resume_and_related_predicate_methods(
        self, thread_status: ThreadStatus
    ) -> None:
        """Test resume and related predicate methods."""
        thread_status.resume()

        assert thread_status.is_pause() is False
        assert thread_status.is_resume() is True

    def test_exception_raised_and_related_predicate_methods(
        self, thread_status: ThreadStatus
    ) -> None:
        """Test exception_raised and related predicate methods."""
        thread_status.exception_raised()

        assert thread_status.is_exception_raised() is True

    def test_wait_for_pause_when_already_paused(
        self, thread_status: ThreadStatus
    ) -> None:
        """Test wait_for_pause when the status is already paused."""
        # immediately return True if already paused
        thread_status.pause()
        start = time.perf_counter()
        assert thread_status.wait_for_pause(0.1) is True
        assert time.perf_counter() - start < 1e-3

    @skip_if_platform_is_windows()
    @skip_if_platform_is_darwin()
    def test_wait_for_pause_when_already_resumed(
        self, thread_status: ThreadStatus
    ) -> None:
        """Test wait_for_pause when the status is already resumed."""
        # wait timeout and return False if resumed
        thread_status.resume()
        start = time.perf_counter()
        assert thread_status.wait_for_pause(0.1) is False
        assert 0.1 <= time.perf_counter() - start < 0.2

    @skip_if_platform_is_windows()
    @skip_if_platform_is_darwin()
    def test_wait_for_pause_when_paused_after_waiting(
        self, thread_status: ThreadStatus
    ) -> None:
        """Test wait_for_pause when the status is resumed at first, and paused
        after waiting."""
        # immediately return True if paused after waiting
        thread_status.resume()
        threading.Timer(0.1, thread_status.pause).start()
        start = time.perf_counter()
        assert thread_status.wait_for_pause(0.5) is True
        assert 0.1 <= time.perf_counter() - start < 0.2


class TestReadOnlyThreadStatus:
    """A test class for ReadOnlyThreadStatus."""

    def test_exposed_methods(self) -> None:
        """Test of exposure of functions from ThreadStatus."""
        thread_status = ThreadStatus()
        read_only_thread_status = ReadOnlyThreadStatus(thread_status)
        assert read_only_thread_status.is_pause == thread_status.is_pause
        assert read_only_thread_status.is_resume == thread_status.is_resume
        assert (
            read_only_thread_status.is_exception_raised
            == thread_status.is_exception_raised
        )
        assert read_only_thread_status.wait_for_pause == thread_status.wait_for_pause


class TestThreadStatusesMonitor:
    """A test class for ThreadStatusesMonitor."""

    @pytest.fixture()
    def inference_thread_status(self) -> ThreadStatus:
        """Fixture for thread status, used for inference."""
        return ThreadStatus()

    @pytest.fixture()
    def read_only_inference_thread_status(
        self, inference_thread_status
    ) -> ReadOnlyThreadStatus:
        """Fixture for read-only thread status, used for inference."""
        return ReadOnlyThreadStatus(inference_thread_status)

    @pytest.fixture()
    def training_thread_status(self) -> ThreadStatus:
        """Fixture for thread status, used for training."""
        return ThreadStatus()

    @pytest.fixture()
    def read_only_training_thread_status(
        self, training_thread_status
    ) -> ReadOnlyThreadStatus:
        """Fixture for read-only thread status, used for training."""
        return ReadOnlyThreadStatus(training_thread_status)

    @pytest.fixture()
    def thread_statuses_monitor(
        self, read_only_inference_thread_status, read_only_training_thread_status
    ) -> ThreadStatusesMonitor:
        """Fixture for thread status monitor."""
        return ThreadStatusesMonitor(
            {
                ThreadTypes.INFERENCE: read_only_inference_thread_status,
                ThreadTypes.TRAINING: read_only_training_thread_status,
            }
        )

    def test_wait_for_all_threads_pause_when_empty_status(self) -> None:
        """Test wait_for_all_threads_pause when statuses is empty."""
        # immediately return True if statuses is empty
        thread_statuses_monitor = ThreadStatusesMonitor(statuses={})
        start = time.perf_counter()
        assert thread_statuses_monitor.wait_for_all_threads_pause(0.1) is True
        assert time.perf_counter() - start < 1e-3

    def test_wait_for_all_threads_pause_all_when_all_threads_paused(
        self, inference_thread_status, training_thread_status, thread_statuses_monitor
    ) -> None:
        """Test wait_for_all_threads_pause when all threads are paused."""
        # immediately return True if all threads are paused
        inference_thread_status.pause()
        training_thread_status.pause()

        start = time.perf_counter()
        assert thread_statuses_monitor.wait_for_all_threads_pause(0.1) is True
        assert time.perf_counter() - start < 1e-2  # test not passed if 1e-3

    @skip_if_platform_is_windows()
    @skip_if_platform_is_darwin()
    @pytest.mark.parametrize(
        "is_inference_resumed, is_training_resumed",
        [
            (True, False),
            (False, True),
            (True, True),
        ],
    )
    def test_wait_for_all_threads_pause_when_some_threads_resumed(
        self,
        caplog,
        is_inference_resumed,
        is_training_resumed,
        inference_thread_status,
        training_thread_status,
        thread_statuses_monitor,
    ) -> None:
        """Test wait_for_all_threads_pause when some threads are resumed."""
        # wait timeout and return False if some threads are resumed
        inference_thread_status.pause()
        training_thread_status.pause()

        if is_inference_resumed:
            inference_thread_status.resume()
        if is_training_resumed:
            training_thread_status.resume()

        start = time.perf_counter()
        assert thread_statuses_monitor.wait_for_all_threads_pause(0.1) is False
        assert 0.1 <= time.perf_counter() - start < 0.2

        # check log messages
        if is_inference_resumed:
            check_log_message(
                expected_log_message="Timeout waiting for 'inference' thread to pause after 0.1 seconds.",
                log_level="ERROR",
                caplog=caplog,
            )
        if is_training_resumed:
            check_log_message(
                expected_log_message="Timeout waiting for 'training' thread to pause after 0.1 seconds.",
                log_level="ERROR",
                caplog=caplog,
            )

    @skip_if_platform_is_windows()
    @skip_if_platform_is_darwin()
    @pytest.mark.parametrize(
        "is_inference_resumed, is_training_resumed",
        [
            (True, False),
            (False, True),
            (True, True),
        ],
    )
    def test_wait_for_all_threads_pause_all_when_paused_after_waiting(
        self,
        is_inference_resumed,
        is_training_resumed,
        inference_thread_status,
        training_thread_status,
        thread_statuses_monitor,
    ) -> None:
        """Test wait_for_all_threads_pause when all threads are paused after
        waiting."""
        # immediately return True if all threads are paused after waiting
        inference_thread_status.pause()
        training_thread_status.pause()

        if is_inference_resumed:
            inference_thread_status.resume()
            threading.Timer(0.1, inference_thread_status.pause).start()
        if is_training_resumed:
            training_thread_status.resume()
            threading.Timer(0.1, training_thread_status.pause).start()

        start = time.perf_counter()
        assert thread_statuses_monitor.wait_for_all_threads_pause(0.5) is True
        assert 0.1 <= time.perf_counter() - start < 0.2

    @pytest.mark.parametrize(
        "is_inference_exception_raised, is_training_exception_raised",
        [
            (False, False),
            (True, False),
            (False, True),
            (True, True),
        ],
    )
    def test_check_exception_raised(
        self,
        caplog,
        is_inference_exception_raised,
        is_training_exception_raised,
        inference_thread_status,
        training_thread_status,
        thread_statuses_monitor,
    ) -> None:
        """Test check_exception_raised: return value and log messages."""
        if is_inference_exception_raised:
            inference_thread_status.exception_raised()
        if is_training_exception_raised:
            training_thread_status.exception_raised()

        assert thread_statuses_monitor.check_exception_raised() is any(
            [is_inference_exception_raised, is_training_exception_raised]
        )

        # check log messages
        if is_inference_exception_raised:
            check_log_message(
                expected_log_message="An exception has occurred in the 'inference' thread.",
                log_level="ERROR",
                caplog=caplog,
            )
        if is_training_exception_raised:
            check_log_message(
                expected_log_message="An exception has occurred in the 'training' thread.",
                log_level="ERROR",
                caplog=caplog,
            )

    def test_check_all_threads_paused_empty_statuses(self) -> None:
        """Test check_all_threads_paused when statuses is empty."""
        thread_statuses_monitor = ThreadStatusesMonitor(statuses={})
        assert thread_statuses_monitor.check_all_threads_paused() is True

    @pytest.mark.parametrize(
        "is_inference_paused, is_training_paused, expected_result",
        [
            (False, False, False),  # No threads paused
            (True, False, False),  # Some threads paused
            (False, True, False),  # Some threads paused
            (True, True, True),  # All threads paused
        ],
    )
    def test_check_all_threads_paused(
        self,
        is_inference_paused,
        is_training_paused,
        expected_result,
        inference_thread_status,
        training_thread_status,
        thread_statuses_monitor,
    ) -> None:
        """Test check_all_threads_paused with different thread pause states."""
        # Set initial state to resumed
        inference_thread_status.resume()
        training_thread_status.resume()

        # Set the specified pause states
        if is_inference_paused:
            inference_thread_status.pause()
        if is_training_paused:
            training_thread_status.pause()

        assert thread_statuses_monitor.check_all_threads_paused() is expected_result

    def test_check_any_threads_paused_empty_statuses(self) -> None:
        """Test check_any_threads_paused when statuses is empty."""
        thread_statuses_monitor = ThreadStatusesMonitor(statuses={})
        assert thread_statuses_monitor.check_any_threads_paused() is False

    @pytest.mark.parametrize(
        "is_inference_paused, is_training_paused, expected_result",
        [
            (False, False, False),  # No threads paused
            (True, False, True),  # Some threads paused
            (False, True, True),  # Some threads paused
            (True, True, True),  # All threads paused
        ],
    )
    def test_check_any_threads_paused(
        self,
        is_inference_paused,
        is_training_paused,
        expected_result,
        inference_thread_status,
        training_thread_status,
        thread_statuses_monitor,
    ) -> None:
        """Test check_any_threads_paused with different thread pause states."""
        # Set initial state to resumed
        inference_thread_status.resume()
        training_thread_status.resume()

        # Set the specified pause states
        if is_inference_paused:
            inference_thread_status.pause()
        if is_training_paused:
            training_thread_status.pause()

        assert thread_statuses_monitor.check_any_threads_paused() is expected_result
