from typing import override
from unittest.mock import call

import pytest

from pamiq_core.threads import (
    BackgroundThread,
    ControllerCommandHandler,
    ReadOnlyController,
    ReadOnlyThreadStatus,
    Thread,
    ThreadController,
    ThreadTypes,
)
from tests.helpers import check_log_message


class TestThread:
    def test_init_with_THREAD_TYPE_attribute(self) -> None:
        """Test that the Thread class can be initialized with a THREAD_TYPE
        attribute."""

        class TestThreadWithThreadType(Thread):
            THREAD_TYPE = ThreadTypes.CONTROL

        _ = TestThreadWithThreadType()  # no error should be raised

    def test_init_without_THREAD_TYPE_attribute(self) -> None:
        """Test that the Thread class raises an error if THREAD_TYPE attribute
        is not defined."""

        class TestThreadWithoutThreadType(Thread):
            # class without THREAD_TYPE attribute
            pass

        with pytest.raises(
            AttributeError,
            match="Subclasses must define `THREAD_TYPE` attribute before instantiation.",
        ):
            _ = TestThreadWithoutThreadType()

    def test_run_without_exception(self, caplog, mocker) -> None:
        """Test that the run method works without exceptions."""

        class TestThreadWithoutException(Thread):
            THREAD_TYPE = ThreadTypes.CONTROL

            def __init__(self) -> None:
                super().__init__()
                self._counter = 0

            @override
            def is_running(self) -> bool:
                # rerutn True for 3 times and then False
                self._counter += 1
                return self._counter <= 3

        thread = TestThreadWithoutException()

        # set the spy on the methods
        spy_on_start = mocker.spy(thread, "on_start")
        spy_on_tick = mocker.spy(thread, "on_tick")
        spy_on_end = mocker.spy(thread, "on_end")
        spy_on_exception = mocker.spy(thread, "on_exception")
        spy_on_finally = mocker.spy(thread, "on_finally")

        thread.run()

        # Check method calls and log messages
        check_log_message(
            expected_log_message="Start 'control' thread.",
            log_level="INFO",
            caplog=caplog,
        )
        spy_on_start.assert_called_once_with()
        assert spy_on_tick.call_count == 3
        spy_on_end.assert_called_once_with()
        spy_on_exception.assert_not_called()  # not called
        spy_on_finally.assert_called_once_with()
        check_log_message(
            expected_log_message="End 'control' thread.",
            log_level="INFO",
            caplog=caplog,
        )

    def test_run_with_exception(self, caplog, mocker) -> None:
        """Test that the run method works with exceptions."""

        class TestThreadWithException(Thread):
            THREAD_TYPE = ThreadTypes.CONTROL

            @override
            def on_tick(self) -> None:
                raise RuntimeError("Test exception")

        thread = TestThreadWithException()

        # set the spy on the methods
        spy_on_start = mocker.spy(thread, "on_start")
        spy_on_tick = mocker.spy(thread, "on_tick")
        spy_on_end = mocker.spy(thread, "on_end")
        spy_on_exception = mocker.spy(thread, "on_exception")
        spy_on_finally = mocker.spy(thread, "on_finally")

        with pytest.raises(RuntimeError, match="Test exception"):
            thread.run()

        # Check method calls and log messages
        check_log_message(
            expected_log_message="Start 'control' thread.",
            log_level="INFO",
            caplog=caplog,
        )
        spy_on_start.assert_called_once_with()
        spy_on_tick.assert_called_once_with()
        spy_on_end.assert_not_called()  # not called
        check_log_message(
            expected_log_message="An exception has occurred in 'control' thread.",
            log_level="ERROR",
            caplog=caplog,
        )
        spy_on_exception.assert_called_once_with()
        spy_on_finally.assert_called_once_with()
        check_log_message(
            expected_log_message="An exception has occurred in 'control' thread.",
            log_level="ERROR",
            caplog=caplog,
        )

        # Check log messages and calls


# This class is defined here since it is used in the fixtures in the TestBackgroundThread
class ValidBackgroundThread(BackgroundThread):
    """A valid BackgroundThread subclass for testing purposes."""

    THREAD_TYPE = ThreadTypes.INFERENCE

    def __init__(self) -> None:
        super().__init__()


class TestBackgroundThread:
    @pytest.fixture
    def background_thread(self) -> ValidBackgroundThread:
        """Fixture to create a valid BackgroundThread instance."""
        return ValidBackgroundThread()

    @pytest.fixture
    def thread_controller(self) -> ThreadController:
        """Fixture to create a thread controller."""
        return ThreadController()

    @pytest.fixture
    def read_only_controller(self, thread_controller) -> ReadOnlyController:
        """Fixture to create a read-only controller."""
        return thread_controller.read_only

    @pytest.fixture
    def controller_command_handler(
        self, read_only_controller
    ) -> ControllerCommandHandler:
        """Fixture to create a ControllerCommandHandler instance."""
        return ControllerCommandHandler(read_only_controller)

    @pytest.fixture
    def background_thread_with_controller(
        self, background_thread, read_only_controller
    ) -> ValidBackgroundThread:
        """Fixture to create a BackgroundThread with a controller."""
        background_thread.attach_controller(read_only_controller)
        return background_thread

    def test_init_with_THREAD_TYPE_attribute_other_than_control(self) -> None:
        """Test that the Thread class can be initialized with a THREAD_TYPE
        attribute other than 'control'."""

        class TestThreadWithAppropriateThreadType(BackgroundThread):
            THREAD_TYPE = ThreadTypes.TRAINING

            def __init__(self) -> None:
                super().__init__()

        _ = TestThreadWithAppropriateThreadType()  # no error should be raised

    def test_init_with_control_THREAD_TYPE(self) -> None:
        """Test that the Thread class can NOT be initialized with a 'control'
        THREAD_TYPE attribute."""

        class TestThreadWithControlThreadType(BackgroundThread):
            THREAD_TYPE = ThreadTypes.CONTROL

            def __init__(self) -> None:
                super().__init__()

        with pytest.raises(
            ValueError,
            match="BackgroundThread cannot be of type 'control'.",
        ):
            _ = TestThreadWithControlThreadType()

    def test_thread_status(self, background_thread) -> None:
        """Test that the thread status returns ReadOnlyThreadStatus
        instance."""
        assert isinstance(background_thread.thread_status, ReadOnlyThreadStatus)

    def test_attach_controller(self, background_thread, read_only_controller) -> None:
        """Test that the controller can be attached to the thread."""

        background_thread.attach_controller(read_only_controller)
        assert isinstance(
            background_thread._controller_command_handler, ControllerCommandHandler
        )

    def test_on_paused(self, background_thread, mocker) -> None:
        """Test that the on_paused method changes thread status."""
        background_thread.on_paused()

        assert background_thread.thread_status.is_pause() is True

    def test_on_resumed(self, background_thread, mocker) -> None:
        """Test that the on_resumed method changes thread status."""

        background_thread.on_resumed()

        assert background_thread.thread_status.is_resume() is True

    def test_start(self, background_thread, mocker) -> None:
        """Test that the start method actually starts the thread."""
        spy_thread_start = mocker.spy(background_thread._thread, "start")
        _ = mocker.patch.object(
            background_thread, "is_running", return_value=False
        )  # make this to finish the `run()` method

        background_thread.start()

        spy_thread_start.assert_called_once_with()

    def test_join(self, background_thread, mocker) -> None:
        """Test that the join method actually joins the thread."""
        spy_thread_join = mocker.spy(background_thread._thread, "join")

        background_thread.start()
        background_thread.join()

        spy_thread_join.assert_called_once_with()

    def test_is_alive_when_alive(self, background_thread, mocker) -> None:
        """Test that the is_alive returns True when the thread is alive."""
        mocker.patch.object(background_thread._thread, "is_alive", return_value=True)

        assert background_thread.is_alive() is True

    def test_is_alive_when_not_alive(self, background_thread, mocker) -> None:
        """Test that the is_alive returns False when the thread is not
        alive."""
        mocker.patch.object(background_thread._thread, "is_alive", return_value=False)

        assert background_thread.is_alive() is False

    def test_is_running_when_manage_loop_true(
        self, background_thread_with_controller, mocker
    ) -> None:
        """Test that the is_running returns True when
        `controller_command_handler.manage_loop()` is True."""
        _ = mocker.patch.object(
            background_thread_with_controller._controller_command_handler,
            "manage_loop",
            return_value=True,
        )

        assert background_thread_with_controller.is_running() is True

    def test_is_running_when_manage_loop_false(
        self, background_thread_with_controller, mocker
    ) -> None:
        """Test that the is_running returns False when
        `controller_command_handler.manage_loop()` is False."""
        _ = mocker.patch.object(
            background_thread_with_controller._controller_command_handler,
            "manage_loop",
            return_value=False,
        )

        assert background_thread_with_controller.is_running() is False

    def test_on_exception(self, background_thread, mocker) -> None:
        """Test that the on_exception method changes thread status."""

        background_thread.on_exception()

        assert background_thread.thread_status.is_exception_raised() is True
