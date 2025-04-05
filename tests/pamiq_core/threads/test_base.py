from typing import override
from unittest.mock import call

import pytest

from pamiq_core.threads import Thread, ThreadTypes
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
        spy_logger_info = mocker.spy(
            thread._logger, "info"
        )  # INFO level logs are not being emitted thus test with `assert_has_calls()`.

        thread.run()

        # Check method calls
        spy_on_start.assert_called_once_with()
        assert spy_on_tick.call_count == 3
        spy_on_end.assert_called_once_with()
        spy_on_exception.assert_not_called()  # not called
        spy_on_finally.assert_called_once_with()

        # Check log calls
        spy_logger_info.assert_has_calls(
            [
                call("Start 'control' thread."),
                call("End 'control' thread."),
            ]
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
        spy_logger_info = mocker.spy(
            thread._logger, "info"
        )  # INFO level logs are not being emitted thus test with `assert_has_calls()`.

        with pytest.raises(RuntimeError, match="Test exception"):
            thread.run()

        # Check method calls
        spy_on_start.assert_called_once_with()
        spy_on_tick.assert_called_once_with()
        spy_on_end.assert_not_called()  # not called
        spy_on_exception.assert_called_once_with()
        spy_on_finally.assert_called_once_with()

        # Check log messages and calls
        check_log_message(
            expected_log_message="An exception has occurred in 'control' thread.",
            log_level="ERROR",
            caplog=caplog,
        )
        spy_logger_info.assert_has_calls(
            [
                call("Start 'control' thread."),
                call("End 'control' thread."),
            ]
        )
