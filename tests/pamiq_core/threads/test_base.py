from typing import override

import pytest

from pamiq_core.threads import Thread, ThreadTypes
from tests.helpers import check_log_message


class TestThread:
    def test_init_with_THREAD_TYPE_attribute(self) -> None:
        """Test that the Thread class can be initialized with a THREAD_TYPE
        attribute."""

        class TestThreadWithThreadType(Thread):
            THREAD_TYPE = ThreadTypes.CONTROL

            @override
            def worker(self) -> None:
                pass

        _ = TestThreadWithThreadType()  # no error should be raised

    def test_init_without_THREAD_TYPE_attribute(self) -> None:
        """Test that the Thread class raises an error if THREAD_TYPE attribute
        is not defined."""

        class TestThreadWithoutThreadType(Thread):
            @override
            def worker(self) -> None:
                pass

        with pytest.raises(
            AttributeError,
            match="Subclasses must define `THREAD_TYPE` attribute before instantiation.",
        ):
            _ = TestThreadWithoutThreadType()

    def test_run_without_error(self, mocker) -> None:
        """Test that `run()` method executes worker() method."""

        class TestThreadWithoutError(Thread):
            THREAD_TYPE = ThreadTypes.CONTROL

            @override
            def worker(self) -> None:
                pass

        thread = TestThreadWithoutError()
        worker_spy = mocker.spy(thread, "worker")
        thread.run()
        worker_spy.assert_called_once_with()

    def test_run_with_error(self, caplog) -> None:
        """Test that `run()` raises exception and output logs if worker raises
        exception."""

        class TestThreadWithError(Thread):
            THREAD_TYPE = ThreadTypes.CONTROL

            @override
            def worker(self) -> None:
                raise RuntimeError("Test error")

        thread = TestThreadWithError()

        with pytest.raises(RuntimeError, match="Test error"):
            thread.run()

        # check the log message
        check_log_message(
            expected_log_message="An exception has occurred in 'control' thread.",
            log_level="ERROR",
            caplog=caplog,
        )
