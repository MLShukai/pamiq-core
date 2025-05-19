import platform
import re
import sys

import pytest


def skip_if_platform_is_windows():
    return pytest.mark.skipif(
        sys.platform == "win32", reason=f"Platform is {platform.system()}"
    )


def skip_if_platform_is_darwin():
    return pytest.mark.skipif(
        sys.platform == "darwin", reason=f"Platform is {platform.system()}"
    )


def skip_if_kernel_is_linuxkit():
    return pytest.mark.skipif(
        "linuxkit" in platform.release(),
        reason=f"Linux kernel is linuxkit ({platform.release()})",
    )


def check_log_message(
    expected_log_message: str, log_level: str | None, caplog: pytest.LogCaptureFixture
):
    """Check if the expected log message is in the log messages.

    Args:
        expected_log_message: expected log message pattern string
        log_level: log level of the expected log message.
        caplog: caplog fixture.

    Raises:
        AssertionError: if the expected log message is not in the log messages of specified log level.
    """

    try:
        re.compile(expected_log_message)
    except re.error:
        expected_log_message = re.escape(expected_log_message)

    if log_level:
        error_level_log_messages = [
            record.message for record in caplog.records if record.levelname == log_level
        ]
    else:
        # if no log_level is specified, then check all log messages
        error_level_log_messages = [record.message for record in caplog.records]

    assert any(
        re.match(expected_log_message, message) for message in error_level_log_messages
    )
