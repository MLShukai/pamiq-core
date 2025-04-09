import platform
import re
from pathlib import Path

import pytest


def skip_if_platform_is_not_linux():
    return pytest.mark.skipif(
        platform.system() != "Linux", reason="Platform is not linux."
    )


def skip_if_kernel_is_linuxkit():
    osrelease = Path("/proc/sys/kernel/osrelease")
    skip = False
    if osrelease.is_file() and "linuxkit" in osrelease.read_text():
        skip = True

    return pytest.mark.skipif(skip, reason="Linux kernel is linuxkit.")


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
