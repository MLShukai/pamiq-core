import platform
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
