import platform

import pytest


def skip_if_platform_is_not_linux():
    return pytest.mark.skipif(
        platform.system() != "Linux", reason="Platform is not linux."
    )
