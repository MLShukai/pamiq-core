import importlib
import sys

import pytest
from pytest_mock import MockerFixture


def test_torch_import_error(mocker: MockerFixture) -> None:
    """Test that ModuleNotFoundError is raised with the correct message when
    torch is not found."""
    # Mock sys.modules to simulate torch not being installed
    mocker.patch.dict(sys.modules, {"torch": None})
    # Import the module that imports torch
    with pytest.raises(ModuleNotFoundError) as excinfo:
        # Force reload to trigger the import error
        if "pamiq_core.torch" in sys.modules:
            del sys.modules["pamiq_core.torch"]
        importlib.import_module("pamiq_core.torch")

    # Check that the error message contains the pip install instruction
    error_message = str(excinfo.value)
    assert "torch module not found" in error_message
    assert "pip install pamiq-core[torch]" in error_message
