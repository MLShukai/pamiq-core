from pamiq_core.utils.reflection import get_class_module_path


class TestClass:
    """A class used only for testing purposes."""

    pass


def test_get_class_module_path() -> None:
    """Test the get_class_module_path function."""

    assert (
        get_class_module_path(TestClass)
        == "tests.pamiq_core.utils.test_reflection.TestClass"
    )
    assert (
        get_class_module_path(int) == "builtins.int"
    )  # "builtins" is the module name for built-in objects
