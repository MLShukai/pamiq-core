from pamiq_core.utils import get_class_module_path


def test_get_class_module_path():
    class TestClass:
        pass

    assert (
        get_class_module_path(TestClass)
        == "tests.pamiq_core.utils.test_reflection.TestClass"
    )
    assert (
        get_class_module_path(int) == "builtins.int"
    )  # "builtins" is the module name for built-in objects
