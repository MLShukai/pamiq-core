def get_class_module_path(cls: type) -> str:
    """Get the module path of a class.

    Args:
        cls: The class to get the module path.

    Returns:
        str: The module path of the class.
    """
    return f"{cls.__module__}.{cls.__name__}"
