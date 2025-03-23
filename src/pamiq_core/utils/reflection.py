def get_class_module_path(cls: type) -> str:
    """Get the module path of a class."""
    return f"{cls.__module__}.{cls.__name__}"
