from typing import Any


def are_dict_values_same_entities(x: Any, y: Any) -> bool:
    if isinstance(x, dict) and isinstance(y, dict):
        if x.keys() != y.keys():
            return False
        return all(are_dict_values_same_entities(x[key], y[key]) for key in x)
    elif isinstance(x, list) and isinstance(y, list):
        if len(x) != len(y):
            return False
        return all(
            are_dict_values_same_entities(element1, element2)
            for element1, element2 in zip(x, y)
        )
    else:
        return x is y
