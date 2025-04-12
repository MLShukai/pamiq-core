from typing import Any, override

import pytest
import torch

from .are_dict_values_same_entities import are_dict_values_same_entities


def test_same_dicts_with_same_pointers():
    shared = {"a": 1}
    d1 = {"key": shared}
    d2 = {"key": shared}
    assert are_dict_values_same_entities(d1, d2) is True


def test_same_dicts_with_different_pointers_and_same_values():
    d1 = {"key": {"a": 1}}
    d2 = {"key": {"a": 1}}
    assert are_dict_values_same_entities(d1, d2) is True


def test_different_keys():
    d1 = {"a": 1}
    d2 = {"b": 1}
    assert are_dict_values_same_entities(d1, d2) is False


def test_different_lengths_of_lists():
    d1 = {"a": [1, 2]}
    d2 = {"a": [1, 2, 3]}
    assert are_dict_values_same_entities(d1, d2) is False


def test_tensor_same_pointer():
    t = torch.tensor([1.0, 2.0])
    d1 = {"a": t}
    d2 = {"a": t}
    assert are_dict_values_same_entities(d1, d2) is True


def test_tensor_different_pointers_and_same_value():
    d1 = {"a": torch.tensor([1.0, 2.0])}
    d2 = {"a": torch.tensor([1.0, 2.0])}
    assert are_dict_values_same_entities(d1, d2) is False


def test_tensor_in_list_same_pointer():
    t = torch.tensor([1.0, 2.0])
    d1 = {"a": [t]}
    d2 = {"a": [t]}
    assert are_dict_values_same_entities(d1, d2) is True


def test_tensor_in_list_different_pointers_and_same_value():
    d1 = {"a": [torch.tensor([1.0, 2.0])]}
    d2 = {"a": [torch.tensor([1.0, 2.0])]}
    assert are_dict_values_same_entities(d1, d2) is False


def test_tensor_in_nested_dict_same_pointer():
    t = torch.tensor([1.0])
    d1 = {"a": {"b": t}}
    d2 = {"a": {"b": t}}
    assert are_dict_values_same_entities(d1, d2) is True


def test_tensor_in_nested_dict_different_pointer_and_same_value():
    d1 = {"a": {"b": torch.tensor([1.0])}}
    d2 = {"a": {"b": torch.tensor([1.0])}}
    assert are_dict_values_same_entities(d1, d2) is False
