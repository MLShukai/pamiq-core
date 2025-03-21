import copy
from threading import RLock
from typing import Any, Protocol, override

import torch
import torch.nn as nn

from pamiq_core.model import InferenceModel, TrainingModel


def test_get_device() -> None:
    pass


def test_default_infer_procedure() -> None:
    pass


class TestTorchInferenceModel:
    def test_raw_model(self) -> None:
        pass

    def test_infer(self) -> None:
        pass


class TestTorchTrainingModel:
    def test_create_inference(self) -> None:
        pass

    def sync_impl(self) -> None:
        pass

    def forward(self) -> None:
        pass
