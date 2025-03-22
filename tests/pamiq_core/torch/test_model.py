import copy
from threading import RLock
from typing import Any, Protocol, override

import pytest
import torch
import torch.nn as nn
from pytest_mock import MockerFixture

from pamiq_core.model import InferenceModel, TrainingModel
from pamiq_core.torch import (
    TorchInferenceModel,
    TorchTrainingModel,
    default_infer_procedure,
)


def test_get_device() -> None:
    pass


def test_default_infer_procedure() -> None:
    pass


class TestTorchInferenceModel:
    @pytest.fixture
    def model(self) -> nn.Module:
        return nn.Linear(3, 5)

    @pytest.fixture
    def torch_inference_model(self, model: nn.Module) -> TorchInferenceModel:
        torch_inference_model = TorchInferenceModel(model, default_infer_procedure)
        return torch_inference_model

    def test_raw_model(
        self, torch_inference_model: TorchInferenceModel, model: nn.Module
    ) -> None:
        assert model is torch_inference_model.raw_model

    def test_infer(
        self, torch_inference_model: TorchInferenceModel, model: nn.Module
    ) -> None:
        input_tensor = torch.randn([2, 3])
        output_tensor = torch_inference_model.infer(input_tensor)
        expected_tensor = model(input_tensor)
        assert torch.equal(output_tensor, expected_tensor)


class TestTorchTrainingModel:
    def test_create_inference(self) -> None:
        pass

    def test_sync_impl(self) -> None:
        pass

    def test_forward(self) -> None:
        pass
