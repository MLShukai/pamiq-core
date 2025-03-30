import copy
from threading import RLock
from typing import Any, Protocol, override

import pytest
import torch
import torch.nn as nn

from pamiq_core.model import InferenceModel, TrainingModel
from pamiq_core.torch import (
    TorchInferenceModel,
    TorchTrainingModel,
    default_infer_procedure,
)


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
        assert model is torch_inference_model._raw_model

    def test_infer(
        self, torch_inference_model: TorchInferenceModel, model: nn.Module
    ) -> None:
        input_tensor = torch.randn([2, 3])
        output_tensor = torch_inference_model.infer(input_tensor)
        expected_tensor = model(input_tensor)
        assert torch.equal(output_tensor, expected_tensor)


@pytest.mark.parametrize(
    ["has_inference_model", "inference_thread_only"],
    [[True, True], [True, False], [False, False]],
)
class TestTorchTrainingModel:
    @pytest.fixture
    def model(self) -> nn.Module:
        return nn.Linear(3, 5)

    @pytest.fixture
    def torch_training_model(
        self, model: nn.Module, has_inference_model: bool, inference_thread_only: bool
    ) -> TorchTrainingModel:
        torch_training_model = TorchTrainingModel(
            model, has_inference_model, inference_thread_only
        )
        return torch_training_model

    def test_create_inference(
        self,
        torch_training_model: TorchTrainingModel,
        has_inference_model: bool,
        inference_thread_only: bool,
    ) -> None:
        if has_inference_model:
            torch_inference_model = torch_training_model.inference_model
            # check if the internal modelsz are same
            assert torch.equal(
                torch_training_model.model.weight,
                torch_inference_model._raw_model.weight,
            )
            # check about pointers
            if inference_thread_only:
                assert torch_training_model.model is torch_inference_model._raw_model
            else:
                assert (
                    torch_training_model.model is not torch_inference_model._raw_model
                )

    def test_forward(
        self,
        model: nn.Module,
        torch_training_model: TorchTrainingModel,
        has_inference_model: bool,
        inference_thread_only: bool,
    ) -> None:
        input_tensor = torch.randn([2, 3])
        output_tensor = torch_training_model.forward(input_tensor)
        expected_tensor = model(input_tensor)
        assert torch.equal(output_tensor, expected_tensor)

    def test_sync_impl(
        self,
        torch_training_model: TorchTrainingModel,
        has_inference_model: bool,
        inference_thread_only: bool,
    ) -> None:
        if has_inference_model and not inference_thread_only:
            torch_inference_model = torch_training_model.inference_model
            # make differences between params of torch_training_model and torch_inference_model.
            torch_training_model.model.weight.data += 1.0
            torch_training_model.forward(
                torch.zeros([2, 3])
            ).mean().backward()  # Assign grad
            # check if differences are made correctly.
            assert not torch.equal(
                torch_training_model.model.weight,
                torch_inference_model._raw_model.weight,
            )
            assert isinstance(torch_training_model.model.weight.grad, torch.Tensor)
            assert torch_inference_model._raw_model.weight.grad is None

            weight_data = torch_training_model.model.weight.data.clone()
            weight_grad = torch_training_model.model.weight.grad.clone()
            torch_training_model.sync()

            assert torch.equal(torch_training_model.model.weight.data, weight_data)
            assert torch.equal(torch_training_model.model.weight.grad, weight_grad)
            assert torch.equal(
                torch_training_model.model.weight,
                torch_inference_model._raw_model.weight,
            )
            assert (
                torch_training_model.model.weight
                is not torch_inference_model._raw_model.weight
            )
            assert torch_inference_model._raw_model.weight.grad is None
