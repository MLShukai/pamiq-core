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
        input_tensor = torch.randn([2, 3], requires_grad=True)
        output_tensor = torch_inference_model.infer(input_tensor)
        expected_tensor = model(input_tensor)
        # check if output tensors match.
        assert torch.equal(output_tensor, expected_tensor)
        # check if grad tracking is disabled.
        assert not output_tensor.requires_grad
        assert output_tensor.grad_fn is None
        # check if backward results in an error.
        with pytest.raises(RuntimeError):
            output_tensor.backward()


class TestTorchTrainingModel:
    @pytest.fixture
    def model(self) -> nn.Module:
        return nn.Linear(3, 5)

    @pytest.fixture
    def torch_training_model_default(self, model: nn.Module) -> TorchTrainingModel:
        return TorchTrainingModel(
            model, has_inference_model=True, inference_thread_only=False
        )

    @pytest.fixture
    def torch_training_model_with_no_inference_model(
        self, model: nn.Module
    ) -> TorchTrainingModel:
        return TorchTrainingModel(
            model, has_inference_model=False, inference_thread_only=False
        )

    @pytest.fixture
    def torch_training_model_inference_only(
        self, model: nn.Module
    ) -> TorchTrainingModel:
        return TorchTrainingModel(
            model, has_inference_model=True, inference_thread_only=True
        )

    @pytest.fixture
    def torch_training_models(
        self,
        torch_training_model_default: TorchTrainingModel,
        torch_training_model_with_no_inference_model: TorchTrainingModel,
        torch_training_model_inference_only: TorchTrainingModel,
    ) -> list[TorchTrainingModel]:
        return [
            torch_training_model_default,
            torch_training_model_with_no_inference_model,
            torch_training_model_inference_only,
        ]

    def test_create_inference(
        self, torch_training_models: list[TorchTrainingModel]
    ) -> None:
        for torch_training_model in torch_training_models:
            if torch_training_model.has_inference_model:
                torch_inference_model = torch_training_model.inference_model
                # check if the internal models are same
                assert torch.equal(
                    torch_training_model.model.weight,
                    torch_inference_model._raw_model.weight,
                )
                # check about pointers
                if torch_training_model.inference_thread_only:
                    assert (
                        torch_training_model.model is torch_inference_model._raw_model
                    )
                else:
                    assert (
                        torch_training_model.model
                        is not torch_inference_model._raw_model
                    )

    def test_forward(
        self, model: nn.Module, torch_training_models: list[TorchTrainingModel]
    ) -> None:
        for torch_training_model in torch_training_models:
            input_tensor = torch.randn([2, 3])
            output_tensor = torch_training_model.forward(input_tensor)
            expected_tensor = model(input_tensor)
            assert torch.equal(output_tensor, expected_tensor)

    def test_sync_impl(self, torch_training_model_default: TorchTrainingModel) -> None:
        torch_training_model = torch_training_model_default
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
