import copy
import logging
from pathlib import Path
from threading import RLock
from typing import override

import pytest
import torch
import torch.nn as nn
from pytest_mock import MockerFixture

from pamiq_core.torch.model import (
    TorchInferenceModel,
    TorchTrainingModel,
    UnwrappedContextManager,
    default_infer_procedure,
    get_device,
)

logger = logging.getLogger(__name__)

CPU_DEVICE = torch.device("cpu")
CUDA_DEVICE = torch.device("cuda:0")


def get_available_devices() -> list[torch.device]:
    devices = [CPU_DEVICE]
    if torch.cuda.is_available():
        devices.append(CUDA_DEVICE)
    return devices


logger.info("Available devices: " + ", ".join(map(str, get_available_devices())))

parametrize_device = pytest.mark.parametrize("device", get_available_devices())


@parametrize_device
class TestGetDevice:
    def test_get_device_with_parameters(self, device: torch.device) -> None:
        model = nn.Linear(2, 3).to(device)
        assert get_device(model) == device

    def test_get_device_with_buffers_only(self, device: torch.device) -> None:
        class BufferOnlyModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("sample_buffer", torch.tensor([1.0]))

        model = BufferOnlyModule().to(device)
        assert get_device(model) == device

    def test_get_device_with_empty_module(self, device: torch.device) -> None:
        model = nn.Module().to(device)
        assert get_device(model, CPU_DEVICE) == CPU_DEVICE
        assert get_device(model, CUDA_DEVICE) == CUDA_DEVICE
        assert get_device(model) == torch.get_default_device()


@parametrize_device
def test_default_infer_procedure(device: torch.device) -> None:
    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.p = torch.nn.Parameter(torch.randn([2, 3]))

        @override
        def forward(
            self, a: torch.Tensor, b: torch.Tensor, use_b: bool = True
        ) -> torch.Tensor:
            x = a @ self.p
            if use_b:
                x = x + b
            return x

    model = Model().to(device)
    a = torch.randn([5, 2])
    b = torch.randn([5, 3])

    # check whether tensors are correctly passed to the model
    # when they are included in *args and **kwds.
    output_tensor = default_infer_procedure(model, a, b=b)
    expected_tensor = model(a.to(device), b.to(device))
    assert torch.equal(output_tensor, expected_tensor)
    assert output_tensor.device == device
    # Check whether an arg that is not a tensor can also be correctly passed to.
    output_tensor = default_infer_procedure(model, a, b, use_b=False)
    expected_tensor = model(a.to(device), b, use_b=False)
    assert torch.equal(output_tensor, expected_tensor)
    assert output_tensor.device == device


class TestUnwrappedContextManager:
    """Test UnwrappedContextManager class functionality."""

    @pytest.fixture
    def model(self) -> nn.Module:
        return nn.Linear(3, 5)

    @pytest.fixture
    def lock(self) -> RLock:
        return RLock()

    def test_context_manager_with_inference_mode(
        self, model: nn.Module, lock: RLock
    ) -> None:
        """Test context manager with inference mode enabled."""
        ctx_manager = UnwrappedContextManager(model, lock, inference_mode=True)

        with ctx_manager as accessed_model:
            assert accessed_model is model
            # Test that model works in inference mode
            input_tensor = torch.randn(2, 3)
            output = accessed_model(input_tensor)
            assert output.shape == (2, 5)
            assert not output.requires_grad  # Inference mode disables gradients

    def test_context_manager_without_inference_mode(
        self, model: nn.Module, lock: RLock
    ) -> None:
        """Test context manager with inference mode disabled."""
        ctx_manager = UnwrappedContextManager(model, lock, inference_mode=False)

        with ctx_manager as accessed_model:
            assert accessed_model is model
            # Test that gradients work when inference_mode=False
            input_tensor = torch.randn(2, 3, requires_grad=True)
            output = accessed_model(input_tensor)
            assert output.shape == (2, 5)
            assert output.requires_grad
            # Verify backward pass works
            output.mean().backward()
            assert input_tensor.grad is not None

    def test_lock_acquisition_and_release(
        self, model: nn.Module, mocker: MockerFixture
    ) -> None:
        """Test that lock is properly acquired and released."""
        mock_lock = mocker.Mock()
        ctx_manager = UnwrappedContextManager(model, mock_lock, inference_mode=True)

        # Enter context
        returned_model = ctx_manager.__enter__()
        mock_lock.acquire.assert_called_once()
        assert returned_model is model

        # Exit context
        ctx_manager.__exit__(None, None, None)
        mock_lock.release.assert_called_once()

    def test_nested_context_with_rlock(self, model: nn.Module, lock: RLock) -> None:
        """Test nested contexts work properly with RLock."""
        ctx1 = UnwrappedContextManager(model, lock, inference_mode=True)
        ctx2 = UnwrappedContextManager(model, lock, inference_mode=True)

        with ctx1 as model1:
            with ctx2 as model2:
                assert model1 is model
                assert model2 is model


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
            output_tensor.mean().backward()

    def test_unwrap_returns_correct_context_manager(
        self,
        torch_inference_model: TorchInferenceModel,
        model: nn.Module,
    ) -> None:
        """Test that unwrap returns an UnwrappedContextManager with correct
        parameters."""
        # Test with default inference_mode=True
        ctx_manager = torch_inference_model.unwrap()
        assert isinstance(ctx_manager, UnwrappedContextManager)
        assert torch.is_inference_mode_enabled() is False
        with ctx_manager as m:
            assert m is model
            assert torch.is_inference_mode_enabled() is True
        assert torch.is_inference_mode_enabled() is False

        # Test with inference_mode=False
        ctx_manager_no_inference = torch_inference_model.unwrap(inference_mode=False)
        assert isinstance(ctx_manager_no_inference, UnwrappedContextManager)
        with ctx_manager_no_inference as m:
            assert m is model
            assert torch.is_inference_mode_enabled() is False


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

    def test_create_inference_with_torch_training_model_default(
        self, torch_training_model_default: TorchTrainingModel
    ) -> None:
        torch_inference_model = torch_training_model_default.inference_model
        # check if the internal models have same params.
        assert torch.equal(
            torch_training_model_default.model.weight,
            torch_inference_model._raw_model.weight,
        )
        # two models must have different pointers.
        assert (
            torch_training_model_default.model is not torch_inference_model._raw_model
        )

    def test_create_inference_with_torch_training_model_inference_only(
        self, torch_training_model_inference_only: TorchTrainingModel
    ) -> None:
        torch_inference_model = torch_training_model_inference_only.inference_model
        # check if the internal models have same params.
        assert torch.equal(
            torch_training_model_inference_only.model.weight,
            torch_inference_model._raw_model.weight,
        )
        # two models must have same pointers.
        assert (
            torch_training_model_inference_only.model
            is torch_inference_model._raw_model
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

    def test_save_and_load_state(
        self,
        torch_training_model_default: TorchTrainingModel,
        tmp_path: Path,
    ):
        test_path = tmp_path / "model_params"
        torch_training_model_default.save_state(test_path)
        assert (tmp_path / "model_params.pt").is_file()
        saved_params = copy.deepcopy(torch_training_model_default.model.state_dict())
        # make differences
        for model_weight in torch_training_model_default.model.parameters():
            model_weight.data += 1.0
        # check if the differences between the models are made correctly.
        model_params = torch_training_model_default.model.state_dict()
        assert model_params is not saved_params
        assert list(model_params.keys()) == list(saved_params.keys())
        for key in saved_params.keys():
            assert not torch.equal(model_params[key], saved_params[key])
        # check if load can be performed correctly.
        torch_training_model_default.load_state(test_path)
        loaded_params = torch_training_model_default.model.state_dict()
        assert loaded_params is not saved_params
        assert list(loaded_params.keys()) == list(saved_params.keys())
        for key in saved_params.keys():
            assert torch.equal(loaded_params[key], saved_params[key])

    @parametrize_device
    def test_initialize_with_parameter_file(self, tmp_path: Path, device):
        """Test initialization with a parameter file."""
        # Create a model with custom parameters
        custom_model = nn.Linear(3, 5)

        # Save the parameters to a file
        param_file = tmp_path / "model_params.pt"
        torch.save(custom_model.state_dict(), param_file)

        # Initialize a training model with the parameter file
        training_model = TorchTrainingModel(
            nn.Linear(3, 5),
            has_inference_model=True,
            inference_thread_only=False,
            pretrained_parameter_file=param_file,
            device=device,
        )

        custom_model.to(device)

        # Verify the parameters were loaded correctly
        assert torch.equal(training_model.model.weight.data, custom_model.weight.data)
        assert torch.equal(training_model.model.bias.data, custom_model.bias.data)

        # Ensure the inference model also has the correct parameters
        inference_model = training_model.inference_model
        assert torch.equal(
            inference_model._raw_model.weight.data, custom_model.weight.data
        )
        assert torch.equal(inference_model._raw_model.bias.data, custom_model.bias.data)

    @parametrize_device
    def test_initialize_with_compile(self, device, mocker: MockerFixture):
        model = nn.Linear(10, 20)
        spy_compile = mocker.spy(model, "compile")
        training_model = TorchTrainingModel(model, compile=True, device=device)
        inference_model = training_model.inference_model

        spy_compile.assert_called_with()
        assert spy_compile.call_count == 2  # Include inference model

        assert training_model(torch.randn(8, 10, device=device)).shape == (8, 20)
        assert inference_model(torch.randn(8, 10, device=device)).shape == (8, 20)
        input = torch.randn(8, 10, device=device)
        assert torch.allclose(training_model(input), inference_model(input))

        training_model.sync()

    def test_string_inference_procedure(self):
        """Test using a string to specify the inference procedure."""

        class CustomModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(3, 5)

            @override
            def forward(self, x):
                return self.linear(x)

            def custom_inference(self, x, temperature=1.0):
                """Custom inference method with temperature scaling."""
                output = self.forward(x)
                return output / temperature

        # Test with string inference procedure
        model = CustomModel()
        training_model = TorchTrainingModel(
            model, inference_procedure="custom_inference"
        )

        input_tensor = torch.randn(2, 3)

        # Test that the custom inference procedure is used
        inference_model = training_model.inference_model
        output_default = inference_model.infer(input_tensor)
        output_with_temp = inference_model.infer(input_tensor, temperature=2.0)

        # Verify temperature scaling works
        assert torch.allclose(output_default / 2.0, output_with_temp, atol=1e-6)

    def test_invalid_string_inference_procedure(self):
        """Test error handling for invalid string inference procedure."""

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(3, 5)

            @override
            def forward(self, x):
                return self.linear(x)

            # Add a non-callable class attribute
            not_callable = "This is not a method"

        model = SimpleModel()

        # Test with non-existent method name
        with pytest.raises(
            AttributeError,
            match="The model class SimpleModel does not have a method named 'non_existent_method'",
        ):
            TorchTrainingModel(model, inference_procedure="non_existent_method")

        # Test with attribute that is not callable
        with pytest.raises(
            ValueError,
            match="The specified inference_procedure 'not_callable' is not a callable method",
        ):
            TorchTrainingModel(model, inference_procedure="not_callable")
