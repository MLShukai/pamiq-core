from typing import override

import pytest
import torch
import torch.nn as nn
from pytest_mock import MockerFixture

from pamiq_core.interaction.agent import Agent
from pamiq_core.model import InferenceModel, InferenceModelsDict
from pamiq_core.torch import TorchAgent, TorchInferenceModel


class SimpleModel(nn.Module):
    """Simple PyTorch model for testing."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(10, 5)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class TorchAgentImpl(TorchAgent[torch.Tensor, torch.Tensor]):
    """Concrete implementation of TorchAgent for testing."""

    @override
    def step(self, observation: torch.Tensor) -> torch.Tensor:
        """Simple implementation that returns zeros."""
        del observation  # Unused parameter
        return torch.zeros(5)


class TestTorchAgent:
    """Tests for TorchAgent class."""

    def test_torch_agent_inherits(self):
        """Test that TorchAgent inherits from Agent."""
        assert issubclass(TorchAgent, Agent)

    @pytest.fixture
    def simple_model(self) -> SimpleModel:
        """Fixture providing a simple PyTorch model."""
        return SimpleModel()

    @pytest.fixture
    def torch_inference_model(
        self, simple_model: SimpleModel
    ) -> TorchInferenceModel[SimpleModel]:
        """Fixture providing a TorchInferenceModel."""
        return TorchInferenceModel(simple_model, lambda model, x: model(x))

    @pytest.fixture
    def mock_non_torch_model(self, mocker: MockerFixture) -> InferenceModel:
        """Fixture providing a mock non-torch inference model."""
        return mocker.Mock(InferenceModel)

    @pytest.fixture
    def inference_models_dict(
        self,
        torch_inference_model: TorchInferenceModel[SimpleModel],
        mock_non_torch_model: InferenceModel,
    ) -> InferenceModelsDict:
        """Fixture providing an InferenceModelsDict with test models."""
        return InferenceModelsDict(
            {
                "torch_model": torch_inference_model,
                "non_torch_model": mock_non_torch_model,
            }
        )

    @pytest.fixture
    def agent(self) -> TorchAgentImpl:
        """Fixture providing a TorchAgentImpl instance."""
        return TorchAgentImpl()

    @pytest.fixture
    def agent_with_models(
        self, agent: TorchAgentImpl, inference_models_dict: InferenceModelsDict
    ) -> TorchAgentImpl:
        """Fixture providing a TorchAgentImpl with attached models."""
        agent.attach_inference_models(inference_models_dict)
        return agent

    def test_get_torch_inference_model_success(
        self, agent_with_models: TorchAgentImpl
    ) -> None:
        """Test successful retrieval of a TorchInferenceModel."""
        model = agent_with_models.get_torch_inference_model("torch_model", SimpleModel)
        assert isinstance(model, TorchInferenceModel)

    def test_get_torch_inference_model_with_default_module_type(
        self, agent_with_models: TorchAgentImpl
    ) -> None:
        """Test retrieval of a TorchInferenceModel with default nn.Module
        type."""
        model = agent_with_models.get_torch_inference_model("torch_model")
        assert isinstance(model, TorchInferenceModel)

    def test_get_torch_inference_model_not_torch_model(
        self, agent_with_models: TorchAgentImpl
    ) -> None:
        """Test that ValueError is raised when model is not
        TorchInferenceModel."""
        with pytest.raises(
            ValueError,
            match="Model non_torch_model is not an instance of TorchInferenceModel",
        ):
            agent_with_models.get_torch_inference_model("non_torch_model")

    def test_get_torch_inference_model_wrong_module_type(
        self, agent_with_models: TorchAgentImpl
    ) -> None:
        """Test that TypeError is raised when internal model has wrong type."""

        class WrongModel(nn.Module):
            pass

        with pytest.raises(
            TypeError, match="Internal model is not an instance of WrongModel"
        ):
            agent_with_models.get_torch_inference_model("torch_model", WrongModel)
