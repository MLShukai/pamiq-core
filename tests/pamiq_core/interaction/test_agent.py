from typing import override

import pytest
from pytest_mock import MockerFixture

from pamiq_core.data import DataCollector, DataCollectorsDict
from pamiq_core.interaction.agent import Agent
from pamiq_core.interaction.event_mixin import InteractionEventMixin
from pamiq_core.model import InferenceModel, InferenceModelsDict
from pamiq_core.state_persistence import PersistentStateMixin


class AgentImpl(Agent[str, int]):
    """Concrete implementation of Agent for testing."""

    @override
    def on_inference_models_attached(self) -> None:
        super().on_inference_models_attached()
        self.model = self.get_inference_model("test_model")

    @override
    def on_data_collectors_attached(self) -> None:
        super().on_data_collectors_attached()
        self.collector = self.get_data_collector("test_collector")

    @override
    def step(self, observation: str) -> int:
        """Simple implementation that returns a fixed action."""
        return 42


class TestAgent:
    """Tests for Agent class."""

    def test_agent_inherits(self):
        """Test that Agent Super Class."""
        assert issubclass(Agent, InteractionEventMixin)
        assert issubclass(Agent, PersistentStateMixin)

    def test_abstract_method(self):
        """Test agent's abstract method."""
        assert Agent.__abstractmethods__ == frozenset({"step"})

    @pytest.fixture
    def mock_inference_model(self, mocker: MockerFixture) -> InferenceModel:
        """Fixture providing a mock inference model."""
        return mocker.Mock(InferenceModel)

    @pytest.fixture
    def mock_data_collector(self, mocker: MockerFixture) -> DataCollector:
        """Fixture providing a mock data collector."""
        return mocker.Mock(DataCollector)

    @pytest.fixture
    def mock_collector_for_get(self, mocker: MockerFixture) -> DataCollector:
        """Fixture providing a mock data collector."""
        return mocker.Mock(DataCollector)

    @pytest.fixture
    def inference_models_dict(self, mock_inference_model) -> InferenceModelsDict:
        """Fixture providing an InferenceModelsDict with a test model."""
        return InferenceModelsDict({"test_model": mock_inference_model})

    @pytest.fixture()
    def data_collectors_dict(
        self, mock_data_collector, mock_collector_for_get
    ) -> DataCollectorsDict:
        """Fixture providing a DataCollectorsDict with a test collector."""

        return DataCollectorsDict(
            {
                "test_collector": mock_data_collector,
                "mock_collector_for_get": mock_collector_for_get,
            }
        )

    @pytest.fixture
    def agent(self) -> AgentImpl:
        """Fixture providing an AgentImpl instance."""
        return AgentImpl()

    @pytest.fixture
    def agent_attached(
        self,
        agent: AgentImpl,
        inference_models_dict: InferenceModelsDict,
        data_collectors_dict: DataCollectorsDict,
    ) -> AgentImpl:
        """Fixture providing an AgentImpl with attached models and
        collectors."""
        agent.attach_inference_models(inference_models_dict)
        agent.attach_data_collectors(data_collectors_dict)
        return agent

    def test_attach_inference_models(
        self,
        agent: AgentImpl,
        inference_models_dict: InferenceModelsDict,
        mock_inference_model,
    ) -> None:
        """Test that attaching inference models works correctly."""
        agent.attach_inference_models(inference_models_dict)
        assert agent.model == mock_inference_model

    def test_attach_data_collectors(
        self,
        agent: AgentImpl,
        data_collectors_dict: DataCollectorsDict,
        mock_data_collector,
    ) -> None:
        """Test that attaching data collectors works correctly."""
        agent.attach_data_collectors(data_collectors_dict)
        assert agent.collector == mock_data_collector

    def test_get_inference_model(
        self, agent_attached: AgentImpl, mock_inference_model
    ) -> None:
        """Test getting an inference model by name."""
        model = agent_attached.get_inference_model("test_model")
        assert model == mock_inference_model

    def test_get_data_collector(
        self, agent_attached: AgentImpl, mock_collector_for_get
    ) -> None:
        """Test acquiring a data collector by name."""
        assert (
            agent_attached.get_data_collector("mock_collector_for_get")
            == mock_collector_for_get
        )

    def test_step(self, agent_attached: AgentImpl) -> None:
        """Test the step method returns the expected action."""
        action = agent_attached.step("test observation")
        assert action == 42
