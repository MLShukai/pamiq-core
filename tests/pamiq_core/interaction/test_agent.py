from pathlib import Path
from typing import override

import pytest
from pytest_mock import MockerFixture

from pamiq_core.data import DataCollector, DataCollectorsDict
from pamiq_core.interaction.agent import Agent
from pamiq_core.interaction.event_mixin import InteractionEventMixin
from pamiq_core.model import InferenceModel, InferenceModelsDict
from pamiq_core.state_persistence import PersistentStateMixin
from pamiq_core.thread import ThreadEventMixin


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


class ChildAgentImpl(Agent[None, None]):
    """Concrete implementation of Agent for testing."""

    @override
    def step(self, observation: None) -> None:
        pass


class TestAgent:
    """Tests for Agent class."""

    def test_agent_inherits(self):
        """Test that Agent Super Class."""
        assert issubclass(Agent, InteractionEventMixin)
        assert issubclass(Agent, PersistentStateMixin)
        assert issubclass(Agent, ThreadEventMixin)

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

    @pytest.fixture
    def child_agent(self) -> ChildAgentImpl:
        """Fixture providing a child agent implementation."""
        return ChildAgentImpl()

    @pytest.fixture
    def parent_agent(self, child_agent: AgentImpl) -> AgentImpl:
        """Fixture providing a parent agent with a child agent."""
        return AgentImpl(agents={"child": child_agent})

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

    def test_attach_inference_models_propagation(
        self,
        parent_agent: AgentImpl,
        child_agent: ChildAgentImpl,
        mocker: MockerFixture,
    ):
        """Test that attaching inference models propagates to child agents."""
        # Create mock inference model
        mock_inference_model = mocker.Mock(InferenceModel)
        inference_models = InferenceModelsDict({"test_model": mock_inference_model})

        # Spy on the child's attach_inference_models method
        spy_child_attach = mocker.spy(child_agent, "attach_inference_models")

        # Attach to parent
        parent_agent.attach_inference_models(inference_models)

        # Verify child's method was called with the same models
        spy_child_attach.assert_called_once_with(inference_models)

    def test_attach_data_collectors_propagation(
        self,
        parent_agent: AgentImpl,
        child_agent: AgentImpl,
        data_collectors_dict: DataCollectorsDict,
        mocker: MockerFixture,
    ):
        """Test that attaching data collectors propagates to child agents."""
        # Spy on the child's attach_data_collectors method
        spy_child_attach = mocker.spy(child_agent, "attach_data_collectors")

        # Attach to parent
        parent_agent.attach_data_collectors(data_collectors_dict)

        # Verify child's method was called with the same collectors
        spy_child_attach.assert_called_once_with(data_collectors_dict)

    def test_setup_propagation(
        self,
        parent_agent: AgentImpl,
        child_agent: ChildAgentImpl,
        mocker: MockerFixture,
    ):
        """Test that setup event propagates to child agents."""
        # Spy on the child's method
        spy_child_setup = mocker.spy(child_agent, "setup")

        # Call setup on parent
        parent_agent.setup()

        # Verify child's setup was called
        spy_child_setup.assert_called_once_with()

    def test_teardown_propagation(
        self,
        parent_agent: AgentImpl,
        child_agent: ChildAgentImpl,
        mocker: MockerFixture,
    ):
        """Test that teardown event propagates to child agents."""
        # Spy on the child's method
        spy_child_teardown = mocker.spy(child_agent, "teardown")

        # Call teardown on parent
        parent_agent.teardown()

        # Verify child's teardown was called
        spy_child_teardown.assert_called_once_with()

    def test_save_and_load_state_with_child_agents(
        self,
        parent_agent: AgentImpl,
        child_agent: ChildAgentImpl,
        mocker: MockerFixture,
        tmp_path: Path,
    ):
        """Test saving and loading state with child agents."""
        # Spy on save and load methods
        spy_child_save = mocker.spy(child_agent, "save_state")
        spy_child_load = mocker.spy(child_agent, "load_state")

        # Save state
        save_path = tmp_path / "agent_state"
        parent_agent.save_state(save_path)
        assert save_path.is_dir()

        # Verify child's save_state was called with correct path
        spy_child_save.assert_called_once_with(save_path / "child")

        # Load state
        parent_agent.load_state(save_path)

        # Verify child's load_state was called with correct path
        spy_child_load.assert_called_once_with(save_path / "child")

    def test_save_and_load_state_no_child_agents(
        self,
        agent_attached: AgentImpl,
        tmp_path: Path,
    ):
        """Test saving and loading state with child agents."""
        # Spy on save and load methods

        # Save state
        save_path = tmp_path / "agent_state"
        agent_attached.save_state(save_path)
        assert not save_path.exists()

    def test_on_paused_propagation(
        self,
        parent_agent: AgentImpl,
        child_agent: ChildAgentImpl,
        mocker: MockerFixture,
    ):
        """Test that on_paused event propagates to child agents."""
        # Spy on the child's on_paused method
        spy_child_on_paused = mocker.spy(child_agent, "on_paused")

        # Call on_paused on parent
        parent_agent.on_paused()

        # Verify child's on_paused was called
        spy_child_on_paused.assert_called_once_with()

    def test_on_resumed_propagation(
        self,
        parent_agent: AgentImpl,
        child_agent: ChildAgentImpl,
        mocker: MockerFixture,
    ):
        """Test that on_resumed event propagates to child agents."""
        # Spy on the child's on_resumed method
        spy_child_on_resumed = mocker.spy(child_agent, "on_resumed")

        # Call on_resumed on parent
        parent_agent.on_resumed()

        # Verify child's on_resumed was called
        spy_child_on_resumed.assert_called_once_with()
