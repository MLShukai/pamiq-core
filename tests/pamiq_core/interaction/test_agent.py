from pamiq_core.interaction.agent import Agent
from pamiq_core.interaction.event_mixin import InteractionEventMixin
from pamiq_core.state_persistence import PersistentStateMixin


class TestAgent:
    """Tests for Agent class."""

    def test_agent_inherits(self):
        """Test that Agent Super Class."""
        assert issubclass(Agent, InteractionEventMixin)
        assert issubclass(Agent, PersistentStateMixin)

    def test_abstract_method(self):
        """Test agent's abstract method."""
        assert Agent.__abstractmethods__ == frozenset({"step"})
