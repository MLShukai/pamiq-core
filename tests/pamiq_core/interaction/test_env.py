from pamiq_core.interaction.env import Environment
from pamiq_core.interaction.event_mixin import InteractionEventMixin
from pamiq_core.state_persistence import PersistentStateMixin


class TestEnvironment:
    """Tests for Environment class."""

    def test_environment_inherits_from_required_mixins(self):
        """Test Environment subclass."""
        assert issubclass(Environment, InteractionEventMixin)
        assert issubclass(Environment, PersistentStateMixin)

    def test_abstract_methods(self):
        """Test abstract methods of Environment."""
        assert Environment.__abstractmethods__ == frozenset({"observe", "affect"})
