from pamiq_core.interaction.env import Environment
from pamiq_core.interaction.event_mixin import InteractionEventMixin
from pamiq_core.state_persistence import PersistentStateMixin
from pamiq_core.thread import ThreadEventMixin


class TestEnvironment:
    """Tests for Environment class."""

    def test_environment_inherits_from_required_mixins(self):
        """Test Environment subclass."""
        assert issubclass(Environment, InteractionEventMixin)
        assert issubclass(Environment, PersistentStateMixin)
        assert issubclass(Environment, ThreadEventMixin)

    def test_abstract_methods(self):
        """Test abstract methods of Environment."""
        assert Environment.__abstractmethods__ == frozenset({"observe", "affect"})
