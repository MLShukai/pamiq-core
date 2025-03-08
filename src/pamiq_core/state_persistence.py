from datetime import datetime
from pathlib import Path


class PersistentStateMixin:
    """Mixin class for objects with persistable state.

    This mixin provides the ability to save and load state. Classes that
    inherit from this mixin must implement `save_state()` and
    `load_state()`.
    """

    def save_state(self, path: Path):
        """Save state to `path`"""
        pass

    def load_state(self, path: Path):
        """Load state from `path`"""
        pass


class StateStore:
    """Class to save and load multiple persistable objects at once.

    This class saves the state of each registered object to the
    specified directory. It is also possible to restore the state from
    the directory.
    """

    def __init__(
        self, states_dir: Path, state_name_format: str = "%Y-%m-%d_%H-%M-%S,%f.state"
    ) -> None:
        """
        Args:
            states_dir: Root path to the directory where states are saved
            state_name_format: Format for the subdirectory name (defaults to timestamp)
        """
        self.states_dir = Path(states_dir)
        self.states_dir.mkdir(exist_ok=True)
        self.state_name_format = state_name_format
        self._registered_states: dict[str, PersistentStateMixin] = {}

    def register(self, name: str, state: PersistentStateMixin) -> None:
        """Register a persistable object with a unique name.

        Args:
            name: Unique name to identify the state
            state: Object implementing PersistentStateMixin

        Raises:
            KeyError: If `name` is already registered
        """
        if name in self._registered_states:
            raise KeyError(f"State with name '{name}' is already registered")
        self._registered_states[name] = state

    def save_state(self) -> Path:
        """Save the all states of registered objects.

        Returns:
            Path: Path to the directory where the states are saved

        Raises:
            FileExistsError: If the directory (`states_path`) already exists (This only occurs if multiple attempts to create directories are at the same time)
        """
        state_path = self.states_dir / datetime.now().strftime(self.state_name_format)
        state_path.mkdir()
        for name, state in self._registered_states.items():
            state.save_state(state_path / name)
        return state_path

    def load_state(self, state_path: Path) -> None:
        """Restores the state from the `state_path` directory.

        Args:
            state_path: Path to the directory where the state is saved

        Raises:
            FileNotFoundError: If the specified path does not exist
        """
        if not state_path.exists():
            raise FileNotFoundError(f"State path: '{state_path}' not found!")
        for name, state in self._registered_states.items():
            state.load_state(state_path / name)
