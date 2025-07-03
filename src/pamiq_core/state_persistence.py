import logging
import pickle
import shutil
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any, override


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
        self,
        states_dir: str | Path,
        state_name_format: str = "%Y-%m-%d_%H-%M-%S,%f.state",
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

    def load_state(self, state_path: str | Path) -> None:
        """Restores the state from the `state_path` directory.

        Args:
            state_path: Path to the directory where the state is saved

        Raises:
            FileNotFoundError: If the specified path does not exist
        """
        state_path = Path(state_path)
        if not state_path.exists():
            raise FileNotFoundError(f"State path: '{state_path}' not found!")
        for name, state in self._registered_states.items():
            state.load_state(state_path / name)


def save_pickle(obj: Any, path: Path | str) -> None:
    """Saves an object to a file using pickle serialization.

    Args:
        obj: Any Python object to be serialized.
        path: Path or string pointing to the target file location.

    Raises:
        OSError: If there is an error writing to the specified path.
        pickle.PickleError: If the object cannot be pickled.
    """
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: Path | str) -> Any:
    """Loads an object from a pickle file.

    Args:
        path: Path or string pointing to the pickle file.

    Returns:
        The unpickled Python object.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        OSError: If there is an error reading from the specified path.
        pickle.PickleError: If the file contains invalid pickle data.
        ModuleNotFoundError: If a module required for unpickling is not available.
    """
    with open(path, "rb") as f:
        return pickle.load(f)


class PeriodicSaveCondition:
    """Save state condition based on periodic time intervals.

    This condition triggers state saving at regular time intervals.
    """

    def __init__(self, interval: float) -> None:
        """Initializes PeriodicSaveCondition.

        Args:
            interval: Time interval in seconds between state saves.
        """
        # Import here to avoid circular dependency
        from .utils.schedulers import TimeIntervalScheduler

        self._flag = False

        def set_true() -> None:
            self._flag = True

        self._scheduler = TimeIntervalScheduler(interval, set_true)

    def __call__(self) -> bool:
        """Check if interval has elapsed and state should be saved."""
        self._scheduler.update()
        # get return value and reset flag.
        out, self._flag = self._flag, False
        return out


class StatesKeeper(ABC):
    """Abstract base class for managing and cleaning up saved state
    directories.

    This class provides a framework for implementing different state retention
    policies. Subclasses must implement the `select_removal_states` method to
    define which states should be removed during cleanup.
    """

    def __init__(self) -> None:
        """Initialize the StatesKeeper with a logger."""
        super().__init__()
        from pamiq_core.utils.reflection import (
            get_class_module_path,  # Avoid circular import problem.
        )

        self.logger = logging.getLogger(get_class_module_path(self.__class__))

    def append(self, path: Path) -> None:
        """Appends the state path from StateStore output.

        Args:
            path: Output of StateStore.save_state() in ControlThread.
        """
        pass

    @abstractmethod
    def select_removal_states(self) -> Iterable[Path]:
        """Select which state directories should be removed during cleanup.

        This method must be implemented by subclasses to define their
        specific retention policy.

        Returns:
            An iterable of Path objects representing state directories to remove.
        """
        pass

    def cleanup(self) -> list[Path]:
        """Remove state directories selected for removal.

        Calls `select_removal_states` to determine which directories to remove,
        then deletes them from the filesystem.

        Returns:
            List of paths that were removed.
        """
        removed_paths: list[Path] = []
        for path in self.select_removal_states():
            if path.exists():
                shutil.rmtree(path)
                self.logger.info(f"Removed: '{path}'")
                removed_paths.append(path)
        return removed_paths


class LatestStatesKeeper(StatesKeeper):
    """Keeps a fixed number of most recent state directories by removing older
    ones.

    This class implements a retention policy that keeps only the N most
    recent state directories based on their modification time.
    """

    def __init__(
        self,
        states_dir: str | Path,
        max_keep: int,
        state_name_pattern: str = "*.state",
    ) -> None:
        """Initialize the LatestStatesKeeper.

        Args:
            states_dir: Directory where states are stored.
            max_keep: Maximum number of state directories to keep.
            state_name_pattern: Glob pattern to match state directories.
        """
        super().__init__()
        if max_keep < 0:
            raise ValueError("max_keep must be non-negative")

        states_dir = Path(states_dir)
        self.max_keep = max_keep

        if not states_dir.exists():
            self.logger.warning(
                f"States directory {states_dir} does not exist. Creating it."
            )
            states_dir.mkdir(parents=True, exist_ok=True)

        state_paths = list(states_dir.glob(state_name_pattern))
        state_paths.sort(key=lambda p: p.stat().st_mtime)
        self._state_paths = deque(state_paths)

    @override
    def append(self, path: Path) -> None:
        return self._state_paths.append(path)

    @override
    def select_removal_states(self) -> Iterable[Path]:
        """Select state directories to remove based on the retention policy.

        Returns:
            An iterable of Path objects representing state directories to remove.
        """
        if len(self._state_paths) <= self.max_keep:
            return []

        # Return states beyond max_keep limit
        return [
            self._state_paths.popleft()
            for _ in range(len(self._state_paths) - self.max_keep)
        ]
