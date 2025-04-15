import logging
import pickle
import shutil
import threading
import time  # can not use pamiq_core.time because circular import problem.
from datetime import datetime
from pathlib import Path
from typing import Any


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


class LatestStatesKeeper:
    """Keeps a fixed number of state directories by removing older ones.

    This class monitors and manages the number of state directories in a
    specified path, ensuring only the most recent ones (up to max_keep)
    are retained, preventing disk space exhaustion.
    """

    def __init__(
        self,
        states_dir: str | Path,
        state_name_pattern: str = "*.state",
        max_keep: int = 10,
    ) -> None:
        """Initialize the state keeper.

        Args:
            states_dir: Directory where states are stored.
            state_name_pattern: Pattern to match state directories.
            max_keep: Maximum number of state directories to keep.
        """
        if max_keep < 0:
            raise ValueError("max_keep must be larger than 0.")

        self.states_dir = Path(states_dir)
        self.state_name_pattern = state_name_pattern
        self.max_keep = max_keep
        self._running = False
        self._thread = None

        from pamiq_core.utils.reflection import (
            get_class_module_path,  # Avoid circular import problem.
        )

        self._logger = logging.getLogger(get_class_module_path(self.__class__))

        if not self.states_dir.exists():
            self._logger.warning(
                f"States directory {self.states_dir} does not exist. Creating it."
            )
            self.states_dir.mkdir(parents=True, exist_ok=True)

    def start(self, background: bool = True) -> None:
        """Start the background cleanup thread if not already running.

        Args:
            background: Whether to run in background thread.
        """
        if self._running:
            return
        self._running = True
        if background:
            self._thread = threading.Thread(target=self._cleanup)
            self._thread.start()
            self._logger.info(
                f"Started background state cleanup thread. Max keep: {self.max_keep}"
            )
        else:
            self._cleanup()

    def stop(self) -> None:
        """Stop the background cleanup thread if running."""
        if self._running:
            self._running = False
            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=1.0)
            self._logger.info("Stopped background state cleanup thread")

    def _cleanup(self) -> None:
        """Background thread function that periodically cleans up states."""

        try:
            while self._running:
                self.cleanup()
                time.sleep(0.1)  # Check every 0.1 seconds

        except Exception as e:
            self._logger.error(f"Error in background cleanup: {e}")
        except KeyboardInterrupt:
            pass

    def cleanup(self) -> list[Path]:
        """Clean up old state directories, keeping only the most recent ones.

        Returns:
            List of removed state directory paths.
        """
        # Get all state directories matching the pattern
        state_dirs = list(self.states_dir.glob(self.state_name_pattern))

        # Sort by modification time (newest first)
        state_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)

        # Keep only max_keep number of directories
        dirs_to_remove = state_dirs[self.max_keep :]

        removed_dirs: list[Path] = []
        for dir_path in dirs_to_remove:
            shutil.rmtree(dir_path)
            removed_dirs.append(dir_path)
            self._logger.info(f"Removed old state directory: {dir_path}")
        return removed_dirs
