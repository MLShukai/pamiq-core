from __future__ import annotations

from collections import UserDict
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Self, override

from pamiq_core.state_persistence import PersistentStateMixin

from .buffer import DataBuffer
from .interface import DataCollector, DataUser


class DataUsersDict(UserDict[str, DataUser[Any]], PersistentStateMixin):
    """A dictionary mapping names to data users with helper methods for
    collector management.

    Provides utilities for managing multiple data users and their
    associated collectors, with convenient initialization from data
    buffers.
    """

    @override
    def __init__(self, *args: Any, **kwds: Any) -> None:
        """Initialize a DataUsersDict with an empty data collectors dictionary.

        Uses parent class initialization and creates an empty
        DataCollectorsDict to manage associated collectors.
        """
        self._data_collectors_dict = DataCollectorsDict()
        super().__init__(*args, **kwds)

    @property
    def data_collectors_dict(self) -> DataCollectorsDict:
        """Get the dictionary of data collectors associated with this users
        dictionary."""
        return self._data_collectors_dict

    @override
    def __setitem__(self, key: str, item: DataUser[Any]) -> None:
        """Set a data user in the dictionary and create its associated
        collector.

        Updates both the main dictionary with the user and the collectors dictionary
        with a new collector for that user.

        Args:
            key: Name to associate with the data user.
            item: DataUser instance to add to the dictionary.
        """
        super().__setitem__(key, item)
        self._data_collectors_dict[key] = item._collector  # pyright: ignore[reportPrivateUsage]

    @classmethod
    def from_data_buffers(
        cls,
        buffer_map: Mapping[str, DataBuffer[Any, Any]] | None = None,
        /,
        **kwds: DataBuffer[Any, Any],
    ) -> Self:
        """Creates a DataUsersDict from a mapping of data buffers.

        Args:
            buffer_map: Optional mapping of names to data buffers.
            **kwds: Additional name-buffer pairs as keyword arguments.

        Returns:
            New DataUsersDict instance with users created from buffers.
        """
        data: dict[str, DataBuffer[Any, Any]] = {}
        if buffer_map is not None:
            data.update(buffer_map)
        if len(kwds) > 0:
            data.update(kwds)
        return cls({k: DataUser(buf) for k, buf in data.items()})

    @override
    def save_state(self, path: Path) -> None:
        """Save state of all contained DataUser objects to the given path.

        Creates a directory at the specified path and saves each DataUser's state
        in a subdirectory named after its key in this dictionary.

        Args:
            path: Directory path where the states should be saved
        """
        path.mkdir()
        for name, user in self.items():
            user.save_state(path / name)

    @override
    def load_state(self, path: Path) -> None:
        """Load state for all contained DataUser objects from the given path.

        Loads each DataUser's state from a subdirectory named after its key
        in this dictionary.

        Args:
            path: Directory path from where the states should be loaded
        """
        for name, user in self.items():
            user.load_state(path / name)


class DataCollectorsDict(UserDict[str, DataCollector[Any]]):
    """A dictionary for managing exclusive access to data collectors.

    Manages exclusive access to data collectors to ensure each collector
    is only used once per step. Collectors must be explicitly acquired
    before use and can only be acquired once at a time.
    """

    @override
    def __init__(self, *args: Any, **kwds: Any) -> None:
        super().__init__(*args, **kwds)
        self._acquired_collectors: set[str] = set()

    def acquire(self, collector_name: str) -> DataCollector[Any]:
        """Acquires a data collector for exclusive use within a step.

        Args:
            collector_name: Name of the collector to acquire.

        Returns:
            The requested data collector.

        Raises:
            KeyError: If collector is already acquired or not found.
        """
        if collector_name in self._acquired_collectors:
            raise KeyError(f"Data collector '{collector_name}' is already acquired.")
        if collector_name not in self:
            raise KeyError(f"Data collector '{collector_name}' not found.")
        self._acquired_collectors.add(collector_name)
        return self[collector_name]
