import pickle
import random
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import override

from ..buffer import DataBuffer, StepData


class RandomReplacementBuffer[T](DataBuffer[T]):
    """Buffer implementation that randomly replaces elements when full.

    This buffer keeps track of collected data and, when full, randomly
    replaces existing elements based on a configurable probability.

    See this page to learn more properties:
        https://zenn.dev/gesonanko/scraps/b581e75bfd9f3e
    """

    def __init__(
        self,
        collecting_data_names: Iterable[str],
        max_size: int,
        replace_probability: float = 1.0,
    ) -> None:
        """Initialize a RandomReplacementBuffer.

        Args:
            collecting_data_names: Names of data fields to collect.
            max_size: Maximum number of data points to store.
            replace_probability: Probability of replacing an existing element when buffer is full.
                Must be between 0.0 and 1.0 inclusive. Default is 1.0 (always replace).

        Raises:
            ValueError: If replace_probability is not between 0.0 and 1.0 inclusive.
        """
        super().__init__(collecting_data_names, max_size)

        if not (1.0 >= replace_probability >= 0.0):
            raise ValueError(
                "replace_probability must be between 0.0 and 1.0 inclusive"
            )

        self._lists_dict: dict[str, list[T]] = {
            name: [] for name in collecting_data_names
        }

        self._replace_probability = replace_probability
        self._current_size = 0

    @property
    def is_full(self) -> bool:
        """Check if the buffer has reached its maximum capacity.

        Returns:
            True if the buffer is full, False otherwise.
        """
        return self._current_size >= self.max_size

    @override
    def add(self, step_data: StepData[T]) -> None:
        """Add a new data sample to the buffer.

        If the buffer is full, the new data may replace an existing entry
        based on the configured replacement probability.

        Args:
            step_data: Dictionary containing data for one step. Must contain
                all fields specified in collecting_data_names.

        Raises:
            KeyError: If a required data field is missing from step_data.
        """
        for name in self.collecting_data_names:
            if name not in step_data:
                raise KeyError(f"Required data '{name}' not found in step_data")

        if self.is_full:
            if random.random() > self._replace_probability:
                return
            replace_index = random.randint(0, self.max_size - 1)
            for name in self.collecting_data_names:
                self._lists_dict[name][replace_index] = step_data[name]
        else:
            for name in self.collecting_data_names:
                self._lists_dict[name].append(step_data[name])
            self._current_size += 1

    @override
    def get_data(self) -> Mapping[str, list[T]]:
        """Retrieve all stored data from the buffer.

        Returns:
            Dictionary mapping data field names to lists of their values.
            Returns a copy of the internal data to prevent modification.
        """
        return {name: data.copy() for name, data in self._lists_dict.items()}

    @override
    def __len__(self) -> int:
        """Returns the current number of samples in the buffer.

        Returns:
            int: The number of samples currently stored in the buffer.
        """
        return self._current_size

    @override
    def save_state(self, path: Path) -> None:
        """Save the buffer state to the specified path.

        Creates a directory at the given path and saves each data list as a
        separate pickle file.

        Args:
            path: Directory path where to save the buffer state.
        """
        path.mkdir()
        for name, data in self._lists_dict.items():
            with open(path / f"{name}.pkl", "wb") as f:
                pickle.dump(data, f)

    @override
    def load_state(self, path: Path) -> None:
        """Load the buffer state from the specified path.

        Loads data lists from pickle files in the given directory.

        Args:
            path: Directory path from where to load the buffer state.

        Raises:
            ValueError: If loaded data lists have inconsistent lengths.
        """
        lists_dict: dict[str, list[T]] = {}
        size: int | None = None
        for name in self.collecting_data_names:
            with open(path / f"{name}.pkl", "rb") as f:
                obj = list(pickle.load(f))[: self.max_size]
            if size is None:
                size = len(obj)
            if size != len(obj):
                raise ValueError("Inconsistent list lengths in loaded data")
            lists_dict[name] = obj

        self._lists_dict = lists_dict
        if size is None:
            self._current_size = 0
        else:
            self._current_size = size
