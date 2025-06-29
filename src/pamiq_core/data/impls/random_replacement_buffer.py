import math
import pickle
import random
from pathlib import Path
from typing import override

from ..buffer import DataBuffer


class RandomReplacementBuffer[T](DataBuffer[T, list[T]]):
    """Buffer implementation that randomly replaces elements when full.

    This buffer keeps track of collected data and, when full, randomly
    replaces existing elements based on a configurable probability.
    """

    def __init__(
        self,
        max_size: int,
        replace_probability: float | None = None,
        expected_survival_length: int | None = None,
    ) -> None:
        """Initialize a RandomReplacementBuffer.

        Args:
            max_size: Maximum number of data points to store.
            replace_probability: Probability of replacing an existing element when buffer is full.
                Must be between 0.0 and 1.0 inclusive. If None and expected_survival_length is provided,
                this will be computed automatically. Default is 1.0 if both are None.
            expected_survival_length: Expected number of steps that data should survive in the buffer.
                Used to automatically compute replace_probability if replace_probability is None.
                Cannot be specified together with replace_probability.

        Raises:
            ValueError: If replace_probability is not between 0.0 and 1.0 inclusive, or if both
                replace_probability and expected_survival_length are specified.
        """
        super().__init__(max_size)

        if replace_probability is None:
            if expected_survival_length is None:
                replace_probability = 1.0
            else:
                replace_probability = (
                    self.compute_replace_probability_from_expected_survival_length(
                        max_size, expected_survival_length
                    )
                )
        elif expected_survival_length is not None:
            raise ValueError(
                "Cannot specify both replace_probability and expected_survival_length. "
                "Please specify only one of them."
            )
        if not (1.0 >= replace_probability >= 0.0):
            raise ValueError(
                "replace_probability must be between 0.0 and 1.0 inclusive"
            )

        self._data_list: list[T] = []

        self._replace_probability = replace_probability
        self._current_size = 0

    @staticmethod
    def compute_replace_probability_from_expected_survival_length(
        max_size: int, survival_length: int
    ) -> float:
        """Compute the replace probability from expected survival length.

        This method calculates the replacement probability needed to achieve
        a desired expected survival length for data in the buffer.

        The computation is based on the mathematical analysis described in below:
            https://zenn.dev/gesonanko/scraps/b581e75bfd9f3e

        Args:
            max_size: Maximum size of the buffer.
            survival_length: Expected number of steps that data should survive.

        Returns:
            The computed replacement probability between 0.0 and 1.0.
        """
        gamma = 0.5772156649015329  # Euler-Mascheroni constant
        p = max_size / survival_length * (math.log(max_size) + gamma)
        return min(max(p, 0.0), 1.0)  # Clamp value between 0 to 1.

    @property
    def is_full(self) -> bool:
        """Check if the buffer has reached its maximum capacity.

        Returns:
            True if the buffer is full, False otherwise.
        """
        return self._current_size >= self.max_size

    @override
    def add(self, data: T) -> None:
        """Add a new data sample to the buffer.

        If the buffer is full, the new data may replace an existing entry
        based on the configured replacement probability.

        Args:
            data: Data element to add to the buffer.
        """
        if self.is_full:
            if random.random() > self._replace_probability:
                return
            replace_index = random.randint(0, self.max_size - 1)
            self._data_list[replace_index] = data
        else:
            self._data_list.append(data)
            self._current_size += 1

    @override
    def get_data(self) -> list[T]:
        """Retrieve all stored data from the buffer.

        Returns:
            List of all stored data elements.
            Returns a copy of the internal data to prevent modification.
        """
        return self._data_list.copy()

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

        Saves the data list to a pickle file with .pkl extension.

        Args:
            path: File path where to save the buffer state (without extension).
        """
        with open(path.with_suffix(".pkl"), "wb") as f:
            pickle.dump(self._data_list, f)

    @override
    def load_state(self, path: Path) -> None:
        """Load the buffer state from the specified path.

        Loads data list from pickle file with .pkl extension.

        Args:
            path: File path from where to load the buffer state (without extension).
        """
        with open(path.with_suffix(".pkl"), "rb") as f:
            self._data_list = list(pickle.load(f))[: self.max_size]
        self._current_size = len(self._data_list)
