# Data

The `data` module provides functionality for collecting, storing, and managing data needed for training models. It enables efficient data flow between inference and training threads, ensuring that learning can happen continuously during agent-environment interactions.

## Basic Concepts

PAMIQ-Core's data system is built around three key components:

1. **DataBuffer**: Stores and manages collected data samples
2. **DataCollector**: Provides an interface for collecting data in the inference thread
3. **DataUser**: Provides an interface for accessing collected data in the training thread

These components work together to create a thread-safe data pipeline:

```
DataCollector in Agent (inference thread)
↓
DataBuffer
↑
DataUser in Trainer (training thread)
```

## Data Flow Architecture

### DataCollector

The `DataCollector` provides a thread-safe interface for collecting data in the inference thread:

```python
from pamiq_core import Agent
from typing import override

class DataCollectingAgent(Agent[float, int]):
    """An agent that collects experience data during inference."""

    @override
    def on_data_collectors_attached(self) -> None:
        """Called when data collectors are attached to the agent."""
        self.experience_collector = self.get_data_collector("experience")

    @override
    def step(self, observation: float) -> int:
        """Process observation and decide on action."""
        # Make a decision
        action = int(observation > 0)

        # Collect experience data
        self.experience_collector.collect({
            "observation": observation,
            "action": action,
            "reward": 1.0 if action == 1 else -1.0
        })

        return action
```

The `collect` method adds a single step's data to an internal queue. This data is later transferred to the data buffer when `update` is called.

### DataUser

The `DataUser` provides access to collected data in the training thread:

```python
from pamiq_core import Trainer
from typing import override

class ExperienceTrainer(Trainer):
    """A trainer that uses collected experience data."""

    @override
    def on_data_users_attached(self) -> None:
        """Called when data users are attached to the trainer."""
        self.experience_data = self.get_data_user("experience")

    @override
    def train(self) -> None:
        """Train models using collected data."""
        # Update to transfer collected data from collectors to buffer
        self.experience_data.update()

        # Get the latest data
        data = self.experience_data.get_data()

        # Use the data for training
        observations = data["observation"]
        actions = data["action"]
        rewards = data["reward"]

        print(f"Training on {len(observations)} samples")
        # ... (training logic)
```

The `update` method transfers data from the collector's queue to the buffer, making it available for training.

## Implementing Custom DataBuffers

You can implement custom data buffers to handle specific data storage requirements. A data buffer must implement three key methods:

1. `add`: Add a new data sample
2. `get_data`: Retrieve all stored data
3. `__len__`: Return the current number of samples

Here's an example of a simple custom buffer:

```python
from pamiq_core.data import DataBuffer
from typing import override

class SimpleBuffer[T](DataBuffer[T]):
    """A simple buffer that stores data in lists."""

    @override
    def __init__(self, collecting_data_names: list[str], max_size: int) -> None:
        """Initialize the buffer.

        Args:
            collecting_data_names: Names of data fields to collect
            max_size: Maximum number of samples to store
        """
        super().__init__(collecting_data_names, max_size)
        self._data = {name: [] for name in collecting_data_names}
        self._count = 0

    @override
    def add(self, step_data: dict[str, T]) -> None:
        """Add a new data sample to the buffer.

        Args:
            step_data: Dictionary containing data for one step
        """
        # Verify all required fields are present
        for name in self._collecting_data_names:
            if name not in step_data:
                raise KeyError(f"Required data '{name}' not found in step_data")

        # Add data to buffer
        if self._count < self.max_size:
            for name in self._collecting_data_names:
                self._data[name].append(step_data[name])
            self._count += 1
        else:
            # Replace oldest data (index 0)
            for name in self._collecting_data_names:
                self._data[name].pop(0)
                self._data[name].append(step_data[name])

    @override
    def get_data(self) -> dict[str, list[T]]:
        """Retrieve all stored data from the buffer.

        Returns:
            Dictionary mapping data field names to lists of values
        """
        return {name: data.copy() for name, data in self._data.items()}

    @override
    def __len__(self) -> int:
        """Return the current number of samples in the buffer.

        Returns:
            Number of samples currently stored
        """
        return self._count
```

## Built-in DataBuffers

PAMIQ-Core provides several pre-implemented data buffers to handle common use cases:

### SequentialBuffer

The `SequentialBuffer` stores data in sequence and discards the oldest data when the buffer reaches its maximum size:

```python
from pamiq_core.data.impls import SequentialBuffer

# Create a buffer for state, action, and reward data with max size 1000
buffer = SequentialBuffer(["state", "action", "reward"], max_size=1000)

# Add data
buffer.add({"state": [0.1, 0.2], "action": 1, "reward": 0.5})

# Get all data
data = buffer.get_data()
```

This buffer is useful for:

- Experience replay in reinforcement learning
- Training on the most recent experiences
- Sequential data processing

### RandomReplacementBuffer

The `RandomReplacementBuffer` fills up to its maximum size and then randomly replaces existing samples with a configurable probability:

```python
from pamiq_core.data.impls import RandomReplacementBuffer

# Create a buffer with 80% replacement probability
buffer = RandomReplacementBuffer(
    ["state", "action", "reward"],
    max_size=1000,
    replace_probability=0.8
)
```

This buffer is useful for:

- Maintaining diversity in training data
- Preserving rare or important samples
- Balancing between old and new experiences

The detailed characteristics of this buffer type are discussed in [this article](https://zenn.dev/gesonanko/scraps/b581e75bfd9f3e).

## Thread Safety Considerations

The data system in PAMIQ-Core is designed to be thread-safe, with several important mechanisms:

1. **Collector Acquisition**: Data collectors must be acquired before use, ensuring they can only be accessed by one component at a time
2. **Queue-based Transfer**: Data is transferred between threads using thread-safe queues
3. **Lock Protection**: Critical sections are protected by locks to prevent race conditions

______________________________________________________________________

## API Reference

More details, Checkout to the [API Reference](../api/data.md)
