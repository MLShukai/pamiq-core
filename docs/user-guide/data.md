# Data

!!! warning "Breaking Changes in v0.5"

    The data module has undergone significant changes from v0.4:

    - `StepData` and `BufferData` type removed.
    - `collecting_data_names` parameter removed from constructors
    - `max_size` renamed to `max_queue_size` (can be None)
    - For dictionary data, use `DictSequentialBuffer` or `DictRandomReplacementBuffer`

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
        reward = 1.0 if action == 1 else -1.0

        # Collect experience data as a single value (e.g., tuple)
        experience = (observation, action, reward)
        self.experience_collector.collect(experience)

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
        experiences = self.experience_data.get_data()

        # Unpack the data for training
        if experiences:
            observations, actions, rewards = zip(*experiences)
            print(f"Training on {len(experiences)} samples")
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

class SimpleBuffer[T](DataBuffer[T, list[T]]):
    """A simple buffer that stores data in a list."""

    @override
    def __init__(self, max_size: int) -> None:
        """Initialize the buffer.

        Args:
            max_size: Maximum number of samples to store
        """
        super().__init__(max_queue_size=max_size)
        self._data: list[T] = []
        self._max_size = max_size

    @override
    def add(self, data: T) -> None:
        """Add a new data sample to the buffer.

        Args:
            data: Data element to add
        """
        if len(self._data) < self._max_size:
            self._data.append(data)
        else:
            # Replace oldest data (index 0)
            self._data.pop(0)
            self._data.append(data)

    @override
    def get_data(self) -> list[T]:
        """Retrieve all stored data from the buffer.

        Returns:
            List of all stored data elements
        """
        return self._data.copy()

    @override
    def __len__(self) -> int:
        """Return the current number of samples in the buffer.

        Returns:
            Number of samples currently stored
        """
        return len(self._data)

    # Note: Also implement save_state() and load_state() methods
    # for state persistence (see DataBuffer base class)
```

!!! note "DataBuffer Type Parameters"

    All DataBuffer implementations have two type parameters:

    - `T`: The type of individual data elements
    - `R`: The return type of the `get_data()` method

    For example, `SequentialBuffer[T]` is `DataBuffer[T, list[T]]`, meaning it returns a list of values.

## Built-in DataBuffers

PAMIQ-Core provides several pre-implemented data buffers to handle common use cases:

### SequentialBuffer

The `SequentialBuffer` stores data in sequence and discards the oldest data when the buffer reaches its maximum size:

```python
from pamiq_core.data.impls import SequentialBuffer

# Create a buffer for experiences with max size 1000
buffer = SequentialBuffer[tuple[list[float], int, float]](max_size=1000)

# Add data
experience = ([0.1, 0.2], 1, 0.5)  # (state, action, reward)
buffer.add(experience)

# Get all data
experiences = buffer.get_data()
```

For dictionary data, use `DictSequentialBuffer`:

```python
from pamiq_core.data.impls import DictSequentialBuffer

# Create a buffer for dictionary data
buffer = DictSequentialBuffer[float](["state", "action", "reward"], max_size=1000)

# Add data
buffer.add({"state": 0.1, "action": 1.0, "reward": 0.5})

# Get all data as a dictionary
data = buffer.get_data()  # {"state": [0.1, ...], "action": [1.0, ...], ...}
```

These buffers are useful for:

- Experience replay in reinforcement learning
- Training on the most recent experiences
- Sequential data processing

### RandomReplacementBuffer

The `RandomReplacementBuffer` fills up to its maximum size and then randomly replaces existing samples with a configurable probability:

```python
from pamiq_core.data.impls import RandomReplacementBuffer

# Create a buffer with 80% replacement probability
buffer = RandomReplacementBuffer[int](
    max_size=1000,
    replace_probability=0.8
)
```

For dictionary data, use `DictRandomReplacementBuffer`:

```python
from pamiq_core.data.impls import DictRandomReplacementBuffer

# Create a buffer for dictionary data with replacement
buffer = DictRandomReplacementBuffer[float](
    ["state", "action", "reward"],
    max_size=1000,
    replace_probability=0.8
)
```

These buffers are useful for:

- Maintaining diversity in training data
- Preserving rare or important samples
- Balancing between old and new experiences

The detailed characteristics of this buffer type are discussed in [this article](https://zenn.dev/gesonanko/scraps/b581e75bfd9f3e).

## Thread Safety Considerations

The data system in PAMIQ-Core is designed to be thread-safe, with several important mechanisms:

1. **Collector Acquisition**: Data collectors must be acquired before use, ensuring they can only be accessed by one component at a time
2. **Queue-based Transfer**: Data is transferred between threads using thread-safe queues
3. **Lock Protection**: Critical sections are protected by locks to prevent race conditions

## Migration from v0.4 to v0.5

### Single Value Buffers

If you only need to store one type of data:

```python
# Old (v0.4)
buffer = SequentialBuffer(["observation"], max_size=1000)
collector.collect({"observation": obs})

# New (v0.5)
buffer = SequentialBuffer[float](max_size=1000)
collector.collect(obs)
```

### Dictionary Data Buffers

If you need to store multiple named values:

```python
# Old (v0.4)
buffer = RandomReplacementBuffer(
    ["state", "action", "reward"],
    max_size=1000
)
collector.collect({"state": s, "action": a, "reward": r})

# New (v0.5)
buffer = DictRandomReplacementBuffer[float](
    ["state", "action", "reward"],
    max_size=1000
)
collector.collect({"state": s, "action": a, "reward": r})
```

### State Persistence

State files now use `.pkl` extension automatically - no code changes needed.

______________________________________________________________________

## API Reference

More details, Checkout to the [API Reference](../api/data.md)
