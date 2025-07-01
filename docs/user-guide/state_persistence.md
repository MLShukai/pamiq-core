# State Persistence

State persistence is a core feature of PAMIQ-Core that allows saving and loading the system state at any point during its operation. This capability is essential for continual learning systems, allowing agents to resume operation from where they left off, recover from crashes, or transfer knowledge between sessions.

## Basic Concepts

All major components in PAMIQ-Core implement state persistence through the `PersistentStateMixin` interface, which provides two key methods:

- `save_state(self, path: Path)`: Saves the component's state to the specified path
- `load_state(self, path: Path)`: Loads the component's state from the specified path

These methods are implemented across all user-facing components, including:

- `Agent`, `Environment`, and related interaction module classes
- `DataBuffer`
- `TrainingModel`
- `Trainer`

When you launch a PAMIQ-Core system, the state persistence mechanism is automatically configured, and states are saved at intervals specified in your `LaunchConfig`.

## Custom State Persistence

When implementing custom components, you can override the `save_state` and `load_state` methods to add your own state persistence logic:

```python
from pathlib import Path
from pamiq_core import Agent
from typing import override

class MyCustomAgent(Agent[float, int]):
    """Custom agent with state persistence."""

    def __init__(self):
        super().__init__()
        self.episode_count = 0
        self.total_reward = 0.0
        self.learning_rate = 0.01

    @override
    def step(self, observation: float) -> int:
        """Process observation and return action."""
        # Example decision logic
        action = 1 if observation > 0 else 0
        return action

    @override
    def save_state(self, path: Path) -> None:
        """Save custom agent state.

        Args:
            path: Directory path where to save the state
        """
        # Always call parent method first to handle built-in state persistence
        super().save_state(path)

        # Create directory if it doesn't exist
        path.mkdir(exist_ok=True)

        # Save custom state variables
        with open(path / "episode_stats.txt", "w") as f:
            f.write(f"episode_count: {self.episode_count}\n")
            f.write(f"total_reward: {self.total_reward}\n")
            f.write(f"learning_rate: {self.learning_rate}\n")

    @override
    def load_state(self, path: Path) -> None:
        """Load custom agent state.

        Args:
            path: Directory path from where to load the state
        """
        # Always call parent method first to handle built-in state persistence
        super().load_state(path)

        # Load custom state variables
        try:
            with open(path / "episode_stats.txt", "r") as f:
                lines = f.readlines()
                for line in lines:
                    key, value = line.strip().split(": ")
                    if key == "episode_count":
                        self.episode_count = int(value)
                    elif key == "total_reward":
                        self.total_reward = float(value)
                    elif key == "learning_rate":
                        self.learning_rate = float(value)
        except FileNotFoundError:
            # Handle case when loading from a state that doesn't have custom data
            self.episode_count = 0
            self.total_reward = 0.0
            self.learning_rate = 0.01
```

## State Directory Structure

When PAMIQ-Core saves a state, it creates a directory with the following structure:

```
[timestamp].state/
├── interaction/
│   ├── agent/
│   │   └── ... (agent state files)
│   └── environment/
│       └── ... (environment state files)
├── models/
│   └── ... (model state files)
├── data/
│   └── ... (data buffer state files)
├── trainers/
│   └── ... (trainer state files)
└── time.pkl (time controller state)
```

This organized structure makes it easy to inspect and manage saved states.

## State Management

The state persistence system in PAMIQ-Core automatically manages state directories:

- States are saved based on the `save_state_condition` specified in the `LaunchConfig`
- Old states can be automatically cleaned up using a `StatesKeeper` instance
- States can be loaded during system launch using the `saved_state_path` parameter

```python
from pamiq_core import launch, LaunchConfig
from pamiq_core.state_persistence import PeriodicSaveCondition, LatestStatesKeeper

# Launch with automatic state saving every 5 minutes, keeping the 10 most recent states
launch(
    interaction=interaction,
    models=models,
    buffers=buffers,
    trainers=trainers,
    config=LaunchConfig(
        states_dir="./saved_states",
        save_state_condition=PeriodicSaveCondition(300.0),  # Save every 5 minutes
        states_keeper=LatestStatesKeeper(
            states_dir="./saved_states",
            max_keep=10
        )
    )
)
```

### Save State Conditions

The `save_state_condition` parameter accepts any callable that returns a boolean. When `True`, the system will save its state. PAMIQ-Core provides built-in conditions:

**PeriodicSaveCondition**: Saves state at regular time intervals

```python
from pamiq_core.state_persistence import PeriodicSaveCondition

config = LaunchConfig(
    save_state_condition=PeriodicSaveCondition(300.0)  # Every 5 minutes
)
```

**Custom Conditions**: You can create custom conditions

```python
from pamiq_core import time

# Save state at specific wall clock times (e.g., every hour on the hour)
def save_on_the_hour():
    current_time = time.time()
    minutes_elapsed = (current_time % 3600) / 60  # Minutes past the hour
    return minutes_elapsed < 0.1  # True for the first 6 seconds of each hour

config = LaunchConfig(
    save_state_condition=save_on_the_hour
)
```

### State Cleanup with StatesKeeper

The `StatesKeeper` is an abstract base class that manages the retention and cleanup of saved state directories. PAMIQ-Core provides a built-in implementation:

**LatestStatesKeeper**: Keeps only the N most recent state directories based on modification time

```python
from pamiq_core.state_persistence import LatestStatesKeeper

# Keep only the 5 most recent states
states_keeper = LatestStatesKeeper(
    states_dir="./saved_states",
    max_keep=5,
    state_name_pattern="*.state"  # Optional: pattern to match state directories
)

config = LaunchConfig(
    states_keeper=states_keeper
)
```

You can also implement custom state retention policies by subclassing `StatesKeeper`:

```python
from pamiq_core.state_persistence import StatesKeeper
from pathlib import Path
from collections.abc import Iterable
from typing import override

class SizeBasedStatesKeeper(StatesKeeper):
    """Keep states until total size exceeds a limit."""

    def __init__(self, states_dir: Path, max_total_size_mb: float):
        super().__init__()
        self.states_dir = states_dir
        self.max_total_size_bytes = max_total_size_mb * 1024 * 1024
        self._state_paths: list[Path] = []

    @override
    def append(self, path: Path) -> None:
        self._state_paths.append(path)

    @override
    def select_removal_states(self) -> Iterable[Path]:
        # Calculate total size and remove oldest states if exceeding limit
        total_size = 0
        removal_states = []

        # Sort by modification time (oldest first)
        sorted_paths = sorted(self._state_paths, key=lambda p: p.stat().st_mtime)

        for path in reversed(sorted_paths):
            if path.exists():
                size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
                total_size += size
                if total_size > self.max_total_size_bytes:
                    removal_states.append(path)

        # Remove selected paths from our tracking
        for path in removal_states:
            self._state_paths.remove(path)

        return removal_states
```

## Thread Safety

The state persistence system is designed to be thread-safe:

- The `save_state` operation pauses all threads before saving to ensure consistency
- State saving and loading operations are coordinated by the control thread
- Components have appropriate synchronization mechanisms to handle concurrent access

These safety features ensure that states are saved and loaded correctly even in multi-threaded environments.

## API Reference

More details, Checkout to the [API Reference](../api/state_persistence.md)
