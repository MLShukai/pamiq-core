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

- States are saved at regular intervals as specified in the `LaunchConfig`
- Old states can be automatically cleaned up based on the `max_keep_states` parameter
- States can be loaded during system launch using the `saved_state_path` parameter

```python
from pamiq_core import launch, LaunchConfig

# Launch with automatic state saving every 5 minutes, keeping the 10 most recent states
launch(
    interaction=interaction,
    models=models,
    data=data,
    trainers=trainers,
    config=LaunchConfig(
        states_dir="./saved_states",
        save_state_interval=300.0,  # 5 minutes
        max_keep_states=10
    )
)
```

## Thread Safety

The state persistence system is designed to be thread-safe:

- The `save_state` operation pauses all threads before saving to ensure consistency
- State saving and loading operations are coordinated by the control thread
- Components have appropriate synchronization mechanisms to handle concurrent access

These safety features ensure that states are saved and loaded correctly even in multi-threaded environments.

## API Reference

More details, Checkout to the [API Reference](../api/state_persistence.md)
