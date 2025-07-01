# Launch

The `launch` function is the entry point for starting a PAMIQ-Core system. It initializes all components, connects them together, and manages the system's lifecycle.

!!! warning "Breaking Changes in v0.5"

    The `launch` function parameter `data` has been renamed to `buffers` for better clarity. If you're upgrading from v0.4, update your code:

    ```python
    # Before (v0.4)
    launch(
        interaction=interaction,
        models=models,
        data={"buffer": my_buffer},  # Old parameter name
        trainers=trainers,
        config=config
    )

    # After (v0.5)
    launch(
        interaction=interaction,
        models=models,
        buffers={"buffer": my_buffer},  # New parameter name
        trainers=trainers,
        config=config
    )
    ```

## Basic Usage

To launch a PAMIQ-Core system, you need to provide your interaction components, models, data buffers, trainers, and configuration settings:

```python
from pamiq_core import launch, LaunchConfig, Interaction

# Create your agent and environment
agent = YourAgent()
environment = YourEnvironment()

# Create an interaction between them
interaction = Interaction(agent, environment)

# Launch the system
launch(
    interaction=interaction,
    models={"model_name": your_model},
    buffers={"buffer_name": your_data_buffer},
    trainers={"trainer_name": your_trainer},
    config=LaunchConfig(
        states_dir="./saved_states",
        max_uptime=3600,  # Run for 1 hour
        time_scale=2.0    # Run at 2x speed
    )
)
```

## Common Configuration Scenarios

### Accelerated Learning

To speed up time for faster training:

```python
config = LaunchConfig(
    time_scale=10.0  # Run at 10x speed
)
```

### Disabling Web API

To run the system without remote control capabilities:

```python
config = LaunchConfig(
    web_api_address=None,  # Disable web API server
)
```

### Resumable Training

To save system state for later resumption:

```python
from pamiq_core.state_persistence import PeriodicSaveCondition, LatestStatesKeeper

# Initial run with automatic state management
launch(
    interaction=interaction,
    models=models,
    buffers=buffers,
    trainers=trainers,
    config=LaunchConfig(
        states_dir="./saved_states",
        save_state_condition=PeriodicSaveCondition(600.0),  # Save every 10 minutes
        states_keeper=LatestStatesKeeper(
            states_dir="./saved_states",
            max_keep=5  # Keep only the 5 most recent states
        )
    )
)

# Later, resume from the last saved state
latest_state = list(Path("./saved_states").glob("*.state"))[-1]
launch(
    interaction=interaction,
    models=models,
    buffers=buffers,
    trainers=trainers,
    config=LaunchConfig(
        states_dir="./saved_states",
        saved_state_path=latest_state
    )
)
```

## API Reference

More details, Checkout to the [API Reference](../api/launch.md)
