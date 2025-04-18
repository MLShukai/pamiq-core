# Launch

The `launch` function is the entry point for starting a PAMIQ-Core system. It initializes all components, connects them together, and manages the system's lifecycle.

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
    data={"buffer_name": your_data_buffer},
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
    time_scale=10.0,  # Run at 10x speed
    save_state_interval=300.0  # Save every 5 minutes
)
```

### Resumable Training

To save system state for later resumption:

```python
# Initial run
launch(
    interaction=interaction,
    models=models,
    data=data,
    trainers=trainers,
    config=LaunchConfig(
        states_dir="./saved_states",
        save_state_interval=600.0  # Save every 10 minutes
    )
)

# Later, resume from the last saved state
latest_state = list(Path("./saved_states").glob("*.state"))[-1]
launch(
    interaction=interaction,
    models=models,
    data=data,
    trainers=trainers,
    config=LaunchConfig(
        states_dir="./saved_states",
        saved_state_path=latest_state
    )
)
```

## API Reference

More details, Checkout to the [API Reference](../api/launch.md)
