# Interaction

The `interaction` module defines interfaces for agents and environments, and manages the data flow between them. This module provides the foundation for building intelligent systems that can interact with their surroundings.

## Agent Implementation

Agents process observations from the environment and decide on actions to take. To implement a custom agent, you need to:

1. Define the observation and action types
2. Override the `step` method to process observations and return actions

```python
from pamiq_core import Agent
from typing import override

class MyAgent(Agent[list[float], int]):
    """Custom agent that processes observations as float lists and outputs integer actions."""

    @override
    def step(self, observation: list[float]) -> int:
        """Process observation and decide on an action.

        Args:
            observation: Current observation from environment

        Returns:
            The chosen action
        """
        # Simple decision logic based on observation
        if sum(observation) > 0:
            return 1
        else:
            return 0
```

### Accessing Inference Models

To access inference models for decision making, override the `on_inference_models_attached` callback:

```python
@override
def on_inference_models_attached(self) -> None:
    """Called when inference models are attached to the agent."""
    self.policy_model = self.get_inference_model("policy")
    self.value_model = self.get_inference_model("value")
```

This callback is executed when the system attaches the models you provided to the `launch` function.

### Collecting Experience Data

To collect data for training, override the `on_data_collectors_attached` callback:

```python
@override
def on_data_collectors_attached(self) -> None:
    """Called when data collectors are attached to the agent."""
    self.experience_collector = self.get_data_collector("experience")

@override
def step(self, observation: list[float]) -> int:
    # Use the model for decision making
    action = self.policy_model(observation)

    # Collect experience for training
    self.experience_collector.collect({
        "observation": observation,
        "action": action
    })

    return action
```

## Environment Implementation

Environments provide observations to agents and receive actions. To implement a custom environment:

```python
from pamiq_core import Environment
from typing import override

class MyEnvironment(Environment[list[float], int]):
    """Custom environment that provides float list observations and accepts integer actions."""

    def __init__(self):
        self.state = [0.0, 0.0, 0.0]

    @override
    def observe(self) -> list[float]:
        """Return the current observation.

        Returns:
            Current environment state as observation
        """
        return self.state.copy()

    @override
    def affect(self, action: int) -> None:
        """Apply the action to change the environment state.

        Args:
            action: Action to apply to the environment
        """
        # Simple environment dynamics
        if action == 1:
            self.state[0] += 0.1
        else:
            self.state[0] -= 0.1
```

## Interaction Class

The `Interaction` class connects an agent and environment, implementing the core interaction loop:

```python
from pamiq_core import Interaction, Agent, Environment

# Create agent and environment
agent = MyAgent()
environment = MyEnvironment()

# Create interaction
interaction = Interaction(agent, environment)

# Manual stepping (normally handled by the system)
interaction.setup()
for _ in range(10):
    interaction.step()  # Performs observe-think-act cycle
interaction.teardown()
```

### Fixed Interval Interaction

For real-time systems, you often need to maintain a consistent timing for the interaction loop. The `FixedIntervalInteraction` class provides this functionality:

```python
from pamiq_core import FixedIntervalInteraction

# Create a fixed interval interaction running at 10Hz (every 0.1 seconds)
interaction = FixedIntervalInteraction.with_sleep_adjustor(
    agent,
    environment,
    interval=0.1
)
```

The `with_sleep_adjustor` factory method is the most common way to create a fixed interval interaction, as it uses CPU sleep to maintain timing.

## Modular Environment

The `ModularEnvironment` class breaks down environment functionality into sensor and actuator components:

```python
from pamiq_core import ModularEnvironment, Sensor, Actuator
from typing import override

class CameraSensor(Sensor[list[float]]):
    @override
    def read(self) -> list[float]:
        """Read camera data."""
        # Code to capture and process image
        return [0.1, 0.2, 0.3]  # Processed image features

class MotorActuator(Actuator[int]):
    @override
    def operate(self, action: int) -> None:
        """Control motors based on action."""
        # Code to control motors
        print(f"Moving motors with command {action}")

# Create modular environment
environment = ModularEnvironment(CameraSensor(), MotorActuator())
```

This approach enables reusable components and cleaner separation of concerns.

## Wrappers

Wrappers transform data flowing between components. PAMIQ-Core provides wrappers for environments, sensors, and actuators:

```python
from pamiq_core import EnvironmentWrapper

# Create a wrapper to normalize observations
def normalize_observation(obs: list[float]) -> list[float]:
    return [x / 10.0 for x in obs]

# Create a wrapper to scale actions
def scale_action(action: int) -> int:
    return action * 2

# Wrap the environment
wrapped_env = EnvironmentWrapper(
    environment,
    obs_wrapper=normalize_observation,
    act_wrapper=scale_action,
)
```

For more complex transformations, you can create a custom wrapper:

```python
from pamiq_core import Wrapper
from typing import override

class ImageProcessingWrapper(Wrapper[list[float], list[float]]):
    @override
    def wrap(self, image: list[float]) -> list[float]:
        """Apply image processing transformations."""
        # Complex image processing logic
        return [x * 0.5 + 0.1 for x in image]
```

## Common Event Hooks

PAMIQ-Core components have common event hooks that you can override:

### Interaction Events

```python
@override
def setup(self) -> None:
    """Called when the interaction starts."""
    super().setup()  # Always call the parent method
    print("Setting up resources...")

@override
def teardown(self) -> None:
    """Called when the interaction ends."""
    super().teardown()  # Always call the parent method
    print("Cleaning up resources...")
```

### Thread Events

```python
@override
def on_paused(self) -> None:
    """Called when the system is paused."""
    super().on_paused()  # Always call the parent method
    print("System paused, suspending external connections...")

@override
def on_resumed(self) -> None:
    """Called when the system is resumed."""
    super().on_resumed()  # Always call the parent method
    print("System resumed, restoring external connections...")
```

These event hooks enable proper lifecycle management and allow components to respond to system state changes.

## API Reference

More details, Checkout to the [API Reference](../api/interaction.md)
