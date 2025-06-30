# Trainer

The `trainer` module defines how models are trained during system operation. It manages the training process, decides when to train, and handles synchronization between training and inference models.

## Basic Concepts

Trainers in PAMIQ-Core operate in the training thread, processing data collected during inference. The key responsibilities of a trainer include:

1. **Deciding when to train**: Using data availability conditions to determine if training should occur
2. **Executing training logic**: Implementing the actual learning algorithm
3. **Synchronizing models**: Ensuring updated parameters are available for inference

The training process flows as follows:

1. Check if training conditions are met via `is_trainable()`
2. Run setup via `setup()`
3. Execute training via `train()`
4. Synchronize models via `sync_models()`
5. Clean up resources via `teardown()`

## Implementing Custom Trainers

To implement a custom trainer, you need to extend the `Trainer` class and override the `train` method:

```python
from pamiq_core import Trainer
from typing import override

class MyTrainer(Trainer):
    """Custom trainer that implements a simple learning algorithm."""

    @override
    def train(self) -> None:
        """Implement the training logic.

        This method is called repeatedly by the training thread when
        conditions for training are met.
        """
        # Get the model
        model = self.get_training_model("policy_model")

        # Training logic
        for _ in range(10):  # 10 optimization steps
            # ... implement training algorithm
            pass

        # Model parameters will be automatically synchronized after training
```

### Training Conditions

PAMIQ-Core trainers can be configured with conditions to determine when training should occur:

```python
# Create a trainer that only trains when:
# 1. The "experience" buffer has at least 1000 samples
# 2. At least 100 new samples have been collected since last training
trainer = MyTrainer(
    training_condition_data_user="experience",
    min_buffer_size=1000,
    min_new_data_count=100
)
```

You can also implement custom training conditions by overriding the `is_trainable` method:

```python
@override
def is_trainable(self) -> bool:
    """Determine if training should be executed.

    Returns:
        True if training should proceed, False otherwise
    """
    # Check parent condition first
    if not super().is_trainable():
        return False

    # Add custom condition - only train during daytime
    current_hour = datetime.now().hour
    return 8 <= current_hour <= 20
```

### Accessing Training Models

To access models for training, override the `on_training_models_attached` callback:

```python
@override
def on_training_models_attached(self) -> None:
    """Called when training models are attached to the trainer."""
    # Store references to models for convenient access
    self.policy_model = self.get_training_model("policy")
    self.value_model = self.get_training_model("value")
```

### Using Experience Data

To access collected data for training, override the `on_data_users_attached` callback:

```python
@override
def on_data_users_attached(self) -> None:
    """Called when data users are attached to the trainer."""
    # Store references to data users for convenient access
    self.experience_data = self.get_data_user("experience")

@override
def train(self) -> None:
    """Train models using collected experience data."""
    # Update to get latest data
    self.experience_data.update()

    # Access the data
    data = self.experience_data.get_data()
    states = data["state"]
    actions = data["action"]
    rewards = data["reward"]

    # Use the data to train models
    # ...
```

## Lifecycle Callbacks

PAMIQ-Core trainers have several lifecycle callbacks that you can override:

### Setup and Teardown

```python
@override
def setup(self) -> None:
    """Called before training starts.

    Use this to initialize resources needed for training.
    """
    super().setup()  # Always call the parent method

    # Initialize training resources
    self.optimizer = SomeOptimizer(self.policy_model.parameters())
    self.batch_size = 64

@override
def teardown(self) -> None:
    """Called after training finishes.

    Use this to clean up resources after training.
    """
    super().teardown()  # Always call the parent method

    # Clean up resources
    self.optimizer = None
```

## Thread Events

As with other PAMIQ-Core components, trainers can respond to system pause and resume events:

```python
@override
def on_paused(self) -> None:
    """Called when the system is paused."""
    super().on_paused()  # Always call the parent method

    # Pause external connections or resources
    if hasattr(self, 'external_service'):
        self.external_service.pause()

@override
def on_resumed(self) -> None:
    """Called when the system is resumed."""
    super().on_resumed()  # Always call the parent method

    # Resume external connections or resources
    if hasattr(self, 'external_service'):
        self.external_service.resume()
```

These event hooks enable proper resource management in response to system state changes.

## PyTorch Integration 🔥

For training deep learning models, PAMIQ-Core provides specialized trainer classes in the `torch` submodule.

For more details on using PyTorch with PAMIQ-Core, see the [PyTorch Integration Guide](./torch.md).

## API Reference

More details, Checkout to the [API Reference](../api/trainer.md)
