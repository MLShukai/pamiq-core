# Schedulers

`utils.schedulers` in PAMIQ-Core are used to execute callbacks at specified intervals. They provide a flexible way to implement periodic actions such as model saving, metric logging, or resource cleanup.

## Base Scheduler

The `Scheduler` abstract base class provides common functionality for all scheduler implementations:

```python
from pamiq_core.utils.schedulers import Scheduler

class MyScheduler(Scheduler):
    def __init__(self, callbacks=None):
        super().__init__(callbacks)
        self.ready = False

    def is_available(self):
        return self.ready

    def update(self):
        # Set ready to True when some condition is met
        self.ready = check_condition()
        # Call parent update which will execute callbacks if is_available() returns True
        super().update()
```

The key components of a scheduler are:

1. **Callbacks**: Functions to execute when the scheduler is triggered
2. **Availability Check**: Determines when callbacks should be executed
3. **Update Method**: Called regularly to check availability and execute callbacks when appropriate

## Time Interval Scheduler

The `TimeIntervalScheduler` executes callbacks at fixed time intervals:

```python
from pamiq_core.utils.schedulers import TimeIntervalScheduler

# Create a scheduler that executes callbacks every 60 seconds
def log_metrics():
    print("Logging metrics...")

metrics_scheduler = TimeIntervalScheduler(60.0, log_metrics)

# In your main loop:
while running:
    # ... other code
    metrics_scheduler.update()
    # ... more code
```

This scheduler is useful for:

- Periodic saving of model checkpoints
- Regular logging of metrics
- Environment cleanup at fixed intervals
- Any task that should occur based on elapsed real time

## Step Interval Scheduler

The `StepIntervalScheduler` executes callbacks after a specified number of steps:

```python
from pamiq_core.utils.schedulers import StepIntervalScheduler

# Create a scheduler that executes callbacks every 100 steps
def evaluate_model():
    print("Evaluating model performance...")

eval_scheduler = StepIntervalScheduler(100, evaluate_model)

# In your training loop:
for step in range(total_steps):
    # ... training code
    eval_scheduler.update()  # Will trigger every 100 calls
    # ... more training code
```

This scheduler is useful for:

- Model evaluation at regular intervals during training
- Gradient accumulation in deep learning
- Periodic data sampling or augmentation
- Any task that should occur based on iteration count

## Registering Multiple Callbacks

You can register multiple callbacks with a single scheduler:

```python
from pamiq_core.utils.schedulers import TimeIntervalScheduler

# Create scheduler with initial callback
scheduler = TimeIntervalScheduler(300.0, save_checkpoint)

# Register additional callbacks
scheduler.register_callback(log_performance)
scheduler.register_callback(clean_memory)

# All three callbacks will execute every 5 minutes
```

You can also remove callbacks when they are no longer needed:

```python
# Remove a specific callback
scheduler.remove_callback(clean_memory)
```

## Integration with PAMIQ-Core Components

Schedulers are used throughout PAMIQ-Core for various periodic tasks:

```python
from pamiq_core.utils.schedulers import TimeIntervalScheduler

class MyTrainer(Trainer):
    def __init__(self):
        super().__init__()
        # Create scheduler for model evaluation
        self.eval_scheduler = TimeIntervalScheduler(
            interval=300.0,  # Every 5 minutes
            callbacks=self.evaluate_model
        )

    def train(self):
        # Training code...
        self.eval_scheduler.update()
        # More training code...

    def evaluate_model(self):
        # Model evaluation logic
        pass
```

## API Reference

More details, Checkout to the [API Reference](../api/schedulers.md)
