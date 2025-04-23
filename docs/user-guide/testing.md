# Test

## Testing Tools

The `testing` module provides tools designed to simplify testing of applications built with the PAMIQ-Core framework. These utilities help streamline the process of connecting components and setting up proper test environments.

### Connect Components

When testing with PAMIQ-Core, components like Trainers and Agents need to have DataUsers, Collectors, TrainingModels, and InferenceModels attached before they can be properly used. The `testing.connect_components` function simplifies this process:

```python
from pamiq_core.testing import connect_components

# Set up components
agent = MyAgent()
trainer = MyTrainer()
buffer = MyDataBuffer(["state", "action", "reward"], max_size=1000)
model = MyTrainingModel()

# Connect all components in one call
components = connect_components(
    agent=agent,
    trainers=trainer,
    buffers={"experience": buffer},
    models={"policy": model}
)

# Now you can access the connected components
data_users = components.data_users
data_collectors = components.data_collectors
training_models = components.training_models
inference_models = components.inference_models
```

The function returns a `ConnectedComponents` named tuple containing all the connected component dictionaries for easy access in your tests.

### API Reference

For more detailed information, check out the [API Reference](../api/testing.md)

## Testing Agents

When testing an Agent implementation, you need to follow this sequence to ensure proper initialization and cleanup:

1. Connect components
2. Call setup
3. Test the step method
4. Call teardown

For example:

```python
def test_my_agent():
    # Create and connect components
    agent = MyAgent()
    buffer = MyDataBuffer(["state", "action"], max_size=100)
    model = MyTrainingModel()

    connect_components(
        agent=agent,
        buffers={"experience": buffer},
        models={"policy": model}
    )

    # Initialize the agent
    agent.setup()

    # Test the step method
    observation = [0.1, 0.2, 0.3]
    action = agent.step(observation)

    # Verify the action is as expected
    assert action == 1

    # Clean up
    agent.teardown()
```

Remember that the `step` method can only be properly tested after components have been connected and `setup` has been called.

## Testing Trainers

Testing a Trainer implementation follows a similar pattern, with a specific sequence:

1. Connect components
2. Call setup
3. Test the train method
4. Call sync_models
5. Call teardown

Here's an example:

```python
def test_my_trainer():
    # Create and connect components
    trainer = MyTrainer()
    buffer = MyDataBuffer(["state", "action", "reward"], max_size=100)
    model = MyTrainingModel()

    components = connect_components(
        trainers=trainer,
        buffers={"experience": buffer},
        models={"policy": model}
    )

    # Initialize the trainer
    trainer.setup()

    # Must check the trainer is trainable.
    assert trainer.is_trainable()

    # Test the train method
    trainer.train()

    # Sync the models
    trainer.sync_models()

    # Verify training effects
    # ...

    # Clean up
    trainer.teardown()
```

While you can use the `run` method to execute the entire workflow (setup → train → sync_models → teardown), note that it will only execute if `is_trainable()` returns `True`. This is important to remember when setting up your test data.

```python
# Alternative approach using run
def test_trainer_run():
    # Create and connect components
    trainer = MyTrainer()
    buffer = MyDataBuffer(["state", "action", "reward"], max_size=100)
    model = MyTrainingModel()

    connect_components(
        trainers=trainer,
        buffers={"experience": buffer},
        models={"policy": model}
    )

    # Prepare test data to ensure is_trainable returns True
    # ...

    # Run the entire training workflow
    result = trainer.run()

    # Verify run was successful
    assert result is True

    # Verify training effects
    # ...
```
