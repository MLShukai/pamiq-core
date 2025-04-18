# Model

The `model` module in PAMIQ-Core defines the interfaces for inference and training models, providing a framework for managing model synchronization in a multi-threaded environment. This allows inference to run continuously while training occurs in the background.

## Basic Concepts

PAMIQ-Core uses two distinct model representations:

1. **InferenceModel**: Used for making predictions in the inference thread
2. **TrainingModel**: Used for training in the training thread

These two models are synchronized after training, allowing the inference model to benefit from ongoing training without interrupting the agent's decision-making process.

## Implementing Custom Models

### InferenceModel

The `InferenceModel` interface defines models that can be used for inference in the agent's decision-making process:

```python
from pamiq_core import InferenceModel
from typing import override

class MyInferenceModel(InferenceModel):
    """A simple custom inference model."""

    def __init__(self, weights: list[float]):
        self.weights = weights

    @override
    def infer(self, features: list[float]) -> float:
        """Make a prediction using the model.

        Args:
            features: Input features for inference

        Returns:
            Prediction result
        """
        return sum(w * x for w, x in zip(self.weights, features))
```

The key method to implement is `infer()`, which takes input data and returns the model's prediction.

### TrainingModel

The `TrainingModel` interface defines models that can be trained and synchronized with inference models:

```python
from pamiq_core import TrainingModel, InferenceModel
from typing import override

class MyTrainingModel(TrainingModel[MyInferenceModel]):
    """A simple custom training model."""

    def __init__(self):
        super().__init__(has_inference_model=True, inference_thread_only=False)
        self.weights = [0.5, 0.3, -0.2]

    @override
    def _create_inference_model(self) -> MyInferenceModel:
        """Create a new inference model instance.

        Returns:
            A new inference model with current weights
        """
        return MyInferenceModel(self.weights.copy)

    @override
    def forward(self, features: list[float]) -> float:
        """Forward pass for training.

        Args:
            features: Input features

        Returns:
            Output of the model
        """
        return sum(w * x for w, x in zip(self.weights, features))

    @override
    def sync_impl(self, inference_model: MyInferenceModel) -> None:
        """Synchronize parameters from training model to inference model.

        Args:
            inference_model: Inference model to update
        """
        inference_model.weights = self.weights.copy()
```

Key methods to implement:

- `_create_inference_model()`: Creates a new inference model instance
- `forward()`: Performs the forward pass during training
- `sync_impl()`: Synchronizes parameters from the training model to the inference model

### Configuration Options

The `TrainingModel` class takes two important configuration parameters:

- **has_inference_model** (default: `True`): Determines whether the training model creates and manages an associated inference model.

    - When `True`: The model will create an inference model that can be used in the inference thread
    - When `False`: No inference model is created, and the model can only be used for training

- **inference_thread_only** (default: `False`): Determines how the model is used in the thread architecture.

    - When `True`: The model is used only in the inference thread and not modified by training. This is useful for pre-trained models that don't need to be updated.
    - When `False`: The model is used for both training and inference, with parameters synchronized between threads.

These two parameters cannot be set to `has_inference_model=False` and `inference_thread_only=True` simultaneously, as this would create a model that can't be used in either thread.

### Model Synchronization

PAMIQ-Core's training thread automatically synchronizes parameters between training and inference models:

1. The training thread updates model parameters
2. The `sync()` method is called after training
3. This triggers `sync_impl()` to copy parameters to the inference model
4. The inference thread continues using the updated model

## Thread Safety

The model architecture in PAMIQ-Core ensures thread safety through careful parameter synchronization:

- Inference models are read-only in the inference thread
- Training models are modified only in the training thread
- Parameter copying happens through a controlled synchronization process
- Concurrent access to models is managed to prevent race conditions

This design allows for continuous inference while training progresses in the background.

## PyTorch Integration 🔥

For deep learning models, PAMIQ-Core offers seamless integration with PyTorch through the `torch` submodule. This provides specialized implementations of `InferenceModel` and `TrainingModel` that handle PyTorch model synchronization efficiently.

For more details on using PyTorch with PAMIQ-Core, see the [PyTorch Integration Guide](./torch.md).

## API Reference

More details, Checkout to the [API Reference](../api/interaction.md)
