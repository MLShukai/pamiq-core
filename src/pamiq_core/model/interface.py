from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

# Define the type restricted to InferenceModel subclasses
T = TypeVar("T", bound="InferenceModel")


class InferenceModel(ABC):
    """Base interface class for model to infer in InferenceThread.

    Needed for multi-thread training and inference in parallel.
    """

    @abstractmethod
    def infer(self, *args, **kwds) -> Any:
        """Perform inference using a model.

        Args:
            *args: Positional arguments required for inference.
            **kwds: Keyword arguments required for inference.
        Returns:
            Any: The result of the inference.
        """
        raise NotImplementedError

    def __call__(self, *args, **kwds) -> Any:
        """Perform inference using a model.

        Args:
            *args:  Positional arguments required for inference.
            **kwds: Keyword arguments required for inference.
        Returns:
            Any: The result of the inference.
        """
        return self.infer(*args, **kwds)


class TrainingModel(Generic[T], ABC):
    """Base interface class to train model in TrainingThread.

    Needed for multi-thread training and inference in parallel.
    """

    def __init__(self, has_inference_model: bool = True, inference_only: bool = False):
        """Initialize the TrainingModel.

        Args:
            has_inference_model: Whether to have inference model.
            inference_only: Whether to do Inference only.
        """
        if (not has_inference_model) and (inference_only):
            raise ValueError
        self.has_inference_model = has_inference_model
        self.inference_only = inference_only

    _inference_model: T | None = None

    @property
    def inference_model(self) -> T:
        """Get inference model."""
        if not self.has_inference_model:
            raise RuntimeError

        if self._inference_model is None:
            self._inference_model = self._create_inference_model()
        return self._inference_model

    @abstractmethod
    def _create_inference_model(self) -> T:
        """Create inference model.

        Returns:
            T: A model to infer.
        """
        raise NotImplementedError

    @abstractmethod
    def forward(self, *args, **kwds) -> Any:
        """Forward path of model.

        Args:
            *args: Positional arguments required for forward path.
            **kwds: Keyword arguments required for forward path.
        Returns:
            Result of forward path of the model.
        """
        raise NotImplementedError

    def __call__(self, *args, **kwds) -> Any:
        """Calls `forward` method."""
        return self.forward(*args, **kwds)

    def sync(self) -> None:
        """Copies params of training model to self._inference_model if
        needed."""
        if self._need_sync:
            self.sync_model(self.inference_model)

    @property
    def _need_sync(self) -> bool:
        """Return whether It is necessary to synchronize training model and
        inference model."""
        return self.has_inference_model and (not self.inference_only)

    def sync_model(self, inference_model: T) -> None:
        """Copies params of training model to self._inference_model if needed.

        Example:
            class TrainingTorchModel(TrainingModel[InferenceTorchModel]):
                @override
                def sync_model(self, inference_model: InferenceTorchModel) -> None:
                    with torch.no_grad():
                        for inference_param, training_param in zip(
                            inference_model.model.parameters(),
                            self.model.parameters(),
                        ):
                            inference_param.copy_(training_param)
        """
        raise NotImplementedError
