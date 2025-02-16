from abc import ABC, abstractmethod
from typing import Any


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


class TrainingModel(ABC):
    """Base interface class to train model in TrainingThread.

    Needed for multi-thread training and inference in parallel.
    """

    def __init__(self, has_inference_model: bool = True, inference_only: bool = False):
        """Initialize the TrainingModel.

        Args:
            has_inference_model: Whether to have InferenceModel.
            inference_only: Whether to do Inference only.
        """
        if inference_only and not has_inference_model:
            raise ValueError
        self.has_inference_model = has_inference_model
        self.inference_only = inference_only

    _inference_model: InferenceModel | None = None

    @property
    def inference_model(self) -> InferenceModel:
        """Get inference model."""
        if not self.has_inference_model:
            raise RuntimeError

        if self._inference_model is None:
            self._inference_model = self._create_inference_model()
        return self._inference_model

    @abstractmethod
    def _create_inference_model(self) -> InferenceModel:
        """Create inference model.

        Returns:
            InferenceModel: A model to infer.
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
