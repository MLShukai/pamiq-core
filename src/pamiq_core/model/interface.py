from abc import ABC, abstractmethod
from typing import Any


class InferenceModel(ABC):
    """Wrapper class for model to infer.

    Needed for multi-thread training and inference in parallel.
    """

    @abstractmethod
    def infer(self, *args, **kwds) -> Any:
        """Perform inference using a model.

        Args:
            *args:
                Positional arguments required for inference.
            **kwds:
                Keyword arguments required for inference.
        Returns:
            Any:
                The result of the inference.
        """
        raise NotImplementedError

    def __call__(self, *args, **kwds) -> Any:
        """Perform inference using a model.

        Args:
            *args:
                Positional arguments required for inference.
            **kwds:
                Keyword arguments required for inference.
        Returns:
            Any:
                The result of the inference.
        """
        return self.infer(*args, **kwds)


class TrainingModel(ABC):
    """Wrapper class to train model.

    Needed for multi-thread training and inference in parallel.
    """

    def __init__(self, has_inference: bool = True, inference_only: bool = False):
        """Initialize.

        Args:
            has_inference (bool):
                Whether to have InferenceModel.
                Default to True.
            inference_only (bool):
                Whether to do Inference only.
                Default to False.
        """
        if inference_only and not has_inference:
            raise ValueError
        self.has_inference = has_inference
        self.Inference_only = inference_only

    _inference_model: InferenceModel | None = None

    @property
    def inference_model(self) -> InferenceModel:
        """Get inference model.

        Returns:
            InferenceModel: A model to infer.
        """
        if not self.has_inference:
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
        """Running model for training.

        Args:
            *args:
                Positional arguments required for training.
            **kwds:
                Keyword arguments required for training.
        Returns:
            Any:
                Result of running the model.
        """
        raise NotImplementedError

    def __call__(self, *args, **kwds) -> Any:
        """Running model for training.

        Args:
            *args:
                Positional arguments required for training.
            **kwds:
                Keyword arguments required for training.
        Returns:
            Any:
                Result of running the model.
        """
        return self.forward(*args, **kwds)
