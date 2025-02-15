from abc import ABC, abstractmethod
from typing import Any


class InferenceModel(ABC):
    @abstractmethod
    def infer(self, *args, **kwds) -> Any:
        raise NotImplementedError

    def __call__(self, *args, **kwds) -> Any:
        return self.infer(*args, **kwds)


class TrainingModel(ABC):
    def __init__(self, has_inference: bool = True, inference_only: bool = False):
        if inference_only and not has_inference:
            raise ValueError
        self.has_inference = has_inference
        self.Inference_only = inference_only

    _inference_model: InferenceModel | None = None

    @property
    def inference_model(self) -> InferenceModel:
        if not self.has_inference:
            raise RuntimeError

        if self._inference_model is None:
            self._inference_model = self._create_inference_model()
        return self._inference_model

    @abstractmethod
    def _create_inference_model(self) -> InferenceModel:
        raise NotImplementedError

    @abstractmethod
    def forward(self, *args, **kwds) -> Any:
        raise NotImplementedError

    def __call__(self, *args, **kwds) -> Any:
        return self.forward(*args, **kwds)
