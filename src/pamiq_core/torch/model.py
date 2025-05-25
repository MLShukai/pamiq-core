import copy
from pathlib import Path
from threading import RLock
from typing import Any, Protocol, override

import torch
import torch.nn as nn

from pamiq_core.model import InferenceModel, TrainingModel


class InferenceProcedureCallable[T: nn.Module](Protocol):
    """Typing for `inference_procedure` argument of TorchTrainingModel because
    `typing.Callable` can not typing `*args` and `**kwds`."""

    def __call__(self, model: T, /, *args: Any, **kwds: Any) -> Any:
        pass


def get_device(
    module: nn.Module, default_device: torch.device | None = None
) -> torch.device:
    """Retrieves the device where the module runs.

    Args:
        module: A module that you want to know which device it runs on.
        default_device: A device to return if any device not found.
    Returns:
        A device that the module uses or default_device.
    """
    for param in module.parameters():
        return param.device
    for buf in module.buffers():
        return buf.device
    if default_device is None:
        default_device = torch.get_default_device()
    return default_device


def default_infer_procedure(model: nn.Module, *args: Any, **kwds: Any) -> Any:
    """Default inference procedure.

    Tensors in `args` and `kwds` are sent to the computing device. If
    you override this method, be careful to send the input tensor to the
    computing device.
    """
    device = get_device(model)
    new_args: list[Any] = []
    new_kwds: dict[str, Any] = {}
    for i in args:
        if isinstance(i, torch.Tensor):
            i = i.to(device)
        new_args.append(i)

    for k, v in kwds.items():
        if isinstance(v, torch.Tensor):
            v = v.to(device)
        new_kwds[k] = v

    return model(*new_args, **new_kwds)


class TorchInferenceModel[T: nn.Module](InferenceModel):
    """Wrapper class for torch model to infer in InferenceThread."""

    def __init__(
        self, model: T, inference_procedure: InferenceProcedureCallable[T]
    ) -> None:
        """Initialize.

        Args:
            model: A torch model for inference.
            inference_procedure: An inference procedure as Callable.
        """
        self._model = model
        self._inference_procedure = inference_procedure
        self._lock = RLock()

    @property
    def _raw_model(self) -> T:
        """Returns the internal dnn model.

        Do not access this property in the inference thread. This
        property is used to switch the model between training and
        inference model."
        """
        return self._model

    @_raw_model.setter
    def _raw_model(self, m: T) -> None:
        """Sets the model in a thread-safe manner."""
        with self._lock:
            self._model = m

    @torch.inference_mode()
    @override
    def infer(self, *args: Any, **kwds: Any) -> Any:
        """Performs the inference in a thread-safe manner."""
        with self._lock:
            return self._inference_procedure(self._model, *args, **kwds)


class TorchTrainingModel[T: nn.Module](TrainingModel[TorchInferenceModel[T]]):
    """Wrapper class for training torch model in TrainingThread.

    Needed for multi-thread training and inference in parallel.
    """

    @override
    def __init__(
        self,
        model: T,
        has_inference_model: bool = True,
        inference_thread_only: bool = False,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        inference_procedure: InferenceProcedureCallable[T] = default_infer_procedure,
        pretrained_parameter_file: str | Path | None = None,
        compile: bool = False,
    ):
        """Initialize.

        Args:
            model: A torch model.
            has_inference_model: Whether to have inference model.
            inference_thread_only: Whether it is an inference thread only.
            device: A device on which the model is placed.
            dtype: Data type of the model.
            inference_procedure: An inference procedure as Callable.
            pretrained_parameter_file: Path to a pre-trained model parameter file to load. If provided, the model parameters will be loaded from this file.
            compile: Whether to compile torch model.
        """
        super().__init__(has_inference_model, inference_thread_only)
        if dtype is not None:
            model = model.type(dtype)
        self.model = model
        if device is None:  # prevents from moving the model to cpu unintentionally.
            device = get_device(model)
        self._inference_procedure = inference_procedure
        self.model.to(device)

        if pretrained_parameter_file is not None:
            self.model.load_state_dict(
                torch.load(pretrained_parameter_file, map_location=device)  # pyright: ignore[reportUnknownMemberType]
            )

        if compile:
            if self.has_inference_model:
                # copy before compile
                self.inference_model._raw_model.compile()  # pyright: ignore[reportPrivateUsage, reportUnknownMemberType]
            self.model.compile()  # pyright: ignore[reportUnknownMemberType, ]

    @override
    def _create_inference_model(self) -> TorchInferenceModel[T]:
        """Create inference model.

        Returns:
            TorchInferenceModel.
        """
        model = self.model
        if not self.inference_thread_only:  # the model does not need to be copied to training thread If it is used only in the inference thread.
            model = copy.deepcopy(model)
        return TorchInferenceModel(model, self._inference_procedure)

    @override
    def sync_impl(self, inference_model: TorchInferenceModel[T]) -> None:
        """Copies params of training model to self._inference_model.

        Args:
            inference_model: InferenceModel to sync.
        """

        self.model.eval()

        # Hold the grads.
        grads: list[torch.Tensor | None] = []
        for p in self.model.parameters():
            grads.append(p.grad)
            p.grad = None

        # Swap the training model and the inference model.
        self.model, inference_model._raw_model = (  # pyright: ignore[reportPrivateUsage]
            inference_model._raw_model,  # pyright: ignore[reportPrivateUsage]
            self.model,
        )
        self.model.load_state_dict(
            self.inference_model._raw_model.state_dict()  # pyright: ignore[reportPrivateUsage]
        )

        # Assign the model grads.
        for i, p in enumerate(self.model.parameters()):
            p.grad = grads[i]

        self.model.train()

    @override
    def forward(self, *args: Any, **kwds: Any) -> Any:
        """forward."""
        return self.model(*args, **kwds)

    @override
    def save_state(self, path: Path) -> None:
        """Save the model params.

        Args:
            path: Path where the states should be saved.
        """
        torch.save(self.model.state_dict(), f"{path}.pt")  # pyright: ignore[reportUnknownMemberType]

    @override
    def load_state(self, path: Path) -> None:
        """Load the model params.

        Args:
            path: Path where the states should be loaded.
        """
        self.model.load_state_dict(torch.load(f"{path}.pt"))  # pyright: ignore[reportUnknownMemberType]
