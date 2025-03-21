import copy
from threading import RLock
from typing import Any, Protocol, override

import Path
import torch
import torch.nn as nn

from pamiq_core.model import InferenceModel, TrainingModel

CPU_DEVICE = torch.device("cpu")


class InferenceProcedureCallable(Protocol):
    """Typing for `inference_procedure` argument of TorchTrainingModel because
    `typing.Callable` can not typing `*args` and `**kwds`."""

    def __call__(
        self, inference_model: "TorchInferenceModel[Any]", *args: Any, **kwds: Any
    ) -> Any: ...


def get_device[T](
    module: nn.Module, default_device: T = CPU_DEVICE
) -> torch.device | T:
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
    return default_device


def default_infer_procedure(module: nn.Module, *args: Any, **kwds: Any) -> Any:
    """Default inference forward flow.

    Tensors in `args` and `kwds` are sent to the computing device. If
    you override this method, be careful to send the input tensor to the
    computing device.
    """
    device = get_device(module, CPU_DEVICE)
    new_args, new_kwds = [], {}
    for i in args:
        if isinstance(i, torch.Tensor):
            i = i.to(device)
        new_args.append(i)

    for k, v in kwds.items():
        if isinstance(v, torch.Tensor):
            v = v.to(device)
        new_kwds[k] = v

    return module(*new_args, **new_kwds)


class TorchInferenceModel[T: nn.Module](InferenceModel):
    """Wrapper class for torch model to infer in InferenceThread."""

    def __init__(
        self, model: T, inference_procedure: InferenceProcedureCallable
    ) -> None:
        """Initialize.

        Args:
            model: A torch model for inference.
            inference_procedure: An inference procedure as Callable.
        """
        self._raw_model = model
        self._inference_procedure = inference_procedure
        self._lock = RLock()

    @property
    def _raw_model(self) -> T:
        """Returns the internal dnn model.

        Do not access this property in the inference thread. This
        property is used to switch the model between training and
        inference model."
        """
        return self._raw_model

    @_raw_model.setter
    def _raw_model(self, m: T) -> None:
        """Sets the model in a thread-safe manner."""
        with self._lock:
            self._raw_model = m

    @torch.inference_mode()
    def infer(self, *args: Any, **kwds: Any) -> Any:
        """Performs the inference in a thread-safe manner."""
        with self._lock:
            return self._inference_procedure(self._raw_model, *args, **kwds)


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
        default_device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        inference_procedure: InferenceProcedureCallable = default_infer_procedure,
        parameter_file: str | Path | None = None,
    ):
        """Initialize.

        Args:
            model: A torch model.
            has_inference_model: Whether to have inference model.
            inference_thread_only: Whether it is an inference thread only.
            default_device: A device if any device not found.
            dtype: Data type of the model.
            inference_procedure: An inference procedure as Callable.
            parameter_file:
        """
        super().__init__(has_inference_model, inference_thread_only)
        if dtype is not None:
            model = model.type(dtype)
        self._raw_model = model
        if (
            default_device is None
        ):  # prevents from moving the model to cpu unintentionally.
            default_device = get_device(model, CPU_DEVICE)
        self._default_device = torch.device(default_device)
        self._inference_procedure = inference_procedure

        self._raw_model.to(self._default_device)
        if self.parameter_file is not None:
            ...  # パラメータ読み込み

    @override
    def create_inference(self) -> TorchInferenceModel[T]:
        """Create inference model.

        Returns:
            TorchInferenceModel.
        """
        model = self._raw_model
        if not self.inference_thread_only:  # the model does not need to be copied to training thread If it is used only in the inference thread.
            model = copy.deepcopy(model)
        return TorchInferenceModel(model)

    @override
    def sync_impl(self, inference_model: TorchInferenceModel[T]) -> None:
        """Copies params of training model to self._inference_model.

        Args:
            inference_model: InferenceModel to sync.
        """

        eval_of_raw_model = getattr(
            self._raw_model, "eval"
        )  # To pass python-no-eval check.
        eval_of_raw_model()

        # Hold the grads.
        grads: list[torch.Tensor | None] = []
        for p in self._raw_model.parameters():
            grads.append(p.grad)
            p.grad = None

        # Swap the training model and the inference model.
        self._raw_model, inference_model._raw_model = (
            inference_model._raw_model,
            self._raw_model,
        )
        self._raw_model.load_state_dict(self.inference_model._raw_model.state_dict())

        # Assign the model grads.
        for i, p in enumerate(self._raw_model.parameters()):
            p.grad = grads[i]

        self._raw_model.train()

    @override
    def forward(self, *args: Any, **kwds: Any) -> Any:
        """forward."""
        return self._raw_model(*args, **kwds)
