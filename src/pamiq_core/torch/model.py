import copy
from pathlib import Path
from threading import RLock
from typing import Any, Protocol, override

import torch
import torch.nn as nn

from pamiq_core.model import InferenceModel, TrainingModel


class InferenceProcedureCallable[T: nn.Module](Protocol):
    """This protocol defines inference procedures called by TorchTrainingModel.

    The callable must have args below:
    - model: The neural network model instance
    - *args: Variable positional arguments to pass to the model
    - **kwds: Variable keyword arguments to pass to the model

    Returns:
        The output from the model's forward pass.
    """

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
    """Default inference procedure with device placement.

    This function automatically moves tensor arguments to the same device
    as the model before performing inference. Non-tensor arguments are
    passed through unchanged.

    Args:
        model: The model to infer.
        *args: Positional arguments to pass to the model. Tensors will be
            moved to the model's device.
        **kwds: Keyword arguments to pass to the model. Tensor values will
            be moved to the model's device.

    Returns:
        The output from the model's forward pass.

    Note:
        When overriding this method, ensure that input tensors are properly
        sent to the correct device to avoid device mismatch.
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


class UnwrappedContextManager[T: nn.Module]:
    """Context manager for accessing the raw PyTorch model with thread safety.

    This context manager provides direct access to the underlying
    PyTorch model while ensuring thread safety through locking and
    optionally enabling/disabling inference mode.
    """

    def __init__(self, model: T, lock: RLock, inference_mode: bool) -> None:
        """Initialize the context manager.

        Args:
            model: The PyTorch model to provide access to.
            lock: The lock to use for thread synchronization.
            inference_mode: If True, torch.inference_mode will be enabled
                during the context, disabling gradient computation. If False,
                gradients will be computed normally.
        """
        self._model = model
        self._lock = lock
        self._inference_mode = inference_mode

    def __enter__(self) -> T:
        """Enter the context and return the model.

        Acquires the lock and optionally enables inference mode before
        returning the model for direct access.

        Returns:
            The PyTorch model for direct manipulation.
        """
        self._torch_inference_mode = torch.inference_mode(self._inference_mode)
        self._torch_inference_mode.__enter__()
        self._lock.acquire()
        return self._model

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        """Exit the context and release resources.

        Exits the inference mode context (if enabled) and releases the
        lock.
        """
        self._lock.release()
        self._torch_inference_mode.__exit__(exc_type, exc_value, traceback)


class TorchInferenceModel[T: nn.Module](InferenceModel):
    """Thread-safe wrapper for PyTorch models used in inference.

    This class provides a thread-safe interface for performing inference
    with PyTorch models in multi-threaded environments. It uses a lock
    to ensure that model updates and inference operations don't interfere
    with each other.

    Type Parameters:
        T: The type of the PyTorch model (must be nn.Module subclass).
    """

    def __init__(
        self, model: T, inference_procedure: InferenceProcedureCallable[T]
    ) -> None:
        """Initialize the TorchInferenceModel.

        Args:
            model: A PyTorch model to wrap for thread-safe inference.
            inference_procedure: A callable that defines how to perform
                inference with the model. It should specify the model as the
                first argument, followed by additional arguments.
        """
        self._model = model
        self._inference_procedure = inference_procedure
        self._lock = RLock()

    @property
    def _raw_model(self) -> T:
        """Return the internal PyTorch model.

        Warning:
            This property should not be accessed directly in the inference
            thread. It is intended for internal use to facilitate model
            synchronization between training and inference models.

        Returns:
            The internal PyTorch model instance.
        """
        return self._model

    @_raw_model.setter
    def _raw_model(self, m: T) -> None:
        """Set the internal model in a thread-safe manner.

        Args:
            m: The new PyTorch model to set.
        """
        with self._lock:
            self._model = m

    @torch.inference_mode()
    @override
    def infer(self, *args: Any, **kwds: Any) -> Any:
        """Perform thread-safe inference with gradient computation disabled.

        This method executes the inference procedure with the model while
        ensuring thread safety through locking and disabling gradient
        computation for efficiency.

        Args:
            *args: Positional arguments to pass to the inference procedure.
            **kwds: Keyword arguments to pass to the inference procedure.

        Returns:
            The output from the inference procedure.
        """
        with self._lock:
            return self._inference_procedure(self._model, *args, **kwds)

    def unwrap(self, inference_mode: bool = True) -> UnwrappedContextManager[T]:
        """Get a context manager for direct access to the underlying model.

        This method returns a context manager that provides thread-safe direct
        access to the raw PyTorch model. This is useful when you need to perform
        operations that are not exposed through the standard inference interface.

        Args:
            inference_mode: If True (default), the context will have
                torch.inference_mode enabled, which disables gradient computation
                for better performance.

        Returns:
            A context manager that yields the raw PyTorch model when entered.

        Example:
            >>> inference_model = TorchInferenceModel(my_model, procedure)
            >>> with inference_model.unwrap() as model:
            ...     # Direct access to the model with inference mode enabled
            ...     output = model.some_custom_method(input)
            ...     hidden_state = model.hidden_layer.weight

        Note:
            The context manager ensures thread safety by acquiring a lock
            for the duration of the context. Avoid holding the context for
            extended periods to prevent blocking other threads.
        """
        return UnwrappedContextManager(self._raw_model, self._lock, inference_mode)


class TorchTrainingModel[T: nn.Module](TrainingModel[TorchInferenceModel[T]]):
    """PyTorch model wrapper for parallel training and inference.

    This class enables efficient multi-threaded operation where training
    and inference can run in parallel on separate model instances. It
    manages model synchronization between threads and provides various
    initialization options.

    Type Parameters:
        T: The type of the PyTorch model (must be nn.Module subclass).
    """

    @override
    def __init__(
        self,
        model: T,
        has_inference_model: bool = True,
        inference_thread_only: bool = False,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        inference_procedure: InferenceProcedureCallable[T]
        | str = default_infer_procedure,
        pretrained_parameter_file: str | Path | None = None,
        compile: bool = False,
    ):
        """Initialize the TorchTrainingModel.

        Args:
            model: The PyTorch model to wrap for training.
            has_inference_model: Whether to create a separate inference model.
                If False, inference is not supported.
            inference_thread_only: If True, the same model instance is shared
                between training and inference (no copying). Use this when
                the model is only used for inference.
            device: Device to place the model on. If None, keeps the model
                on its current device.
            dtype: Data type for the model parameters. If specified, converts
                the model to this dtype.
            inference_procedure: The procedure to use for inference. Can be:
                - A callable following the InferenceProcedureCallable protocol
                - A string naming a method on the model class
                - The default_infer_procedure function (default)
            pretrained_parameter_file: Path to a pre-trained model parameter
                file. If provided, loads parameters from this file after
                initialization.
            compile: Whether to compile the model using torch.compile() for
                potentially better performance. If has_inference_model is True,
                both models are compiled.

        Raises:
            AttributeError: If inference_procedure is a string but doesn't exist in model class attributes.
            ValueError: If inference_procedure is a string but doesn't refer
                to a callable method on the model class.
        """
        super().__init__(has_inference_model, inference_thread_only)
        if dtype is not None:
            model = model.type(dtype)
        self.model = model
        if device is None:  # prevents from moving the model to cpu unintentionally.
            device = get_device(model)

        if isinstance(inference_procedure, str):
            method_name = inference_procedure
            if not hasattr(model.__class__, method_name):
                raise AttributeError(
                    f"The model class {model.__class__.__name__} does not have "
                    f"a method named '{method_name}'"
                )
            inference_procedure = getattr(model.__class__, method_name)
            if not callable(inference_procedure):
                raise ValueError(
                    f"The specified inference_procedure '{method_name}' "
                    f"is not a callable method on {model.__class__.__name__}"
                )

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
        """Create an inference model instance.

        Creates either a deep copy of the training model (for parallel
        operation) or returns a wrapper around the same model instance
        (for inference-only mode).

        Returns:
            A TorchInferenceModel wrapping either a copy of the training
            model or the training model itself, depending on the
            inference_thread_only setting.
        """
        model = self.model
        if not self.inference_thread_only:  # the model does not need to be copied to training thread If it is used only in the inference thread.
            model = copy.deepcopy(model)
        return TorchInferenceModel(model, self._inference_procedure)

    @override
    def sync_impl(self, inference_model: TorchInferenceModel[T]) -> None:
        """Synchronize training model parameters to the inference model.

        This method implements an efficient parameter synchronization strategy
        by swapping model references and copying state dictionaries. It preserves
        gradients on the training model during the sync operation.

        Args:
            inference_model: The inference model to synchronize parameters to.

        Note:
            The models are put in eval mode during sync and returned to train
            mode afterwards to ensure proper behavior of layers like BatchNorm
            and Dropout.
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
        """Forward pass through the training model.

        Args:
            *args: Positional arguments to pass to the model.
            **kwds: Keyword arguments to pass to the model.

        Returns:
            The output from the model's forward pass.
        """
        return self.model(*args, **kwds)

    @override
    def save_state(self, path: Path) -> None:
        """Save the model parameters.

        Args:
            path: Base path for saving the model state. The actual file
                will be saved as "{path}.pt".
        """
        torch.save(self.model.state_dict(), f"{path}.pt")  # pyright: ignore[reportUnknownMemberType]

    @override
    def load_state(self, path: Path) -> None:
        """Load model parameters.

        Args:
            path: Base path for loading the model state. The actual file
                loaded will be "{path}.pt".
        """
        self.model.load_state_dict(torch.load(f"{path}.pt"))  # pyright: ignore[reportUnknownMemberType]
