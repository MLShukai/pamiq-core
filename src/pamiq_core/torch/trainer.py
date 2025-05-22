"""PyTorch trainer implementation for pamiq-core.

This module provides a base class for implementing PyTorch model
training within the pamiq-core framework. It handles optimizer and
learning rate scheduler configuration, state management, and integrates
with the pamiq-core training system.
"""

from abc import abstractmethod
from pathlib import Path
from typing import Any, cast, override

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from pamiq_core.torch import TorchTrainingModel
from pamiq_core.trainer import Trainer

# Type definitions for improved type safety and readability
type StateDict = dict[str, Any]
type OptimizersDict = dict[str, Optimizer]
type LRSchedulersDict = dict[str, LRScheduler]
type OptimizersSetup = OptimizersDict | tuple[OptimizersDict, LRSchedulersDict]


class TorchTrainer(Trainer):
    """Base class for PyTorch model training in pamiq-core.

    This trainer integrates PyTorch models with the pamiq-core framework,
    providing functionality for optimizer configuration, state persistence,
    and model type validation. It automatically handles the setup and teardown
    of optimizers and learning rate schedulers during the training process.

    Subclasses should implement the `configure_optimizers` and `train` methods
    to define the specific training behavior.
    """

    def __init__(
        self,
        training_condition_data_user: str | None = None,
        min_buffer_size: int = 0,
        min_new_data_count: int = 0,
    ) -> None:
        """Initialize the PyTorch trainer.

        Sets up empty containers for optimizers, schedulers, and their
        respective states. Actual optimizer and scheduler instances will
        be created during the setup phase.

        Args:
            training_condition_data_user: Name of the data user to check for trainability.
                If None, trainer is always trainable.
            min_buffer_size: Minimum total data points required in the buffer.
            min_new_data_count: Minimum number of new data points required since last training.
        """
        super().__init__(
            training_condition_data_user,
            min_buffer_size,
            min_new_data_count,
        )

        # Containers for optimizer and scheduler instances
        self.optimizers: OptimizersDict = {}
        self.lr_schedulers: LRSchedulersDict = {}

        # Containers for persistent optimizer and scheduler states
        self.optimizer_states: dict[str, StateDict] = {}
        self.lr_scheduler_states: dict[str, StateDict] = {}

    def get_torch_training_model[T: nn.Module](
        self, name: str, module_cls: type[T] = nn.Module
    ) -> TorchTrainingModel[T]:
        """Get a TorchTrainingModel with type checking. Retrieves a PyTorch
        model training model by name and validates internal model type.

        Args:
            name: Name of the model to retrieve.
            module_cls: Expected internal module class.

        Returns:
            An instance of TorchTrainingModel with specified model type.
        Raises:
            TypeError: If the model is not a TorchTrainingModel or internal model is not specified module class.
        """
        training_model = self.get_training_model(name)
        if not isinstance(training_model, TorchTrainingModel):
            raise TypeError(f"Model {name} is not a instance of TorchTrainingModel")

        training_model = cast(TorchTrainingModel[T], training_model)

        if not isinstance(training_model.model, module_cls):
            raise TypeError(
                f"Internal model is not a instance of {module_cls.__name__}"
            )
        return training_model

    @abstractmethod
    def create_optimizers(self) -> OptimizersSetup:
        """Create and return optimizers and optional schedulers for training.
        Implementations should create optimizers for each model being trained,
        and optionally create learning rate schedulers.
        Returns:
            Either:
            - Dictionary mapping names to optimizers
            - Tuple containing (optimizers dictionary, schedulers dictionary)
        """
        pass

    def _setup_optimizers_and_schedulers(self) -> None:
        """Setup optimizers and schedulers from configuration.

        Creates optimizer and scheduler instances based on the configuration
        provided by `create_optimizers`.
        """
        optimizer_config = self.create_optimizers()

        # Reset existing optimizer and scheduler collections
        self.optimizers.clear()
        self.lr_schedulers.clear()

        # Process configuration based on return type
        if isinstance(optimizer_config, tuple):
            # Configuration includes both optimizers and schedulers
            optimizers, schedulers = optimizer_config
            self.optimizers.update(optimizers)
            self.lr_schedulers.update(schedulers)
        else:
            # Configuration includes only optimizers
            self.optimizers.update(optimizer_config)

        # Restore optimizer states if available
        for name, state in self.optimizer_states.items():
            self.optimizers[name].load_state_dict(state)
        # Restore scheduler states if available
        for name, state in self.lr_scheduler_states.items():
            self.lr_schedulers[name].load_state_dict(state)

    def _keep_optimizer_and_scheduler_states(self) -> None:
        """Keep the current states of optimizers and schedulers.

        Captures and stores the state dictionaries of all active
        optimizers and schedulers for future restoration.
        """
        # Clear previous kept states
        self.optimizer_states.clear()
        self.lr_scheduler_states.clear()

        # Keep current optimizer states
        for name, optimizer in self.optimizers.items():
            self.optimizer_states[name] = optimizer.state_dict().copy()

        # Keep current scheduler states
        for name, scheduler in self.lr_schedulers.items():
            self.lr_scheduler_states[name] = scheduler.state_dict().copy()

    @override
    def setup(self) -> None:
        """Set up the training environment.

        Initializes optimizers and schedulers by calling the `configure_optimizers`
        method and restores their states if previously saved.
        """
        super().setup()
        self._setup_optimizers_and_schedulers()

    @override
    def teardown(self) -> None:
        """Clean up after training.

        Keeps the current state of optimizers and schedulers before
        cleanup.
        """
        super().teardown()
        self._keep_optimizer_and_scheduler_states()

    @override
    def save_state(self, path: Path) -> None:
        """Save trainer state to disk.

        Persists the states of all optimizers and schedulers to the specified
        directory path.

        Args:
            path: Directory path where state should be saved
        """
        super().save_state(path)

        # Before this method performed, the state of the optimizer or LR scheduler have already been kept by the `teardown` method.

        # Save optimizer states to disk
        for name, optimizer_state in self.optimizer_states.items():
            torch.save(optimizer_state, path / f"{name}.optim.pt")  # pyright: ignore[reportUnknownMemberType]

        # Save scheduler states to disk
        for name, scheduler_state in self.lr_scheduler_states.items():
            torch.save(scheduler_state, path / f"{name}.lrsch.pt")  # pyright: ignore[reportUnknownMemberType]

    @override
    def load_state(self, path: Path) -> None:
        """Load trainer state from disk.

        Loads the previously saved states of optimizers and schedulers from
        the specified directory path.

        Args:
            path: Directory path from where state should be loaded

        Raises:
            ValueError: If the path does not exist or is not a directory
        """
        if not path.is_dir():
            raise ValueError(f"Path {path} is not a directory or does not exist")

        super().load_state(path)

        # Load optimizer states from disk
        for optimizer_path in path.glob("*.optim.pt"):
            name = optimizer_path.name.replace(".optim.pt", "")
            self.optimizer_states[name] = torch.load(optimizer_path)  # pyright: ignore[reportUnknownMemberType]

        # Load scheduler states from disk
        for scheduler_path in path.glob("*.lrsch.pt"):
            name = scheduler_path.name.replace(".lrsch.pt", "")
            self.lr_scheduler_states[name] = torch.load(scheduler_path)  # pyright: ignore[reportUnknownMemberType]

        # After This method performed, load the state into the optimizer and LR scheduler by the `setup` method.
