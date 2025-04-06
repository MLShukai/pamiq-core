"""PyTorch trainer implementation for pamiq-core.

This module provides a base class for implementing PyTorch model
training within the pamiq-core framework. It handles optimizer and
learning rate scheduler configuration, state management, and integrates
with the pamiq-core training system.
"""

from abc import abstractmethod
from pathlib import Path
from typing import Any, override

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from pamiq_core.trainer import Trainer

from .model import TorchModelWrapper

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

    def __init__(self) -> None:
        """Initialize the PyTorch trainer.

        Sets up empty containers for optimizers, schedulers, and their
        respective states. Actual optimizer and scheduler instances will
        be created during the setup phase.
        """
        super().__init__()

        # Containers for optimizer and scheduler instances
        self._optimizers: OptimizersDict = {}
        self._lr_schedulers: LRSchedulersDict = {}

        # Containers for persistent optimizer and scheduler states
        self._optimizer_states: dict[str, StateDict] = {}
        self._scheduler_states: dict[str, StateDict] = {}

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
        ...

    @override
    def get_training_model[T: nn.Module](
        self, name: str, module_cls: type[T] = nn.Module
    ) -> TorchModelWrapper[T]:
        """Get a PyTorch training model with type checking.

        Retrieves a PyTorch model wrapper by name and validates that it contains
        a model of the expected type.

        Args:
            name: Name of the model to retrieve
            module_cls: Expected module class type

        Returns:
            TorchModelWrapper containing the requested model

        Raises:
            ValueError: If the model is not a TorchModelWrapper or doesn't match the expected type
        """
        wrapper = super().get_training_model(name)
        if not isinstance(wrapper, TorchModelWrapper):
            raise ValueError(f"Model {name} is not a TorchModelWrapper")

        model = wrapper.model
        if not isinstance(model, module_cls):
            raise ValueError(
                f"Model {name} is not an instance of {module_cls.__name__}"
            )

        return wrapper

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

        Saves the current state of optimizers and schedulers before
        cleanup.
        """
        super().teardown()
        self._save_optimizer_and_scheduler_states()

    def _setup_optimizers_and_schedulers(self) -> None:
        """Setup optimizers and schedulers from configuration.

        Creates optimizer and scheduler instances based on the configuration
        provided by `configure_optimizers` and restores their previous states
        if available.
        """
        optimizer_config = self.create_optimizers()

        # Reset existing optimizer and scheduler collections
        self._optimizers.clear()
        self._lr_schedulers.clear()

        # Process configuration based on return type
        if isinstance(optimizer_config, tuple) and len(optimizer_config) == 2:
            # Configuration includes both optimizers and schedulers
            optimizers, schedulers = optimizer_config
            self._optimizers.update(optimizers)
            self._lr_schedulers.update(schedulers)
        else:
            # Configuration includes only optimizers
            self._optimizers.update(optimizer_config)

        # Restore optimizer states if available
        for name, state in self._optimizer_states.items():
            self._optimizers[name].load_state_dict(state)

        # Restore scheduler states if available
        for name, state in self._scheduler_states.items():
            self._lr_schedulers[name].load_state_dict(state)

    def _save_optimizer_and_scheduler_states(self) -> None:
        """Save the current states of optimizers and schedulers.

        Captures and stores the state dictionaries of all active
        optimizers and schedulers for future restoration.
        """
        # Clear previous saved states
        self._optimizer_states.clear()
        self._scheduler_states.clear()

        # Save current optimizer states
        for name, optimizer in self._optimizers.items():
            self._optimizer_states[name] = optimizer.state_dict().copy()

        # Save current scheduler states
        for name, scheduler in self._lr_schedulers.items():
            self._scheduler_states[name] = scheduler.state_dict().copy()

    @override
    def save_state(self, path: Path) -> None:
        """Save trainer state to disk.

        Persists the states of all optimizers and schedulers to the specified
        directory path.

        Args:
            path: Directory path where state should be saved
        """
        super().save_state(path)
        path.mkdir(exist_ok=True)

        # Save optimizer states to disk
        for name, optimizer_state in self._optimizer_states.items():
            torch.save(optimizer_state, path / f"{name}.optim.pt")

        # Save scheduler states to disk
        for name, scheduler_state in self._scheduler_states.items():
            torch.save(scheduler_state, path / f"{name}.lrsch.pt")

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
            self._optimizer_states[name] = torch.load(optimizer_path)

        # Load scheduler states from disk
        for scheduler_path in path.glob("*.lrsch.pt"):
            name = scheduler_path.name.replace(".lrsch.pt", "")
            self._scheduler_states[name] = torch.load(scheduler_path)
