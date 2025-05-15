import re
from pathlib import Path
from typing import override

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from pytest_mock import MockerFixture

from pamiq_core.model import InferenceModel, TrainingModel, TrainingModelsDict
from pamiq_core.torch import (
    OptimizersSetup,
    TorchTrainer,
    TorchTrainingModel,
    get_device,
)


class TorchTrainerImpl(TorchTrainer):
    """Concrete implementation of TorchTrainer for testing."""

    @override
    def on_training_models_attached(self) -> None:
        super().on_training_models_attached()
        self.model_1 = self.get_torch_training_model("model_1", nn.Linear)

    @override
    def create_optimizers(self) -> OptimizersSetup:
        """Create optimizer and scheduler for testing."""
        self.optimizer_1 = optim.AdamW(
            self.model_1.model.parameters(),
            lr=0.001,
            betas=(0.8, 0.99),
            weight_decay=0.01,
        )
        self.scheduler_1 = optim.lr_scheduler.ExponentialLR(
            self.optimizer_1, gamma=0.998
        )
        return {"optimizer_1": self.optimizer_1}, {"scheduler_1": self.scheduler_1}

    @override
    def train(self) -> None:
        """Implement basic training step for testing."""
        model = self.model_1.model
        device = get_device(model)
        out = model(torch.randn(5, 2).to(device))
        self.optimizer_1.zero_grad()
        out.mean().backward()
        self.optimizer_1.step()
        self.scheduler_1.step()


class OptimizersOnlyTrainer(TorchTrainerImpl):
    """Trainer implementation that only returns optimizers without
    lr_schedulers."""

    @override
    def create_optimizers(self) -> dict[str, optim.Optimizer]:
        """Return only optimizers without lr_schedulers."""
        self.optimizer_1 = optim.SGD(self.model_1.model.parameters(), lr=0.01)
        return {"optimizer_1": self.optimizer_1}


class TestTorchTrainer:
    @pytest.fixture
    def training_models(self, mocker: MockerFixture) -> TrainingModelsDict:
        """Fixture providing training models dictionary with test models."""
        model_2 = mocker.Mock(TrainingModel)
        model_2.inference_model = mocker.Mock(InferenceModel)
        model_2.has_inference_model = True
        model_2.inference_thread_only = False
        return TrainingModelsDict(
            {
                "model_1": TorchTrainingModel(
                    model=nn.Linear(2, 3),
                    has_inference_model=True,
                    inference_thread_only=False,
                ),
                "model_2": model_2,
            }
        )

    @pytest.fixture
    def torch_trainer(self, training_models: TrainingModelsDict) -> TorchTrainer:
        """Fixture providing initialized TorchTrainer instance."""
        torch_trainer = TorchTrainerImpl()
        torch_trainer.attach_training_models(training_models)
        # Call setup to initialize optimizers and lr_schedulers
        torch_trainer.setup()
        return torch_trainer

    def test_get_torch_training_model(self, torch_trainer: TorchTrainer) -> None:
        """Test get_torch_training_model method returns correct model with type
        checking."""
        # Check if the TorchTrainingModel can be retrieved correctly
        assert isinstance(
            torch_trainer.get_torch_training_model("model_1"), TorchTrainingModel
        )

        # Check type validation with correct model type
        assert isinstance(
            torch_trainer.get_torch_training_model("model_1", nn.Linear).model,
            nn.Linear,
        )

        # Check if error is raised correctly when not TorchTrainingModel
        with pytest.raises(
            TypeError, match="Model model_2 is not a instance of TorchTrainingModel"
        ):
            torch_trainer.get_torch_training_model("model_2")

        # Check if error is raised when internal model is not of expected type
        with pytest.raises(
            TypeError, match=f"Internal model is not a instance of {nn.Conv2d.__name__}"
        ):
            torch_trainer.get_torch_training_model("model_1", nn.Conv2d)

    def test_setup_with_setup_optimizers_and_schedulers(
        self, torch_trainer: TorchTrainerImpl
    ) -> None:
        """Test that _setup_optimizers_and_schedulers correctly initializes
        optimizers and lr_schedulers."""
        # Reset optimizers and lr_schedulers
        torch_trainer.optimizers.clear()
        torch_trainer.lr_schedulers.clear()

        # Call the setup method
        torch_trainer.setup()

        # Verify optimizers and lr_schedulers are created
        assert "optimizer_1" in torch_trainer.optimizers
        assert "scheduler_1" in torch_trainer.lr_schedulers

        # Verify optimizer is correctly configured
        optimizer = torch_trainer.optimizers["optimizer_1"]
        assert isinstance(optimizer, optim.AdamW)
        assert optimizer.param_groups[0]["lr"] == 0.001

        # Verify scheduler is correctly configured
        scheduler = torch_trainer.lr_schedulers["scheduler_1"]
        assert isinstance(scheduler, optim.lr_scheduler.ExponentialLR)
        assert scheduler.gamma == 0.998

    def test_setup_with_only_optimizers(
        self, training_models: TrainingModelsDict
    ) -> None:
        """Test setup with only optimizers returned from create_optimizers."""
        # Create trainer that only returns optimizers without lr_schedulers
        trainer = OptimizersOnlyTrainer()
        trainer.attach_training_models(training_models)
        trainer.setup()

        # Verify optimizers are created, but lr_schedulers dict is empty
        assert "optimizer_1" in trainer.optimizers
        assert len(trainer.lr_schedulers) == 0
        assert isinstance(trainer.optimizers["optimizer_1"], optim.SGD)

    def test_teardown_with_keep_optimizer_and_scheduler_states(
        self, torch_trainer: TorchTrainerImpl
    ) -> None:
        """Test that _keep_optimizer_and_scheduler_states correctly captures
        current states."""
        # Clear any existing states
        torch_trainer.optimizer_states.clear()
        torch_trainer.lr_scheduler_states.clear()

        # Call the method to capture states
        torch_trainer.teardown()

        # Verify optimizer states are kept
        assert "optimizer_1" in torch_trainer.optimizer_states
        assert torch_trainer.optimizer_states["optimizer_1"] is not None

        # Verify scheduler states are kept
        assert "scheduler_1" in torch_trainer.lr_scheduler_states
        assert torch_trainer.lr_scheduler_states["scheduler_1"] is not None

    def test_setup_restores_states(self, training_models: TrainingModelsDict) -> None:
        """Test that setup restores optimizer and scheduler states."""
        # Initialize new trainer
        trainer = TorchTrainerImpl()
        trainer.attach_training_models(training_models)

        # Setup initial optimizers and lr_schedulers
        trainer.setup()

        # Modify learning rate
        original_lr = trainer.optimizers["optimizer_1"].param_groups[0]["lr"]
        modified_lr = original_lr * 2.0

        # Store modified state
        trainer.optimizers["optimizer_1"].param_groups[0]["lr"] = modified_lr
        trainer.teardown()

        # Reset optimizers and setup again - should restore the modified state
        trainer.optimizers.clear()
        trainer.setup()

        # Verify state was restored
        assert trainer.optimizers["optimizer_1"].param_groups[0]["lr"] == modified_lr

    def test_save_and_load_state(
        self,
        torch_trainer: TorchTrainer,
        training_models: TrainingModelsDict,
        tmp_path: Path,
    ) -> None:
        """Test that save_state and load_state correctly persist and restore
        states."""
        # Perform training to modify optimizer state
        torch_trainer.train()

        # Keep original optimizer and scheduler states
        torch_trainer.teardown()

        # Extract key information from original states for comparison
        original_opt_state = torch_trainer.optimizer_states["optimizer_1"]
        original_lr = original_opt_state["param_groups"][0]["lr"]
        original_sched_state = torch_trainer.lr_scheduler_states["scheduler_1"]
        original_last_epoch = original_sched_state["last_epoch"]

        # Save state to disk
        state_path = tmp_path / "trainer_state"
        torch_trainer.save_state(state_path)

        # Verify files are created
        assert (state_path / "optimizer_1.optim.pt").is_file()
        assert (state_path / "scheduler_1.lrsch.pt").is_file()

        # Create a new trainer with the same models
        new_trainer = TorchTrainerImpl()
        new_trainer.attach_training_models(training_models)

        # Load state from disk
        new_trainer.load_state(state_path)

        # Verify states are correctly loaded
        loaded_opt_state = new_trainer.optimizer_states["optimizer_1"]
        loaded_lr = loaded_opt_state["param_groups"][0]["lr"]
        loaded_sched_state = new_trainer.lr_scheduler_states["scheduler_1"]
        loaded_last_epoch = loaded_sched_state["last_epoch"]

        # Compare key values
        assert loaded_lr == original_lr
        assert loaded_last_epoch == original_last_epoch

        # Setup should apply the loaded states
        new_trainer.setup()
        assert (
            new_trainer.optimizers["optimizer_1"].param_groups[0]["lr"] == original_lr
        )

    def test_load_state_with_invalid_path(
        self, torch_trainer: TorchTrainer, tmp_path: Path
    ) -> None:
        """Test that load_state raises ValueError with invalid path."""
        # Non-existent path
        invalid_path = tmp_path / "nonexistent"

        with pytest.raises(
            ValueError,
            match=f"Path {re.escape(str(invalid_path))} is not a directory or does not exist",
        ):
            torch_trainer.load_state(invalid_path)
