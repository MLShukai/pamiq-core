import copy
import logging
from pathlib import Path
from typing import Any, override

import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from pamiq_core.model import InferenceModel, TrainingModel
from pamiq_core.torch import (
    LRSchedulersDict,
    OptimizersDict,
    OptimizersSetup,
    StateDict,
    TorchInferenceModel,
    TorchTrainer,
    TorchTrainingModel,
)

from .are_dict_values_same_entities import are_dict_values_same_entities

logger = logging.getLogger(__name__)

CPU_DEVICE = torch.device("cpu")
CUDA_DEVICE = torch.device("cuda:0")


def get_available_devices() -> list[torch.device]:
    devices = [CPU_DEVICE]
    if torch.cuda.is_available():
        devices.append(CUDA_DEVICE)
    return devices


logger.info("Available devices: " + ", ".join(map(str, get_available_devices())))

parametrize_device = pytest.mark.parametrize("device", get_available_devices())


class TorchTrainerImpl(TorchTrainer):
    @override
    def create_optimizers(self) -> OptimizersSetup:
        optimizer_1 = optim.AdamW(
            self.get_training_model("model_1").model.parameters(),
            lr=0.001,
            betas=(0.8, 0.99),
            weight_decay=0.01,
        )
        scheduler_1 = optim.lr_scheduler.ExponentialLR(optimizer_1, gamma=0.998)
        return {"optimizer_1": optimizer_1}, {"scheduler_1": scheduler_1}

    @override
    def train(self) -> None:
        model = self.get_training_model("model_1")
        device = next(model.model.parameters()).device
        out = model(torch.randn(5, 2).to(device))
        self._optimizers["optimizer_1"].zero_grad()
        out.mean().backward()
        self._optimizers["optimizer_1"].step()
        self._schedulers["scheduler_1"].step()


class TestTorchTrainer:
    @pytest.fixture
    def training_models(self, device: torch.device) -> TorchTrainingModel:
        return {
            "model_1": TorchTrainingModel(
                model=nn.Linear(2, 3),
                has_inference_model=True,
                inference_thread_only=False,
                device=device,
            ),
        }

    @pytest.fixture
    def torch_trainer(self, training_models: TorchTrainingModel) -> TorchTrainer:
        torch_trainer = TorchTrainerImpl()
        torch_trainer.attach_training_models(training_models)
        return torch_trainer

    @parametrize_device
    def test_setup(self, torch_trainer: TorchTrainer) -> None:
        torch_trainer.setup()
        # Check if the keys are created correctly.
        assert list(torch_trainer._optimizers.keys()) == ["optimizer_1"]
        assert list(torch_trainer._schedulers.keys()) == ["scheduler_1"]
        # Check if the instances are created correctly.
        assert isinstance(torch_trainer._optimizers["optimizer_1"], optim.AdamW)
        assert isinstance(
            torch_trainer._schedulers["scheduler_1"], optim.lr_scheduler.ExponentialLR
        )

    @parametrize_device
    def test_teardown(self, torch_trainer: TorchTrainer) -> None:
        torch_trainer.setup()
        torch_trainer.teardown()
        # Check if the keys are kept correctly.
        assert list(torch_trainer._optimizer_states.keys()) == ["optimizer_1"]
        assert list(torch_trainer._scheduler_states.keys()) == ["scheduler_1"]
        # Check if the optimizers parameters are kept correctly.
        kept_optimizer_state = torch_trainer._optimizer_states["optimizer_1"]
        optimizer_state = torch_trainer._optimizers["optimizer_1"].state_dict()
        assert are_dict_values_same_entities(kept_optimizer_state, optimizer_state)
        # Check if the schedulers parameters are kept correctly.
        kept_scheduler_state = torch_trainer._scheduler_states["scheduler_1"]
        scheduler_state = torch_trainer._schedulers["scheduler_1"].state_dict()
        assert are_dict_values_same_entities(kept_scheduler_state, scheduler_state)

    @parametrize_device
    def test_save_and_load_state(
        self,
        torch_trainer: TorchTrainer,
        tmp_path: Path,
    ):
        torch_trainer.setup()
        torch_trainer.teardown()
        # define path
        torch_trainer.save_state(tmp_path)
        # check if file saved
        assert (tmp_path / "optimizer_1.optim.pt").is_file()
        assert (tmp_path / "scheduler_1.lrsch.pt").is_file()
        # keep states for comparison below.
        saved_optim_params = copy.deepcopy(torch_trainer._optimizer_states)
        saved_lrsch_params = copy.deepcopy(torch_trainer._scheduler_states)
        # make differences
        torch_trainer.setup()
        torch_trainer.train()
        torch_trainer.teardown()
        # check if the differences between the models are made correctly.
        assert not torch_trainer._optimizer_states == saved_optim_params
        assert not torch_trainer._scheduler_states == saved_lrsch_params
        # check if load can be performed correctly.
        torch_trainer.load_state(tmp_path)
        loaded_optim_params = torch_trainer._optimizer_states
        loaded_lrsch_params = torch_trainer._scheduler_states
        assert loaded_optim_params == saved_optim_params
        assert loaded_lrsch_params == saved_lrsch_params
