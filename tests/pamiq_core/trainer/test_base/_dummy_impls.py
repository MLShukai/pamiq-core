from abc import ABC, abstractmethod
from typing import Any, override

from pamiq_core.data import BufferData, DataBuffer, DataUser, DataUsersDict, StepData
from pamiq_core.model import (
    InferenceModel,
    InferenceModelsDict,
    TrainingModel,
    TrainingModelsDict,
)
from pamiq_core.trainer import Trainer


class DummyInferenceModel(InferenceModel):
    _dummy_param: int = 1234

    @override
    def infer(self, input: list[int]) -> int:
        return sum(input)


class DummyTrainingModel(TrainingModel[DummyInferenceModel]):
    _dummy_param: int = 9999

    @override
    def _create_inference_model(self) -> DummyInferenceModel:
        return DummyInferenceModel()

    @override
    def forward(self, input: list[str]) -> str:
        return "".join(input)

    @override
    def sync_impl(self, inference_model: DummyInferenceModel) -> None:
        inference_model._dummy_param = self._dummy_param


class DummyTrainer(Trainer):
    training_models_attached: bool = False
    data_users_dict_attached: bool = False

    @override
    def on_training_models_attached(self) -> None:
        self.training_models_attached = True

    @override
    def on_data_users_dict_attached(self) -> None:
        self.data_users_dict_attached = True

    @override
    def train(self) -> None:
        dataset = self.data_user.get_dataset()
        data = dataset[0][0]
        self.model1(data)
        self.model2(data)


class DummyDataBuffer(DataBuffer):
    """Simple mock implementation of DataBuffer for testing.

    This implementation uses a list to store data and provides minimal
    functionality needed for testing higher-level components.
    """

    def __init__(self, collecting_data_names: list[str], max_size: int) -> None:
        super().__init__(collecting_data_names, max_size)
        self.data: list[StepData] = []

    @override
    def add(self, step_data: StepData) -> None:
        if len(self.data) < self.max_size:
            self.data.append(step_data)

    @override
    def get_data(self) -> BufferData:
        return {
            name: [d[name] for d in self.data] for name in self._collecting_data_names
        }
