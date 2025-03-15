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
    def forward(self, input: list[int]) -> int:
        return sum(input)

    @override
    def sync_impl(self, inference_model: DummyInferenceModel) -> None:
        inference_model._dummy_param = self._dummy_param


class DummyTrainer(Trainer):
    @override
    def train(self) -> None:
        dummy_visual_data = self.get_data_user("dummy_visual").get_data()
        batch_image = dummy_visual_data["dummy_image"]
        dummy_visual_data = self.get_data_user("dummy_auditory").get_data()
        batch_audio = dummy_visual_data["dummy_audio"]
        # model_A is assumed as inference thread only. Not training.
        self.get_training_model("model_B")(batch_image)
        self.get_training_model("model_C")(batch_audio)


class DummyDataBuffer(DataBuffer):
    """Simple mock implementation of DataBuffer for testing.

    This implementation uses a list to store data and provides minimal
    functionality needed for testing higher-level components.
    """

    def __init__(self, collecting_data_names: list[str], max_size: int) -> None:
        super().__init__(collecting_data_names, max_size)

    @override
    def add(self, step_data: StepData) -> None:
        pass

    @override
    def get_data(self) -> BufferData:
        dataset_name = next(iter(self._collecting_data_names))
        return {dataset_name: [98765, 43210]}
