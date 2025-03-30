from typing import override

from pamiq_core.model import InferenceModel, TrainingModel


class InferenceModelImpl(InferenceModel):
    param: int = 1234

    @override
    def infer(self, input: list[int]) -> int:
        return sum(input)


class TrainingModelImpl(TrainingModel[InferenceModelImpl]):
    param: int = 9999

    @override
    def _create_inference_model(self) -> InferenceModelImpl:
        return InferenceModelImpl()

    @override
    def forward(self, input: list[str]) -> str:
        return "".join(input)

    @override
    def sync_impl(self, inference_model: InferenceModelImpl) -> None:
        inference_model.param = self.param
