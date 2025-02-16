from collections import deque
from collections.abc import Iterable
from typing import Any, override

import pytest

from pamiq_core.model import TrainingModelsDict


class TestTrainingModel:
    @pytest.fixture
    def training_models_dict(self) -> TrainingModelsDict:
        return TrainingModelsDict()

    def test_inference_models_dict(
        self, training_models_dict: TrainingModelsDict
    ) -> None:
        pass
