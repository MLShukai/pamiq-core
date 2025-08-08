try:
    import torch  # pyright: ignore[reportUnusedImport]
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "torch module not found. To use torch features of pamiq-core, "
        "please run the following command:\n\n"
        "    pip install pamiq-core[torch]\n"
    )


from .agent import TorchAgent
from .model import (
    TorchInferenceModel,
    TorchTrainingModel,
    default_infer_procedure,
    get_device,
)
from .trainer import (
    LRSchedulersDict,
    OptimizersDict,
    OptimizersSetup,
    StateDict,
    TorchTrainer,
)

__all__ = [
    "TorchInferenceModel",
    "TorchTrainingModel",
    "default_infer_procedure",
    "get_device",
    "LRSchedulersDict",
    "OptimizersDict",
    "OptimizersSetup",
    "StateDict",
    "TorchTrainer",
    "TorchAgent",
]
