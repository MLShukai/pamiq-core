try:
    import torch  # pyright: ignore[reportUnusedImport]
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "torch module not found. To use torch features of pamiq-core, "
        "please run the following command:\n\n"
        "    pip install pamiq-core[torch]\n"
    )


from .model import (
    TorchInferenceModel,
    TorchTrainingModel,
    default_infer_procedure,
    get_device,
)

__all__ = [
    "TorchInferenceModel",
    "TorchTrainingModel",
    "default_infer_procedure",
    "get_device",
]
