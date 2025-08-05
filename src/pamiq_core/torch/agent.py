from typing import cast

import torch.nn as nn

from pamiq_core.interaction import Agent

from .model import TorchInferenceModel


class TorchAgent[O, A](Agent[O, A]):
    """Agent class specialized for PyTorch models.

    This class extends the base Agent class to provide type-safe access to
    PyTorch inference models.

    Type Parameters:
        O: The observation type.
        A: The action type.
    """

    def get_torch_inference_model[T: nn.Module](
        self, name: str, module_cls: type[T] = nn.Module
    ) -> TorchInferenceModel[T]:
        """Retrieve a TorchInferenceModel with type checking.

        This method retrieves an inference model by name and verifies that it is
        a TorchInferenceModel instance containing the expected PyTorch module type.

        Args:
            name: The name of the inference model to retrieve.
            module_cls: The expected PyTorch module class.

        Returns:
            A TorchInferenceModel instance containing a model of the specified type.

        Raises:
            ValueError: If the retrieved model is not a TorchInferenceModel instance.
            TypeError: If the internal model is not an instance of module_cls.
            KeyError: If no model with the specified name exists.
        """
        inference_model = self.get_inference_model(name)
        if not isinstance(inference_model, TorchInferenceModel):
            raise ValueError(f"Model {name} is not an instance of TorchInferenceModel")

        inference_model = cast(TorchInferenceModel[T], inference_model)

        if not isinstance(inference_model._raw_model, module_cls):  # pyright: ignore[reportPrivateUsage, ]
            raise TypeError(
                f"Internal model is not an instance of {module_cls.__name__}"
            )
        return inference_model
