# [Original design principles]
# In the context of reinforcement learning, `agent.py` defines the agent responsible for selecting actions based on observations from the environment.
# The `step()` method processes each observation, determines the corresponding action, and may also handle data collection.
# The `on_inference_models_attached` and `on_data_collectors_attached` methods initialize the agent's access to required models and data collectors, storing them as instance variables for later use.
#
# [Outline of this implementation]
# In this VAE example, `step()` returns the latent representation as the action.
# (The dimension of the action is verified in the environment. This is auxiliary.)
#
# [Why this implementation?]
# In this VAE example, the environment serves only as a data provider and does not maintain a state or respond to the agent's actions.
# Therefore, the agent's action has no effect on the environment.
# To facilitate environment-agent interaction, we use the latent representation as the action, allowing the environment to validate its dimension and structure.


from typing import override

from torch import Tensor

from pamiq_core import Agent


class EncodingAgent(Agent[Tensor, Tensor]):
    """Agent used for this sample."""

    @override
    def on_inference_models_attached(self) -> None:
        """Prepare the Encoder model as the instance variable."""
        super().on_inference_models_attached()
        self.encoder = self.get_inference_model("encoder")

    @override
    def on_data_collectors_attached(self) -> None:
        """Prepare the data collector as the instance variable."""
        self.data_collector = self.get_data_collector("observation")

    @override
    def step(self, observation: Tensor) -> Tensor:
        """Take an action based on the observation.

        In this example,
        (1) return the latent representation as the action (this is auxiliary.)
        (2) store the observation data in the data collector.

        Args:
            observation (Tensor): The observation tensor to be processed.
        Returns:
            Tensor: The inferred latent representation from the encoder.
        """
        # Collect the observation data
        self.data_collector.collect(
            {"data": observation.cpu()}
        )  # explanation of this line:
        # Here, we add the current observation to the stack.
        # The data stack is used in the training process in `trainer.py`.
        return self.encoder(observation)
