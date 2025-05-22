from typing import override

from torch import Tensor, nn
from torch.distributions import Normal

from pamiq_core.torch import get_device


class Encoder(nn.Module):
    """Encoder for VAE.

    3 layers of linear layers with ReLU activation.
    """

    def __init__(self, feature_size: int) -> None:
        """Initialize the encoder.

        Args:
            feature_size: The size of the input feature.
        """
        super().__init__()

        self.latent_dim = feature_size // 8

        self.network = nn.Sequential(
            nn.Linear(feature_size, feature_size // 2),
            nn.ReLU(),
            nn.Linear(feature_size // 2, feature_size // 4),
            nn.ReLU(),
            nn.Linear(
                feature_size // 4, 2 * self.latent_dim
            ),  # the output is means and log variances for a Normal distribution
        )

    @override
    def forward(self, x: Tensor) -> Normal:
        """Forward pass of the encoder.

        Args:
            x: The input tensor.
        Returns:
            Normal: The distribution of the latent space.
        """

        x = self.network(x)

        mean, logstd = x.chunk(2, dim=1)
        scale = (0.5 * logstd).exp()  # 0.5 offers stability

        return Normal(loc=mean, scale=scale)


class Decoder(nn.Module):
    """Decoder for VAE.

    3 layers of linear layers with ReLU activation.
    """

    def __init__(self, feature_size: int) -> None:
        """Initialize the decoder. The `feature_size` is the size of the OUTPUT
        feature. The input feature size is `feature_size // 8`.

        Args:
            feature_size: The size of the OUTPUT feature.
        """
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(feature_size // 8, feature_size // 4),
            nn.ReLU(),
            nn.Linear(feature_size // 4, feature_size // 2),
            nn.ReLU(),
            nn.Linear(feature_size // 2, feature_size),
        )

    @override
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the decoder.

        Args:
            x: The input tensor.
        Returns:
            Tensor: The output tensor.
        """
        x = self.network(x)
        return x


def encoder_infer(model: Encoder, input: Tensor) -> Tensor:
    """Inference of the encoder. Returns the mean of the distribution.

    Args:
        model: The encoder model.
        input: The input tensor.
    Returns:
        Tensor: The mean of the distribution.
    """

    input = input.to(get_device(model))
    dist: Normal = model(input)
    return dist.mean
