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

        self.network = nn.Sequential(
            nn.Linear(feature_size, feature_size // 2),
            nn.ReLU(),
            nn.Linear(feature_size // 2, feature_size // 4),
            nn.ReLU(),
        )

        self.fc_mean = nn.Linear(feature_size // 4, feature_size // 8)
        self.fc_logvar = nn.Linear(feature_size // 4, feature_size // 8)

    @override
    def forward(self, x: Tensor) -> Normal:
        """Forward pass of the encoder.

        Args:
            x: The input tensor.
        Returns:
            Normal: The distribution of the latent space.
        """

        x = self.network(x)
        mean = self.fc_mean(x)
        logvar = self.fc_logvar(x)
        scale = (0.5 * logvar).exp()  # 0.5 offers stability

        return Normal(loc=mean, scale=scale)

    def infer(self, x: Tensor) -> Tensor:
        """Inference of the encoder.

        Returns the mean of the distribution.

         Args:
             x: The input tensor.
         Returns:
             Tensor: The mean of the distribution.
        """

        # We don't need `self.eval()` nor `torch.inference_mode()` here,
        # because torch extension of pamiq-core do this.
        x = x.to(get_device(self))
        dist: Normal = self(x)
        return dist.mean


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
