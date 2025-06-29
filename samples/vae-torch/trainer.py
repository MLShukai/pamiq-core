# [Original design principles]
# Trainer class is used to train a model by implementing:
# - `on_training_models_attached`: Prepare the training models as instance variables.
# - `on_data_users_attached`: Prepare the data user as instance variable.
# - `create_optimizers`: Create optimizers for the training models.
# - `train`: Train the model by using instance variables and optimizers, prepared before.
#
# [Outline of this implementation]
# In this VAE example, all the functions are implemented in the same way as the original design principles.

import itertools
from pathlib import Path
from typing import override

import torch
import torch.nn.functional as F
from model import Decoder, Encoder
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard.writer import SummaryWriter

from pamiq_core.torch import OptimizersSetup, TorchTrainer, get_device


class VAETrainer(TorchTrainer):
    """Trainer used for this sample."""

    @override
    def __init__(
        self,
        max_epochs: int,
        batch_size: int,
        lr: float = 1e-3,
        log_dir: str | None = None,
    ) -> None:
        """Initialize the VAE trainer.

        Args:
            max_epochs: Maximum number of training epochs.
            batch_size: Size of the training batch.
            lr: Learning rate for the optimizer.
            log_dir: Directory for TensorBoard logs. If None, uses default.
        """

        super().__init__(
            training_condition_data_user="observation",
            min_buffer_size=batch_size,
            min_new_data_count=batch_size,
        )
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.global_step = 0
        self.current_epoch = 0

        # TensorBoard writer
        self.writer = SummaryWriter(log_dir=log_dir) if log_dir else SummaryWriter()

    @override
    def on_training_models_attached(self) -> None:
        """Prepare the encoder and decoder models as instance variables."""
        super().on_training_models_attached()
        self.encoder = self.get_torch_training_model("encoder", Encoder)
        self.decoder = self.get_torch_training_model("decoder", Decoder)
        # Warn not to unwrap by accessing the `.model` attribute (as it will prevent synchronization)

    @override
    def on_data_users_attached(self) -> None:
        """Prepare the data user as an instance variable."""
        super().on_data_users_attached()
        self.data_user = self.get_data_user("observation")

    @override
    def create_optimizers(self) -> OptimizersSetup:
        """Create optimizers for the encoder and decoder models.

        Returns:
            OptimizersSetup: A dictionary containing the optimizer for the VAE.
        """
        params = itertools.chain(
            self.encoder.model.parameters(),
            self.decoder.model.parameters(),
        )
        self.optim = torch.optim.Adam(params, lr=self.lr)
        return {"optimizer": self.optim}

    @override
    def train(self) -> None:
        """Train the VAE model using the provided data.

        This method retrieves the training data, sets up a DataLoader,
        and performs the training loop, including forward passes, loss
        computation, and backpropagation. It logs the training metrics
        to TensorBoard.
        """
        data = self.data_user.get_data()
        dataset = TensorDataset(torch.stack(list(data)))
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        device = get_device(self.encoder.model)

        # Training loop for ordinary VAE
        for epoch in range(self.max_epochs):
            recon_sum = 0.0
            kl_sum = 0.0
            data_count = 0
            for (batch,) in dataloader:
                batch = batch.to(device)

                # Forward pass
                dist = self.encoder(batch)
                z = dist.rsample()
                recon = self.decoder(z)

                # Reconstruction loss (L2 / MSE)
                recon_loss = 0.5 * F.mse_loss(
                    recon, batch, reduction="sum"
                )  # Using half the MSE loss corresponds to
                # assuming the decoder's output distribution has unit variance
                # (i.e., N(recon, I)).

                # KL divergence for Normal
                mu = dist.loc
                logvar = 2 * torch.log(dist.scale)
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - dist.scale.pow(2))

                # Total loss
                loss = recon_loss + kl_loss

                # Backprop
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                # accumulate metrics
                batch_size = batch.size(0)
                recon_sum += recon_loss.item()
                kl_sum += kl_loss.item()
                data_count += batch_size

                # log per batch
                self.global_step += 1
                self.writer.add_scalar(
                    "Loss/Total/Batch", loss.item() / batch_size, self.global_step
                )
                self.writer.add_scalar(
                    "Loss/Reconstruction/Batch",
                    recon_loss.item() / batch_size,
                    self.global_step,
                )
                self.writer.add_scalar(
                    "Loss/KL/Batch", kl_loss.item() / batch_size, self.global_step
                )

            # accumulate metrics per epoch
            avg_total = (recon_sum + kl_sum) / data_count
            avg_recon = recon_sum / data_count
            avg_kl = kl_sum / data_count

            # log per epoch
            self.current_epoch += 1
            self.writer.add_scalar("Loss/Total/Epoch", avg_total, self.current_epoch)
            self.writer.add_scalar(
                "Loss/Reconstruction/Epoch", avg_recon, self.current_epoch
            )
            self.writer.add_scalar("Loss/KL/Epoch", avg_kl, self.current_epoch)

    @override
    def save_state(self, path: Path) -> None:
        """Save the trainer's state to the specified path.

        Args:
            path (Path): The directory where the state will be saved.
        """
        super().save_state(path)
        (path / "global_step").write_text(str(self.global_step))
        (path / "current_epoch").write_text(str(self.current_epoch))

    @override
    def load_state(self, path: Path) -> None:
        """Load the trainer's state from the specified path.

        Args:
            path (Path): The directory from which the state will be loaded.
        """
        super().load_state(path)
        self.global_step = int((path / "global_step").read_text())
        self.current_epoch = int((path / "current_epoch").read_text())
