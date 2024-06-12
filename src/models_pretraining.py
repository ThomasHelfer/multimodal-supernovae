import torch
import torch.nn as nn
from torch.nn import Module

from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torch import Tensor

from matplotlib import pyplot as plt
from src.transformer_utils import TransformerWithTimeEmbeddings, Transformer

from typing import Dict, Tuple


def get_random_mask(
    padding_mask: Tensor, f_mask: float = 0.15
) -> Tuple[Tensor, Tensor]:
    """
    Generates random contiguous masks for the input sequence.

    Args:
        padding_mask (Tensor): Padding mask of shape (B, T), where B is the batch size and T is the sequence length.
        f_mask (float): Fraction of the sequence to mask out. The value should be between 0 and 1.

    Returns:
        Tuple[Tensor, Tensor]: A tuple of two tensors:
            - mask: The modified mask after applying the mask fraction.
            - mask_pred: A tensor indicating which parts of the input were masked out.
    """

    mask = padding_mask.clone()
    mask_pred = padding_mask.clone()

    # Process each sample in the batch
    for i in range(padding_mask.shape[0]):
        # Count the number of observations (non-padded values) in the current sample
        n_obs = padding_mask[i].sum().item()
        # Calculate how many observations to mask based on the fraction
        n_obs_to_mask = int(n_obs * f_mask)

        # Get indices of non-padded values
        inds_padding = torch.where(padding_mask[i] == True)[0]
        # Randomly permute the indices
        inds_perm = torch.randperm(len(inds_padding))
        # Select indices to keep and predict
        inds_to_keep = inds_padding[inds_perm[n_obs_to_mask:]]
        inds_pred = inds_padding[inds_perm[:n_obs_to_mask]]

        # Update the masks based on selected indices
        mask_pred[i, inds_to_keep] = False  # Mark to keep
        mask[i, inds_pred] = False  # Mark for prediction

    return mask, mask_pred


class MaskedLightCurveEncoder(pl.LightningModule):
    """
    A PyTorch Lightning module for training and evaluating a Transformer-based model for masked light curve encoding.

    Attributes:
        f_mask (float): Fraction of the input to be masked.
        transformer_kwargs (Dict): Keyword arguments for the Transformer model.
        optimizer_kwargs (Dict): Keyword arguments for the optimizer.
        lr (float): Learning rate for the optimizer.
        net (nn.Module): The Transformer model used for light curve encoding.
    """

    def __init__(
        self,
        f_mask: float = 0.2,
        nband: int = 1,
        transformer_kwargs: Dict = {"n_out": 1, "emb": 128, "heads": 2, "depth": 4},
        optimizer_kwargs: Dict = {},
        lr: float = 1e-3,
    ) -> None:
        """
        Initializes the MaskedLightCurveEncoder module.

        Args:
            f_mask (float): Fraction of the input to be masked during training.
            nband (int): Number of bands in the input data.
            transformer_kwargs (Dict): Configuration arguments for the Transformer model.
            optimizer_kwargs (Dict): Configuration arguments for the optimizer.
            lr (float): Learning rate for the optimizer.
        """
        super().__init__()

        self.optimizer_kwargs = optimizer_kwargs
        # nbands are concatenated, so we need to adapt nout
        self.lr = lr
        self.f_mask = f_mask

        self.net = TransformerWithTimeEmbeddings(
            nband=nband, agg="pretraining", **transformer_kwargs
        )
        # addional removable layer
        self.last_layer = nn.Linear(transformer_kwargs["emb"], 1)

    def forward(self, x: Tensor, t: Tensor, mask: Tensor = None) -> Tensor:
        """
        Forward pass through the Transformer model.

        Args:
            x (Tensor): Input data tensor of shape (batch_size, seq_len).
            t (Tensor): Time tensor of shape (batch_size, seq_len).
            mask (Tensor, optional): Mask tensor indicating which parts of the input are valid. Defaults to None.

        Returns:
            Tensor: Output from the Transformer model.
        """
        x = x[..., None]  # Add an additional dimension to x
        x = self.net(x, t, mask)  # Pass through the Transformer model
        x = self.last_layer(x)  # Pass through the last layer
        x = x.squeeze(2)

        return x

    def configure_optimizers(self) -> Dict[str, torch.optim.Optimizer]:
        """
        Configure the optimizer.

        Returns:
            Dict[str, torch.optim.Optimizer]: Dictionary containing the optimizer.
        """
        optimizer = torch.optim.RAdam(
            self.parameters(), lr=self.lr, **self.optimizer_kwargs
        )
        return {"optimizer": optimizer}

    def masked_pred(
        self, x: Tensor, t: Tensor, padding_mask: Tensor, f_mask: float = 0.15
    ) -> Tuple[Tensor, Tensor]:
        """
        Make predictions on the unmasked parts of the input.

        Args:
            x (Tensor): Input data tensor of shape (batch_size, seq_len).
            t (Tensor): Time tensor of shape (batch_size, seq_len).
            padding_mask (Tensor): Mask tensor indicating which parts of the input are valid.
            f_mask (float): Fraction of the input to be masked during training.

        Returns:
            Tuple[Tensor, Tensor]: A tuple containing the original and predicted values of the unmasked parts.
        """
        mask_in, mask_pred = get_random_mask(padding_mask, f_mask=f_mask)
        x_masked = x.clone()
        x_masked[~mask_in] = 0  # Mask out the selected parts of the input
        x_pred = self(x_masked, t, mask=padding_mask)
        return x[mask_pred], x_pred[mask_pred]

    def training_step(self, batch: Tuple[Tensor, ...], batch_idx: int) -> Tensor:
        """
        Perform a training step.

        Args:
            batch (Tuple[Tensor, ...]): A tuple containing input tensors (t, x, padding_mask).
            batch_idx (int): Index of the batch.

        Returns:
            Tensor: Computed loss value.
        """
        if len(batch) == 3:
            t, x, padding_mask = batch
            x, x_pred = self.masked_pred(x, t, padding_mask, f_mask=self.f_mask)
            loss = nn.MSELoss()(x, x_pred)
            self.log("train_loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        else:
            _, x, t, padding_mask, spec, freq, maskspec, redshift, _ = batch
            x, x_pred = self.masked_pred(x, t, padding_mask, f_mask=self.f_mask)
            loss = nn.MSELoss()(x, x_pred)
            self.log("train_loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def validation_step(self, batch: Tuple[Tensor, ...], batch_idx: int) -> Tensor:
        """
        Perform a validation step.

        Args:
            batch (Tuple[Tensor, ...]): A tuple containing input tensors (t, x, padding_mask).
            batch_idx (int): Index of the batch.

        Returns:
            Tensor: Computed loss value.
        """
        # determining what datasource it is coming from
        if len(batch) == 3:
            t, x, padding_mask = batch
            x, x_pred = self.masked_pred(x, t, padding_mask, f_mask=self.f_mask)
            loss = nn.MSELoss()(x, x_pred)
            self.log("val_loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        else:
            _, x, t, padding_mask, spec, freq, maskspec, redshift, _ = batch
            x, x_pred = self.masked_pred(x, t, padding_mask, f_mask=self.f_mask)
            loss = nn.MSELoss()(x, x_pred)
            self.log("val_loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        return loss


def plot_masked_pretraining_model(
    model: Module, loader: DataLoader, path: str = None
) -> None:
    """
    Plot and compare predictions from a model against ground truths from a dataset.

    This function takes a model and a data loader, generates predictions for a batch
    of data, and plots these predictions against the true values. It optionally saves
    the plot to a file.

    Parameters:
        model (Module): The machine learning model to evaluate.
        loader (DataLoader): DataLoader containing the dataset to plot.
        path (str, optional): Path to save the plot image file. If None, the plot is not saved.

    """
    # Set model to evaluation mode and move it to CPU
    model = model.eval()
    model.cpu()

    # Load a batch of data
    time_test, mag_test, mask_test = next(iter(loader))
    fig, ax = plt.subplots(4, 2, figsize=(12, 12))

    for i in range(4):
        for j in range(2):
            ii = i * 2 + j

            # Define range to mask
            fill_min, fill_max = 2, 7

            # Prepare masks and data
            mask_tofill = mask_test[ii : ii + 1].clone()
            mag_truth = mag_test[ii : ii + 1].clone()[mask_test[ii : ii + 1]]
            mag_test_in = mag_test[ii : ii + 1].clone()[mask_test[ii : ii + 1]]
            time_test_in = time_test[ii : ii + 1].clone()[mask_test[ii : ii + 1]]

            # Sort data by time
            inds_sort = torch.argsort(time_test_in)
            mag_truth = mag_truth[inds_sort]
            mag_test_in = mag_test_in[inds_sort]
            time_test_in = time_test_in[inds_sort]

            # Apply mask
            mask_tofill[:, fill_min : fill_max + 1] = False
            mag_test_in[fill_min : fill_max + 1] = 0

            # Make predictions
            mag_pred = (
                model(
                    mag_test_in[None, ...],
                    time_test_in[None, ...],
                    mask_test[ii : ii + 1],
                )[0]
                .detach()
                .numpy()[fill_min : fill_max + 1]
            )

            # Plot predictions and truths
            ax[i, j].scatter(
                time_test_in[fill_min : fill_max + 1], mag_pred, label="pred"
            )
            ax[i, j].scatter(time_test_in, mag_truth, label="truth")

            # Adjust the masked area if necessary
            fill_min = min(fill_min, len(time_test_in) - 1)
            fill_max = min(fill_max, len(time_test_in) - 1)

            # Highlight the masked region
            ax[i, j].axvspan(
                time_test_in[fill_min],
                time_test_in[fill_max],
                alpha=0.1,
                color="red",
                label="masked",
            )
            ax[i, j].legend()

            # Set axis labels and limits
            ax[i, j].set_xlabel("time")
            ax[i, j].set_ylabel("mag")
            ax[i, j].set_ylim([-2, 2])

    # Save the plot if a path is specified
    if path:
        plt.savefig(path)
