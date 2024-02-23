import torch.nn as nn
from src.transformer_utils import Transformer
import pytorch_lightning as pl
from typing import Dict, Optional, Tuple
import torch
import math
import sys
from src.loss import sigmoid_loss, clip_loss
from src.utils import get_AUC


class Residual(nn.Module):
    """
    A residual block that adds the input to the output of a function.
    """

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        # Apply the function and add the input to the result
        return self.fn(x) + x


class ConvMixer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        channels=1,
        kernel_size=5,
        patch_size=8,
        n_out=128,
        dropout_prob=0.5,
    ):
        super(ConvMixer, self).__init__()

        # Initial convolution layer
        self.net = nn.Sequential(
            nn.Conv2d(
                channels, dim, kernel_size=patch_size, stride=patch_size, bias=False
            ),
            nn.GELU(),
            nn.BatchNorm2d(dim),
        )

        # Adding depth number of ConvMixer layers with dropout
        for _ in range(depth):
            self.net.append(
                nn.Sequential(
                    Residual(
                        nn.Sequential(
                            nn.Conv2d(
                                dim, dim, kernel_size, groups=dim, padding="same"
                            ),
                            nn.GELU(),
                            nn.BatchNorm2d(dim),
                            nn.Dropout(dropout_prob),  # Add dropout here
                        )
                    ),
                    nn.Conv2d(dim, dim, kernel_size=1),
                    nn.GELU(),
                    nn.BatchNorm2d(dim),
                    nn.Dropout(dropout_prob),  # Add dropout here
                )
            )

        # Projection head with dropout
        self.projection = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(dim, 1024),
            nn.GELU(),
            nn.Dropout(dropout_prob),  # Add dropout here
            nn.Linear(1024, n_out),
        )

    def forward(self, x):
        # Forward pass through the network
        x = self.net(x)
        x = self.projection(x)
        return x


class TimePositionalEncoding(nn.Module):
    def __init__(self, d_emb):
        """
        Inputs
            d_model - Hidden dimensionality.
        """
        super().__init__()
        self.d_emb = d_emb

    def forward(self, t):
        pe = torch.zeros(t.shape[0], t.shape[1], self.d_emb).to(t.device)  # (B, T, D)
        div_term = torch.exp(
            torch.arange(0, self.d_emb, 2).float() * (-math.log(10000.0) / self.d_emb)
        )[None, None, :].to(
            t.device
        )  # (1, 1, D / 2)
        t = t.unsqueeze(2)  # (B, 1, T)
        pe[:, :, 0::2] = torch.sin(t * div_term)  # (B, T, D / 2)
        pe[:, :, 1::2] = torch.cos(t * div_term)  # (B, T, D / 2)
        return pe  # (B, T, D)


class TransformerWithTimeEmbeddings(nn.Module):
    """
    Transformer for classifying sequences
    """

    def __init__(self, n_out, nband=1, agg="mean", **kwargs):
        """
        :param n_out: Number of output emedding.
        :param kwargs: Arguments for Transformer.
        """
        super().__init__()

        self.agg = agg
        self.nband = nband
        self.embedding_mag = nn.Linear(in_features=1, out_features=kwargs["emb"])
        self.embedding_t = TimePositionalEncoding(kwargs["emb"])
        self.transformer = Transformer(**kwargs)

        if nband > 1:
            self.band_emb = nn.Embedding(nband, kwargs["emb"])

        self.projection = nn.Linear(kwargs["emb"], n_out)

        # If using attention, initialize a learnable query vector
        if self.agg == "attn":
            self.query = nn.Parameter(torch.rand(kwargs["emb"]))

    def forward(self, x, t, mask=None):
        """
        :param x: A batch by sequence length integer tensor of token indices.
        :return: predicted log-probability vectors for each token based on the preceding tokens.
        """
        t = t - t[:, 0].unsqueeze(1)
        t_emb = self.embedding_t(t)
        x = self.embedding_mag(x) + t_emb

        # learned embeddings for multibands
        if self.nband > 1:
            onehot = (
                torch.linspace(0, self.nband - 1, self.nband)
                .type(torch.LongTensor)
                .repeat_interleave(x.shape[1] // self.nband)
            )
            onehot = onehot.to(t.device)  # (T,)
            b_emb = (
                self.band_emb(onehot).unsqueeze(0).repeat((x.shape[0], 1, 1))
            )  # (T, D) -> (B, T, D)
            x = x + b_emb

        x = self.transformer(x, mask)  # (B, T, D)

        # Zero out the masked values
        x = x * mask[:, :, None]

        if self.agg == "mean":
            x = x.sum(dim=1) / mask.sum(dim=1)[:, None]
        elif self.agg == "max":
            x = x.max(dim=1)[0]
        elif self.agg == "attn":
            q = self.query.unsqueeze(0).repeat(
                x.shape[0], 1, 1
            )  # Duplicate the query across the batch dimension
            k = v = x
            x, _ = nn.MultiheadAttention(
                embed_dim=128, num_heads=2, dropout=0.0, batch_first=True
            )(q, k, v)
            x = x.squeeze(1)  # (B, 1, D) -> (B, D)

        x = self.projection(x)
        return x


class LightCurveImageCLIP(pl.LightningModule):
    def __init__(
        self,
        enc_dim: int = 128,
        logit_scale: float = 10.0,
        nband: int = 1,
        transformer_kwargs: Dict[str, int] = {
            "n_out": 128,
            "emb": 256,
            "heads": 2,
            "depth": 8,
        },
        conv_kwargs: Dict[str, int] = {
            "dim": 32,
            "depth": 8,
            "channels": 3,
            "kernel_size": 5,
            "patch_size": 10,
            "n_out": 128,
        },
        optimizer_kwargs: Dict = {},
        lr: float = 1e-4,
        loss: str = "sigmoid",
    ):
        """
        Initialize the LightCurveImageCLIP module.

        Args:
        enc_dim (int): Dimension of the encoder.
        logit_scale (float): Initial scale for the logits.
        nband (int): Number of bands.
        transformer_kwargs (Dict[str, int]): Keyword arguments for the transformer encoder.
        conv_kwargs (Dict[str, int]): Keyword arguments for the convolutional encoder.
        optimizer_kwargs (Dict): Keyword arguments for the optimizer.
        lr (float): Learning rate.
        loss (str): Loss function type, either "sigmoid" or "softmax".
        """
        super().__init__()
        self.lr = lr
        self.optimizer_kwargs = optimizer_kwargs
        self.enc_dim = enc_dim

        # Parameters
        self.logit_scale = nn.Parameter(
            torch.tensor(math.log(logit_scale)), requires_grad=True
        )
        self.logit_bias = nn.Parameter(torch.tensor(-10.0), requires_grad=True)

        # Encoders
        self.lightcurve_encoder = TransformerWithTimeEmbeddings(
            nband=nband, **transformer_kwargs
        )
        self.image_encoder = ConvMixer(**conv_kwargs)

        # Projection heads
        self.lightcurve_projection = nn.Linear(transformer_kwargs["n_out"], enc_dim)
        self.image_projection = nn.Linear(conv_kwargs["n_out"], enc_dim)

        self.loss = loss

    def forward(
        self,
        x_img: torch.Tensor,
        x_lc: torch.Tensor,
        t_lc: torch.Tensor,
        mask_lc: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.

        Args:
        x_img (torch.Tensor): Input tensor for images.
        x_lc (torch.Tensor): Input tensor for light curves.
        t_lc (torch.Tensor): Time tensor for light curves.
        mask_lc (Optional[torch.Tensor]): Mask tensor for light curves.

        Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of image and light curve embeddings.
        """
        x_lc = self.lightcurve_embeddings_with_projection(x_lc, t_lc, mask_lc)
        x_img = self.image_embeddings_with_projection(x_img)
        return x_img, x_lc

    def image_embeddings_with_projection(self, x_img):
        """Convenience function to get image embeddings with projection"""
        x_img = self.image_encoder(x_img)
        x_img = self.image_projection(x_img)
        return x_img / x_img.norm(dim=-1, keepdim=True)

    def lightcurve_embeddings_with_projection(self, x_lc, t_lc, mask_lc=None):
        """Convenience function to get light curve embeddings with projection"""
        x_lc = x_lc[..., None]  # Add channel dimension
        x_lc = self.lightcurve_encoder(x_lc, t_lc, mask_lc)
        x_lc = self.lightcurve_projection(x_lc)
        return x_lc / x_lc.norm(dim=-1, keepdim=True)

    def configure_optimizers(self):
        optimizer = torch.optim.RAdam(
            self.parameters(), lr=self.lr, **self.optimizer_kwargs
        )
        return {"optimizer": optimizer}

    def training_step(self, batch, batch_idx):
        x_img, x_lc, t_lc, mask_lc = batch
        x_img, x_lc = self(x_img, x_lc, t_lc, mask_lc)
        if self.loss == "sigmoid":
            loss = sigmoid_loss(x_img, x_lc, self.logit_scale, self.logit_bias).mean()
        elif self.loss == "softmax":
            loss = clip_loss(
                x_img,
                x_lc,
                self.logit_scale,
                self.logit_bias,
                self.image_encoder,
                self.lightcurve_encoder,
            ).mean()
        self.log(
            "train_loss", loss, on_epoch=True, on_step=False, prog_bar=True, logger=True
        )
        return loss

    def on_validation_start(self) -> None:
        """
        Called at the beginning of the validation loop.
        """
        # Initialize an empty list to store embeddings for lightcurve and galaxy images
        self.embs_curves = []
        self.embs_images = []

    def validation_step(self, batch, batch_idx):
        x_img, x_lc, t_lc, mask_lc = batch
        x_img, x_lc = self(x_img, x_lc, t_lc, mask_lc)
        self.embs_curves.append(x_lc)
        self.embs_images.append(x_img)
        if self.loss == "sigmoid":
            loss = sigmoid_loss(x_img, x_lc, self.logit_scale, self.logit_bias).mean()
        elif self.loss == "softmax":
            loss = clip_loss(
                x_img,
                x_lc,
                self.logit_scale,
                self.logit_bias,
                self.image_encoder,
                self.lightcurve_encoder,
            ).mean()
        self.log(
            "val_loss", loss, on_epoch=True, on_step=False, prog_bar=True, logger=True
        )
        return loss

    def on_validation_epoch_end(self) -> None:
        """
        Called at the end of the validation epoch.
        """
        # Concatenate all embeddings into single tensors
        self.embs_curves = torch.cat(self.embs_curves, dim=0)
        self.embs_images = torch.cat(self.embs_images, dim=0)

        AUC = get_AUC(self.embs_curves, self.embs_images)
        self.log(
            "AUC_val", AUC, on_epoch=True, on_step=False, prog_bar=True, logger=True
        )

        # Delete the embeddings to free up memory
        self.embs_curves = None
        self.embs_images = None
