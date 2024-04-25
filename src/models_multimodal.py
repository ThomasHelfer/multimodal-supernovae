import torch.nn as nn
from src.transformer_utils import Transformer
import pytorch_lightning as pl
from typing import Dict, Optional, Tuple
import torch
import math
import sys
from src.loss import sigmoid_loss_multimodal, clip_loss_multimodal
from src.utils import get_AUC
import wandb


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
    def __init__(self, d_emb, norm=10000.0):
        """
        Inputs
            d_model - Hidden dimensionality.
        """
        super().__init__()
        self.d_emb = d_emb
        self.norm = norm

    def forward(self, t):
        pe = torch.zeros(t.shape[0], t.shape[1], self.d_emb).to(t.device)  # (B, T, D)
        div_term = torch.exp(
            torch.arange(0, self.d_emb, 2).float() * (-math.log(self.norm) / self.d_emb)
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

    def __init__(self, n_out, nband=1, agg="mean", time_norm=10000.0, **kwargs):
        """
        :param n_out: Number of output embedding.
        :param kwargs: Arguments for Transformer.
        """
        super().__init__()

        self.agg = agg
        self.nband = nband
        self.embedding_mag = nn.Linear(in_features=1, out_features=kwargs["emb"])
        self.embedding_t = TimePositionalEncoding(kwargs["emb"], time_norm)
        self.transformer = Transformer(**kwargs)

        if nband > 1:
            self.band_emb = nn.Embedding(nband, kwargs["emb"])

        self.projection = nn.Linear(kwargs["emb"], n_out)

        # If using attention, initialize a learnable query vector
        if self.agg == "attn":
            self.query = nn.Parameter(torch.rand(kwargs["emb"]))
            self.agg_attn = nn.MultiheadAttention(
                embed_dim=kwargs['emb'], num_heads=2, dropout=0.0, batch_first=True
            )

    def forward(self, x, t, mask=None):
        """
        :param x: A batch by sequence length integer tensor of token indices.
        :return: predicted log-probability vectors for each token based on the preceding tokens.
        """
        # Add time embeddings
        t_emb = self.embedding_t(t)
        x = self.embedding_mag(x) + t_emb

        # learned embeddings for multibands
        if self.nband > 1:
            # first half of the array is band 0, second half is band 1, etc.
            # creates one-hot encoding for bands
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
            x, _ = self.agg_attn(q, k, v)
            x = x.squeeze(1)  # (B, 1, D) -> (B, D)

        x = self.projection(x)
        return x


from typing import List


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
            "time_norm": 10000.0,
        },
        transformer_spectral_kwargs: Dict[str, int] = {
            "n_out": 128,
            "emb": 256,
            "heads": 2,
            "depth": 8,
            "time_norm": 10000.0,
        },
        conv_kwargs: Dict[str, int] = {
            "dim": 32,
            "depth": 8,
            "channels": 3,
            "kernel_size": 5,
            "patch_size": 10,
            "n_out": 128,
        },
        combinations: List[str] = ["host_galaxy", "spectral"],
        optimizer_kwargs: Dict = {},
        lr: float = 1e-4,
        loss: str = "sigmoid",
        regression: bool = True, 
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
        combination (array): containing
        lr (float): Learning rate.
        loss (str): Loss function type, either "sigmoid" or "softmax".
        """
        super().__init__()
        self.lr = lr
        self.optimizer_kwargs = optimizer_kwargs
        self.enc_dim = enc_dim
        self.combinations = set(combinations)
        self.regression = regression

        # Parameters
        self.logit_scale = nn.Parameter(
            torch.tensor(math.log(logit_scale)), requires_grad=True
        )
        self.logit_bias = nn.Parameter(torch.tensor(-10.0), requires_grad=True)

        # Encoders
        if "lightcurve" in self.combinations:
            # lightcuve typically has two bands
            self.lightcurve_encoder = TransformerWithTimeEmbeddings(
                nband=nband, **transformer_kwargs
            )
            self.lightcurve_projection = nn.Linear(transformer_kwargs["n_out"], enc_dim)

        if "spectral" in self.combinations:
            # Spectral data does not need the nband variable
            self.spectral_encoder = TransformerWithTimeEmbeddings(
                nband=1, **transformer_spectral_kwargs
            )
            self.spectral_projection = nn.Linear(
                transformer_spectral_kwargs["n_out"], enc_dim
            )

        if "host_galaxy" in self.combinations:
            self.image_encoder = ConvMixer(**conv_kwargs)
            self.image_projection = nn.Linear(conv_kwargs["n_out"], enc_dim)

        self.loss = loss
        self.linear = nn.linear(enc_dim * len(self.combinations), 1)

    def forward(
        self,
        x_img: torch.Tensor,
        x_lc: torch.Tensor,
        t_lc: torch.Tensor,
        mask_lc: Optional[torch.Tensor],
        x_sp: torch.Tensor,
        t_sp: torch.Tensor,
        mask_sp: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.

        Args:
        x_img (torch.Tensor): Input tensor for images.
        x_lc (torch.Tensor): Input tensor for light curves.
        t_lc (torch.Tensor): Time tensor for light curves.
        mask_lc (Optional[torch.Tensor]): Mask tensor for light curves.
        x_sp (torch.Tensor): Input tensor with spectral info
        t_sp (torch.Tensor): frequency tensor with spectral info
        Returns:
        List[torch.Tensor] : Array of embeddings.
        """
        if self.regression: 
            x = []
            if "host_galaxy" in self.combinations:
                x_img = self.image_encoder(x_img)
                x_img = self.image_projection(x_img)
                x.append(x_img)
            if "lightcurve" in self.combinations:
                x_lc = x_lc[..., None]  # Add channel dimension
                x_lc = self.lightcurve_encoder(x_lc, t_lc, mask_lc)
                x_lc = self.lightcurve_projection(x_lc)            if "spectral" in self.combinations:
                x.append(x_lc)
            if "spectral" in self.combinations:
                x_sp = x_sp[..., None]  # Add channel dimension
                x_sp = self.spectral_encoder(x_sp, t_sp, mask_sp)
                x_sp = self.spectral_projection(x_sp)
                x.append(x_sp)
            return x
        else:
            x = []
            if "host_galaxy" in self.combinations:
                x.append(self.image_embeddings_with_projection(x_img))
            if "lightcurve" in self.combinations:
                x.append(self.lightcurve_embeddings_with_projection(x_lc, t_lc, mask_lc))
            if "spectral" in self.combinations:
                x.append(self.spectral_embeddings_with_projection(x_sp, t_sp, mask_sp))
            return x

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

    def spectral_embeddings_with_projection(self, x_lc, t_lc, mask_lc=None):
        """Convenience function to get spectral curve embeddings with projection"""
        x_lc = x_lc[..., None]  # Add channel dimension
        x_lc = self.spectral_encoder(x_lc, t_lc, mask_lc)
        x_lc = self.spectral_projection(x_lc)
        return x_lc / x_lc.norm(dim=-1, keepdim=True)

    def configure_optimizers(self):
        optimizer = torch.optim.RAdam(
            self.parameters(), lr=self.lr, **self.optimizer_kwargs
        )
        return {"optimizer": optimizer}

    def training_step(self, batch, batch_idx):
        x_img, x_lc, t_lc, mask_lc, x_sp, t_sp, mask_sp, redshift = batch
        x = self(x_img, x_lc, t_lc, mask_lc, x_sp, t_sp, mask_sp)

        if self.loss == "sigmoid":
            loss = sigmoid_loss_multimodal(x, self.logit_scale, self.logit_bias).mean()
        elif self.loss == "softmax":
            loss = clip_loss_multimodal(
                x,
                self.logit_scale,
                self.logit_bias,
            ).mean()
        elif self.regression: 
            x = torch.cat(x, dim=-1)
            x = self.linear(x)
            loss = nn.MSELoss()(x, redshift)
        self.log(
            "train_loss", loss, on_epoch=True, on_step=False, prog_bar=True, logger=True
        )
        return loss

    def on_validation_start(self) -> None:
        """
        Called at the beginning of the validation loop.
        """
        # Initialize an empty list to store embeddings
        self.embs_list = [[] for i in range(len(self.combinations))]

    def validation_step(self, batch, batch_idx):
        x_img, x_lc, t_lc, mask_lc, x_sp, t_sp, mask_sp, redshift = batch
        x = self(x_img, x_lc, t_lc, mask_lc, x_sp, t_sp, mask_sp)

        for i in range(len(x)):
            self.embs_list[i].append(x[i])
        if self.loss == "sigmoid":
            loss = sigmoid_loss_multimodal(x, self.logit_scale, self.logit_bias).mean()
        elif self.loss == "softmax":
            loss = clip_loss_multimodal(
                x,
                self.logit_scale,
                self.logit_bias,
            ).mean()
        elif self.regression: 
            x = torch.cat(x, dim=-1)
            x = self.linear(x)
            loss = nn.MSELoss()(x, redshift)
        self.log(
            "val_loss", loss, on_epoch=True, on_step=False, prog_bar=True, logger=True
        )
        return loss

    def on_validation_epoch_end(self) -> None:
        """
        Called at the end of the validation epoch.
        """

        # Concatenate all embeddings into single tensors
        for i in range(len(self.embs_list)):
            self.embs_list[i] = torch.cat(self.embs_list[i], dim=0)

        if len(self.combinations) == 2:
            self.log(
                f"AUC_val", get_AUC(self.embs_list[0], self.embs_list[1]), 
                                        on_epoch=True, 
                                        on_step=False, 
                                        prog_bar=True, 
                                        logger=True
            )
        else:
            count = 1 
            for i in range(len(self.combinations) - 1):
                for j in range(i + 1, len(self.combinations)):
                    self.log(
                        f"AUC_val{count}", get_AUC(self.embs_list[i], 
                                        self.embs_list[j]), 
                                        on_epoch=True, 
                                        on_step=False, 
                                        prog_bar=True, 
                                        logger=True
                    )
                    count += 1 

        # Delete the embeddings to free up memory
        self.embs_list = None 
