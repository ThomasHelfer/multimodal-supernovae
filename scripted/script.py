import os, sys


import numpy as np
import pandas as pd
import math
from typing import Dict, Optional
import matplotlib.pyplot as plt

from tqdm import tqdm
from PIL import Image
from einops import rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import TensorDataset, DataLoader, random_split
from src.models_multimodal import ConvMixer, TransformerWithTimeEmbeddings
import os
from src.utils import get_valid_dir
from src.dataloader import load_images, load_lightcurves, plot_lightcurve_and_images
from src.loss import sigmoid_loss, clip_loss

# ### Data preprocessing

data_dirs = ["ZTFBTS/", "/ocean/projects/phy230064p/shared/ZTFBTS/"]

# Get the first valid directory
data_dir = get_valid_dir(data_dirs)

# Load images from data_dir
host_imgs = load_images(data_dir)

# Load light curves from data_dir
time_ary, mag_ary, magerr_ary, mask_ary, nband = load_lightcurves(data_dir)

# Plot a light curve and its corresponding image
plot_lightcurve_and_images(host_imgs, time_ary, mag_ary, magerr_ary, mask_ary, nband)


time = torch.from_numpy(time_ary).float()
mag = torch.from_numpy(mag_ary).float()
mask = torch.from_numpy(mask_ary).bool()


val_fraction = 0.05
batch_size = 32
n_samples_val = int(val_fraction * mag.shape[0])

dataset = TensorDataset(host_imgs, mag, time, mask)

dataset_train, dataset_val = random_split(
    dataset, [mag.shape[0] - n_samples_val, n_samples_val]
)
train_loader = DataLoader(
    dataset_train, batch_size=batch_size, num_workers=1, pin_memory=True, shuffle=True
)
val_loader = DataLoader(
    dataset_val, batch_size=batch_size, num_workers=1, pin_memory=True, shuffle=False
)


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
            "train_loss",
            loss,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
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
                True,
            ).mean()
        self.log(
            "val_loss",
            loss,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
        )
        return loss


# In[57]:

transformer_kwargs = {"n_out": 32, "emb": 32, "heads": 2, "depth": 1, "dropout": 0.0}
conv_kwargs = {
    "dim": 32,
    "depth": 2,
    "channels": 3,
    "kernel_size": 5,
    "patch_size": 10,
    "n_out": 32,
    "dropout_prob": 0.0,
}


clip_model = LightCurveImageCLIP(
    logit_scale=20.0,
    lr=1e-4,
    nband=nband,
    loss="softmax",
    transformer_kwargs=transformer_kwargs,
    conv_kwargs=conv_kwargs,
)

device = "gpu" if torch.cuda.is_available() else "cpu"


trainer = pl.Trainer(max_epochs=5, accelerator=device)
trainer.fit(
    model=clip_model, train_dataloaders=train_loader, val_dataloaders=val_loader
)
