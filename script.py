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
from torchvision.transforms import RandomRotation
from torch.utils.data import TensorDataset, DataLoader, random_split

from src.models_multimodal import ConvMixer, TransformerWithTimeEmbeddings
from src.utils import get_valid_dir, LossTrackingCallback, plot_loss_history, cosine_similarity,get_embs,plot_ROC_curves
from src.dataloader import load_images, load_lightcurves, plot_lightcurve_and_images
from src.loss import sigmoid_loss, clip_loss
from typing import Tuple


# Data preprocessing

data_dirs = ["/home/thelfer1/scr4_tedwar42/thelfer1/ZTFBTS/","ZTFBTS/","/ocean/projects/phy230064p/shared/ZTFBTS/"]

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
magerr = torch.from_numpy(magerr_ary).float()

val_fraction = 0.05
batch_size = 32
n_samples_val = int(val_fraction * mag.shape[0])

dataset = TensorDataset(host_imgs, mag, time, mask, magerr)  

dataset_train, dataset_val = random_split(
    dataset, [mag.shape[0] - n_samples_val, n_samples_val]
)
train_loader_no_aug = DataLoader(
    dataset_train, batch_size=batch_size, num_workers=1, pin_memory=True, shuffle=True
)
val_loader_no_aug = DataLoader(
    dataset_val, batch_size=batch_size, num_workers=1, pin_memory=True, shuffle=False
)

# Custom data loader with noise augmentation using magerr
class NoisyDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, noise_level_img, noise_level_mag, shuffle=True, **kwargs):
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)
        self.max_noise_intensity = noise_level_img
        self.noise_level_mag = noise_level_mag

    def __iter__(self):
        for batch in super().__iter__():
            # Add random noise to images and time-magnitude tensors
            host_imgs, mag, time, mask, magerr = batch

            # Calculate the range for the random noise based on the max_noise_intensity
            noise_range = self.max_noise_intensity * torch.std(host_imgs)

            # Generate random noise within the specified range
            noisy_imgs = host_imgs + (2 * torch.rand_like(host_imgs) - 1) * noise_range

            # Add Gaussian noise to mag using magerr
            noisy_mag = mag + torch.randn_like(mag) * magerr * self.noise_level_mag

            # Randomly apply rotation by multiples of 90 degrees
            rotation_angle = torch.randint(0, 4, (noisy_imgs.size(0),)) * 90
            rotated_imgs = []

            # Apply rotation to each image
            for i in range(noisy_imgs.size(0)):
                rotated_img = RandomRotation([rotation_angle[i], rotation_angle[i]])(noisy_imgs[i])
                rotated_imgs.append(rotated_img)

            # Stack the rotated images back into a tensor
            rotated_imgs = torch.stack(rotated_imgs)
            
            # Return the noisy batch
            yield noisy_imgs, noisy_mag, time, mask

# Define the noise levels for images and magnitude (multiplied by magerr)
noise_level_img = 1  # Adjust as needed
noise_level_mag = 1  # Adjust as needed

val_noise = 0

# Create custom noisy data loaders
train_loader = NoisyDataLoader(dataset_train, batch_size=batch_size, noise_level_img=noise_level_img, noise_level_mag=noise_level_mag, shuffle=True, num_workers=1, pin_memory=True)
val_loader = NoisyDataLoader(dataset_val, batch_size=batch_size, noise_level_img=val_noise, noise_level_mag=val_noise, shuffle=False, num_workers=1, pin_memory=True)



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
            logger=True
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
            ).mean()
        self.log(
            "val_loss",
            loss,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            logger=True
        )
        return loss


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



# Custom call back for tracking loss
loss_tracking_callback = LossTrackingCallback()

device = "gpu" if torch.cuda.is_available() else "cpu"

trainer = pl.Trainer(max_epochs=100, accelerator=device, callbacks=[loss_tracking_callback]
)
trainer.fit(
    model=clip_model, train_dataloaders=train_loader, val_dataloaders=val_loader
)

plot_loss_history(loss_tracking_callback.train_loss_history, loss_tracking_callback.val_loss_history)

# Get embeddings for all images and light curves
embs_curves_train,embs_images_train = get_embs(clip_model,train_loader_no_aug)
embs_curves_val,embs_images_val = get_embs(clip_model,val_loader_no_aug)

plot_ROC_curves(embs_curves_train,embs_images_train,embs_curves_val,embs_images_val)