import os, sys

sys.path.append("../")

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

from tqdm import tqdm
from PIL import Image
from einops import rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import TensorDataset, DataLoader, random_split
from convmixer_model import ConvMixer
from models.transformer_utils import Transformer
import os
from utils import get_valid_dir
from dataloader import load_images, load_lightcurves, plot_lightcurve_and_images

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


time = torch.from_numpy(time_ary).float()
mag = torch.from_numpy(mag_ary).float()
mask = torch.from_numpy(mask_ary).bool()


# In[18]:


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


# ### Contrastive-style losses


# The standard CLIP architecture uses a bidirection (symmetric between modalities, e.g. image and text) version of the so-called SimCLR loss to compute alignment between image and light curve representations.
# $$\mathcal{L}_\mathrm{CLIP}=-\frac{1}{2|\mathcal{B}|} \sum_{i=1}^{|\mathcal{B}|}\left(\log \frac{e^{t\,x_i \cdot y_i}}{\sum_{j=1}^{|\mathcal{B}|} e^{t\,x_i \cdot y_j}}+\log \frac{e^{t\,x_i \cdot y_i}}{\sum_{j=1}^{|\mathcal{B}|} e^{t\,x_j \cdot y_i}}\right)$$
#
# The standard CLIP loss can be quite unstable due to the small number of positive pairs and large number of negative pairs in a batch. It can also often require very large batch sizes to work well. There are many proposed ways of overcoming this, e.g. see https://lilianweng.github.io/posts/2021-05-31-contrastive/ for some approaches.
#
# In addition to theh softmax-based loss, we'll also try a sigmoid loss, from https://arxiv.org/abs/2303.15343:
# $$\mathcal{L}_\mathrm{SigLIP}=-\frac{1}{|\mathcal{B}|} \sum_{i=1}^{|\mathcal{B}|} \sum_{j=1}^{|\mathcal{B}|} \log \frac{1}{1+e^{z_{i j}\left(-t\, {x}_i \cdot {y}_j+b\right)}}$$
# where $x_i$ and $y_j$ are the normalized image and light curve representations, respectively, and $z_{ij}$ is a binary indicator of whether the image and light curve are a match or not.
#
# Let's implement these two.

# In[55]:


def clip_loss(
    image_embeddings,
    text_embeddings,
    logit_scale=1.0,
    logit_bias=0.0,
    image_encoder=None,
    lightcurve_encoder=None,
    printing=False,
):
    logit_scale = logit_scale.exp()

    logits = (text_embeddings @ image_embeddings.T) * logit_scale + logit_bias

    images_loss = nn.LogSoftmax(dim=1)(logits)
    texts_loss = nn.LogSoftmax(dim=0)(logits)

    images_loss = -images_loss.diag()
    texts_loss = -texts_loss.diag()

    n = min(len(image_embeddings), len(text_embeddings))

    images_loss = images_loss.sum() / n
    texts_loss = texts_loss.sum() / n

    loss = (images_loss + texts_loss) / 2
    return loss


def sigmoid_loss(image_embeds, text_embeds, logit_scale=1.0, logit_bias=2.73):
    """Sigmoid-based CLIP loss, from https://arxiv.org/abs/2303.15343"""

    logit_scale = logit_scale.exp()

    bs = text_embeds.shape[0]

    labels = 2 * torch.eye(bs) - torch.ones((bs, bs))
    labels = labels.to(text_embeds.device)

    logits = -text_embeds @ image_embeds.t() * logit_scale + logit_bias
    logits = logits.to(torch.float64)

    loss = -torch.mean(torch.log(torch.sigmoid(-labels * logits)))

    return loss


# In[56]:


class LightCurveImageCLIP(pl.LightningModule):
    def __init__(
        self,
        enc_dim=128,
        logit_scale=10.0,
        nband=1,
        transformer_kwargs={"n_out": 128, "emb": 256, "heads": 2, "depth": 8},
        conv_kwargs={
            "dim": 32,
            "depth": 8,
            "channels": 3,
            "kernel_size": 5,
            "patch_size": 10,
            "n_out": 128,
        },
        optimizer_kwargs={},
        lr=1e-4,
        loss="sigmoid",
    ):
        super().__init__()

        self.lr = lr
        self.optimizer_kwargs = optimizer_kwargs
        self.enc_dim = enc_dim

        # Make temperature and logit bias a learnable parameter
        # Init values log(10) and -10 from https://arxiv.org/abs/2303.15343
        self.logit_scale = nn.Parameter(
            torch.tensor(math.log(logit_scale)), requires_grad=True
        )
        self.logit_bias = nn.Parameter(torch.tensor(-10.0), requires_grad=True)

        # Encoders
        self.lightcurve_encoder = TransformerWithTimeEmbeddings(
            nband=nband, **transformer_kwargs
        )
        self.image_encoder = ConvMixer(**conv_kwargs)

        # Projection heads to common embedding space
        self.lightcurve_projection = nn.Linear(transformer_kwargs["n_out"], enc_dim)
        self.image_projection = nn.Linear(conv_kwargs["n_out"], enc_dim)

        self.loss = loss

    def forward(self, x_img, x_lc, t_lc, mask_lc=None):
        # Light curve encoder
        x_lc = self.lightcurve_embeddings_with_projection(x_lc, t_lc, mask_lc)

        # Image encoder
        x_img = self.image_embeddings_with_projection(x_img)

        # Normalized embeddings
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

x_img, x_lc, t_lc, mask_lc = next(iter(train_loader))
x_img, x_lc = clip_model(x_img, x_lc, t_lc, mask_lc)

sigmoid_loss(x_img, x_lc, clip_model.logit_scale).mean(), clip_loss(
    x_img, x_lc, clip_model.logit_scale
).mean()


# In[58]:


trainer = pl.Trainer(max_epochs=5, accelerator="gpu")
trainer.fit(
    model=clip_model, train_dataloaders=train_loader, val_dataloaders=val_loader
)


print("finished training")
