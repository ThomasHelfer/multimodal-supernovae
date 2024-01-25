#!/usr/bin/env python
# coding: utf-8

# # Self-supervised and multi-modal representation Learning: Notebook 3
#
# Here we will align the image and light curve representations with contrastive learning, using CLIP (https://openai.com/research/clip). Optionally, we can use the light curve encoder we trained previously.
#
# <img src=./assets/clip.png alt= “” width=1024>
#
# - From https://openai.com/research/clip.

# ## Aligning light curves and galaxy images in a common embedding space

# In[1]:


import os, sys

sys.path.append("../")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from PIL import Image

import torch
from einops import rearrange

from IPython.display import Image as IPImage

# ### Data preprocessing


# data_dir = "../data/ZTFBTS/"  # If unzipped locally
data_dir = "/ocean/projects/phy230064p/shared/ZTFBTS/"  # If running on Bridges-2


dir_host_imgs = f"{data_dir}/hostImgs/"
host_imgs = []

for filename in os.listdir(dir_host_imgs):
    file_path = os.path.join(dir_host_imgs, filename)
    if file_path.endswith(".png"):
        host_img = Image.open(file_path).convert("RGB")
        host_img = np.asarray(host_img)
        host_imgs.append(host_img)

host_imgs = np.array(host_imgs)

host_imgs = torch.from_numpy(host_imgs).float()
host_imgs = rearrange(host_imgs, "b h w c -> b c h w")

# Normalize
host_imgs /= 255.0


# Load light curves and pre-process them just like in the previous notebook.


dir_light_curves = f"{data_dir}/light-curves/"


def open_light_curve_csv(filename):
    file_path = os.path.join(dir_light_curves, filename)
    df = pd.read_csv(file_path)
    return df


light_curve_df = open_light_curve_csv("ZTF18aailmnv.csv")
light_curve_df.head()


bands = ['R', 'g']
nband = len(bands)
n_max_obs = 100

lightcurve_files = os.listdir(dir_light_curves)

# For entries with > n_max_obs observations, randomly sample n_max_obs observations (hmjd, mag, and magerr with same sample) from the light curve
# Pad the entries to n_max_obs observations with zeros and create a mask array
mask_list = []
mag_list = []
magerr_list = []
time_list = []

for filename in tqdm(lightcurve_files):
    if filename.endswith(".csv"):
        light_curve_df = open_light_curve_csv(filename)

        # Make sure the csv contains 'time', 'mag', 'magerr', and 'band' columns
        if not all(col in light_curve_df.columns for col in ['time', 'mag', 'magerr', 'band']):
            continue

        time_concat, mag_concat, magerr_concat, mask_concat = [], [], [], [] 
        for band in bands: 
            df_band = light_curve_df[light_curve_df['band'] == band]

            if len(df_band['mag']) > n_max_obs:
                mask = np.ones(n_max_obs, dtype=bool)
                indices = np.random.choice(len(df_band['mag']), n_max_obs)
                time = df_band['time'].values[indices]
                mag = df_band['mag'].values[indices]
                magerr = df_band['magerr'].values[indices]
            else:
                mask = np.zeros(n_max_obs, dtype=bool)
                mask[:len(df_band['mag'])] = True

                # Pad the arrays with zeros
                time = np.pad(df_band['time'], (0, n_max_obs - len(df_band['time'])), 'constant')
                mag = np.pad(df_band['mag'], (0, n_max_obs - len(df_band['mag'])), 'constant')
                magerr = np.pad(df_band['magerr'], (0, n_max_obs - len(df_band['magerr'])), 'constant')

            mask_concat += list(mask)
            time_concat += list(time)
            mag_concat += list(mag)
            magerr_concat += list(magerr)
            
        mask_list.append(mask_concat)
        time_list.append(time_concat)
        mag_list.append(mag_concat)
        magerr_list.append(magerr_concat)

time_ary = np.array(time_list)
mag_ary = np.array(mag_list)
magerr_ary = np.array(magerr_list)
mask_ary = np.array(mask_list)


# Inspect shapes
print(f'{time_ary.shape}, {mag_ary.shape}, {magerr_ary.shape}, {mask_ary.shape}', flush=True) 


# Plot some corresponding light curves and host images (in two columns)
n_rows = 5
n_cols = 2
fig, axs = plt.subplots(n_rows, n_cols, figsize=(10, 20))
l = len(time_ary[0])//nband

for i in range(n_rows):
    axs[i, 0].imshow(host_imgs[i].permute(1, 2, 0))
    axs[i, 0].set_title("Host Image")
    for j in range(nband): 
        t, m, mag, e = time_ary[i][j*l:(j+1)*l], mask_ary[i][j*l:(j+1)*l], mag_ary[i][j*l:(j+1)*l], magerr_ary[i][j*l:(j+1)*l]
        axs[i, 1].errorbar(t[m], mag[m], yerr=e[m], fmt='o')
    axs[i, 1].set_title("Light Curve")


# Banner image
colors = ['firebrick', 'dodgerblue']
n_pairs_per_row = 3
n_rows = 5  # Assuming we still have 5 rows of data

# Adjusting the number of columns
n_cols = n_pairs_per_row * 2  # Two columns per pair

# Creating a new big grid layout for the plots
fig, axs = plt.subplots(n_rows, n_cols, figsize=(50, 30))

for i in range(n_rows):
    for j in range(n_pairs_per_row):
        # Calculating the index for each unique data pair
        index = i * n_pairs_per_row + j

        # Plotting unique host images
        img_col = j * 2  # Column index for image
        axs[i, img_col].imshow(host_imgs[index].permute(1, 2, 0))
        # axs[i, img_col].set_title(f"Host Image {index+1}")
        axs[i, img_col].axis('off')  # Turn off axis for images

        # Plotting unique light curves
        lc_col = img_col + 1  # Column index for light curve
        for nb in range(nband): 
            t, m, mag, e = time_ary[index][nb*l:(nb+1)*l], mask_ary[index][nb*l:(nb+1)*l], mag_ary[index][nb*l:(nb+1)*l], magerr_ary[index][nb*l:(nb+1)*l]
        
            axs[i, lc_col].errorbar(t[m], 
                                    mag[m], 
                                    yerr=e[m], 
                                    fmt='o', ms=14, color=colors[nb]) 
        
        # Turn off axis labels
        axs[i, lc_col].set_xticklabels([])
        axs[i, lc_col].set_yticklabels([]) 

        # Increase thickness of axis spines
        axs[i, lc_col].spines['top'].set_linewidth(2.5)
        axs[i, lc_col].spines['right'].set_linewidth(2.5)
        axs[i, lc_col].spines['bottom'].set_linewidth(2.5)
        axs[i, lc_col].spines['left'].set_linewidth(2.5)
           

plt.tight_layout()
plt.savefig("./assets/banner.png")


# Looks good so far!

# ### Image encoder

# Out host images are pretty simple 60x60 images. Let's try a [ConvMixer](https://arxiv.org/abs/2201.09792) architecture.

# From https://arxiv.org/pdf/2201.09792.pdf
IPImage(filename="assets/convmixer.png", width=1024)


# In[10]:


import torch
import torch.nn as nn
from convmixer_model import ConvMixer


# ### Lightcurve encode


import math
from models.transformer_utils import Transformer


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

    def __init__(self, n_out, nband=1, agg='mean', **kwargs):
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
            self.band_emb = nn.Embedding(nband, kwargs['emb'])

        self.projection = nn.Linear(kwargs["emb"], n_out)
        
        # If using attention, initialize a learnable query vector
        if self.agg == 'attn':
            self.query = nn.Parameter(torch.rand(kwargs['emb']))

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
            onehot = torch.linspace(0, self.nband-1, self.nband).type(torch.LongTensor).repeat_interleave(x.shape[1]//self.nband)
            onehot = onehot.to(t.device)  # (T,)
            b_emb = self.band_emb(onehot).unsqueeze(0).repeat((x.shape[0], 1, 1)) # (T, D) -> (B, T, D) 
            x = x + b_emb 
            
        x = self.transformer(x, mask)  # (B, T, D)

        # Zero out the masked values
        x = x * mask[:, :, None]

        if self.agg == 'mean':
            x = x.sum(dim=1) / mask.sum(dim=1)[:, None]
        elif self.agg == 'max':
            x = x.max(dim=1)[0]
        elif self.agg == 'attn':
            q = self.query.unsqueeze(0).repeat(x.shape[0], 1, 1)  # Duplicate the query across the batch dimension
            k = v = x
            x, _ = nn.MultiheadAttention(embed_dim=128, num_heads=2, dropout=0.0, batch_first=True)(q, k, v)
            x = x.squeeze(1)  # (B, 1, D) -> (B, D)

        x = self.projection(x)
        return x


transformer = TransformerWithTimeEmbeddings(n_out=128, emb=128, nband=nband, heads=1, depth=1)


# In[14]:


# Time and mag tensors
time = torch.from_numpy(time_ary).float()
mag = torch.from_numpy(mag_ary).float()
mask = torch.from_numpy(mask_ary).bool()


# In[15]:


# Pass a batch through
transformer(mag[:4][..., None], time[:4], mask[:4]).shape


# ### Create dataset

# In[16]:


mag.shape


# In[17]:


tiny_host_imgs = host_imgs[:1000]
tiny_mag = mag[:1000]
tiny_time = time[:1000]
tiny_mask = mask[:1000]


# In[18]:


from torch.utils.data import TensorDataset, DataLoader, random_split

val_fraction = 0.05
batch_size = 32
n_samples_val = int(val_fraction * tiny_mag.shape[0])

dataset = TensorDataset(tiny_host_imgs, tiny_mag, tiny_time, tiny_mask)

dataset_train, dataset_val = random_split(
    dataset, [tiny_mag.shape[0] - n_samples_val, n_samples_val]
)
train_loader = DataLoader(
    dataset_train, batch_size=batch_size, num_workers=8, pin_memory=True, shuffle=True
)
val_loader = DataLoader(
    dataset_val, batch_size=batch_size, num_workers=8, pin_memory=True, shuffle=False
)


# ### Contrastive-style losses


import torch.nn.functional as F
import pytorch_lightning as pl


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
    print(f"just clip loss {loss}")

    if image_encoder and lightcurve_encoder:
        picture_loss_reg = 0
        for param in image_encoder.parameters():
            picture_loss_reg += 100 * torch.mean(torch.abs(param))
        if printing:
            print(f"picture {picture_loss_reg}")
        loss += picture_loss_reg
        light_loss_reg = 0
        for param in lightcurve_encoder.parameters():
            light_loss_reg += 100 * torch.mean(torch.abs(param))
        if printing:
            print(f"light {light_loss_reg}")
        loss += light_loss_reg
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
        self.lightcurve_encoder = TransformerWithTimeEmbeddings(nband=nband, **transformer_kwargs)
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


# Play around with various hyperparameters. Does the loss go down/converge? **Is there actually a strong correlation between the host images and light curves?** How can you test this? (E.g. -> overfit on a small batch).

# ## Downstream task: retrieval

# Try your hand at various downstream tasks: compute embeddings over images, see if they are able to identify similar images/light curves better than the base model. See if the model has learned useful joint representations by trying retrieval across modalities (image to light curve and vice versa).

# In[25]:
print("test 11")


clip_model = clip_model.eval()


# In[26]:

print("test 12")


def cosine_similarity(a, b, temperature=1):
    a_norm = a / a.norm(dim=-1, keepdim=True)
    b_norm = b / b.norm(dim=-1, keepdim=True)

    logits = a_norm @ b_norm.T * temperature
    return logits.squeeze()


# In[27]:


emb_host = clip_model.image_embeddings_with_projection(host_imgs.detach())

print("test 1")

# In[28]:


def plot_retrieved_im(idx_src):
    emb_src = clip_model.image_embeddings_with_projection(
        host_imgs[idx_src : idx_src + 1]
    )

    # Compute cosine similarity between host images and source image
    cos_sim = cosine_similarity(
        emb_host,
        emb_src,
    )

    # Get the top 10 most similar host images
    top_10_idx = torch.argsort(cos_sim, descending=True)[:10]
    print(top_10_idx)

    # Plot the top 10 most similar host images in on row
    fig, axs = plt.subplots(1, 10, figsize=(20, 5))

    for i, idx in enumerate(top_10_idx):
        axs[i].imshow(host_imgs[idx].permute(1, 2, 0))
        axs[i].set_title(f"Similarity: {cos_sim[idx]:.6f}")

    # Plot the source image
    plt.figure(figsize=(4, 4))
    plt.imshow(host_imgs[idx_src].permute(1, 2, 0))
    plt.title("Source Image")


plot_retrieved_im(33)

# In[29]:

plot_retrieved_im(16)

# Try the same thing with light curves.
# Light curve retrieval
emb_host = clip_model.lightcurve_embeddings_with_projection(
    mag.detach(), time.detach(), mask.detach()
)

print("test 2")
# In[31]:


idx_src = 4
emb_src = clip_model.lightcurve_embeddings_with_projection(
    mag[idx_src : idx_src + 1], time[idx_src : idx_src + 1], mask[idx_src : idx_src + 1]
)


# In[32]:


cos_sim = cosine_similarity(emb_host, emb_src)

# Plot the top 10 most similar light curves in a row

top_10_idx = torch.argsort(cos_sim, descending=True)[:10]

fig, axs = plt.subplots(1, 10, figsize=(40, 5))

for i, idx in enumerate(top_10_idx):
    axs[i].errorbar(
        time_ary[idx][mask_ary[idx]],
        mag_ary[idx][mask_ary[idx]],
        yerr=magerr_ary[idx][mask_ary[idx]],
        fmt="o",
    )
    axs[i].set_title(f"Similarity: {cos_sim[idx]:.6f}")

print(top_10_idx)
# Plot the source light curve
plt.figure(figsize=(4, 4))
plt.errorbar(
    time_ary[idx_src][mask_ary[idx_src]],
    mag_ary[idx_src][mask_ary[idx_src]],
    yerr=magerr_ary[idx_src][mask_ary[idx_src]],
    fmt="o",
)
plt.title("Source Light Curve")
plt.savefig("sourceimage.png")
plt.close()


print("test 3")
# Looks like we can pick up similar light curves. Can you improve the model -- the light curves encoder (see last notebook), the image model, or the loss function?

# In[33]:


# Light curve retrieval
emb_host = clip_model.lightcurve_embeddings_with_projection(
    mag.detach(), time.detach(), mask.detach()
)


# In[34]:


idx_src = 4
emb_src = clip_model.image_embeddings_with_projection(host_imgs[idx_src : idx_src + 1])

print("test 5")
# In[35]:


cos_sim = cosine_similarity(emb_host, emb_src)

# Plot the top 10 most similar light curves in a row

top_10_idx = torch.argsort(cos_sim, descending=True)[:10]

fig, axs = plt.subplots(1, 10, figsize=(40, 5))

for i, idx in enumerate(top_10_idx):
    axs[i].errorbar(
        time_ary[idx][mask_ary[idx]],
        mag_ary[idx][mask_ary[idx]],
        yerr=magerr_ary[idx][mask_ary[idx]],
        fmt="o",
    )
    axs[i].set_title(f"Similarity: {cos_sim[idx]:.6f}")

print("test 6")

print(top_10_idx)
# Plot the source light curve
plt.figure(figsize=(4, 4))
plt.errorbar(
    time_ary[idx_src][mask_ary[idx_src]],
    mag_ary[idx_src][mask_ary[idx_src]],
    yerr=magerr_ary[idx_src][mask_ary[idx_src]],
    fmt="o",
)
plt.title("Source Light Curve")

print("test 6")
emb_host = clip_model.image_embeddings_with_projection(host_imgs.detach())

idx_src = 2
emb_src = clip_model.lightcurve_embeddings_with_projection(
    mag[idx_src : idx_src + 1], time[idx_src : idx_src + 1], mask[idx_src : idx_src + 1]
)


# Compute cosine similarity between host images and source image
cos_sim = cosine_similarity(
    emb_host,
    emb_src,
)

# Get the top 10 most similar host images
top_10_idx = torch.argsort(cos_sim, descending=True)[:10]
print(top_10_idx)

print("test 8")
# Plot the top 10 most similar host images in on row
fig, axs = plt.subplots(1, 10, figsize=(20, 5))

for i, idx in enumerate(top_10_idx):
    axs[i].imshow(host_imgs[idx].permute(1, 2, 0))
    axs[i].set_title(f"Similarity: {cos_sim[idx]:.6f}")

# Plot the source image
plt.figure(figsize=(4, 4))
plt.imshow(host_imgs[idx_src].permute(1, 2, 0))
plt.title("Source Image")
plt.savefig("sourceimage.png")
plt.close()


# ### Downstream tasks: detecting interesting samples from a big survey sample

# Since we have access to a huge collection of (bulk) light curves (see `xx_ssl_lightcurve_encoder.ipynb`), we can also try to find interesting samples from this super noisy dataset based on the learned multimodal representations, by asking which light curves from the bulk samples are similar to given "interesting" light curves from the BTS sample.

# ### Downstream tasks: fine-tuning and classification

# Since we also have access to properties like redshift, we can train a regression head on top of the learned representations, to then classify the bulk light curves.
print("test 9")


df_properties = pd.read_csv("../data/ZTFBTS/ZTFBTS_TransientTable.csv")
df_properties.head()
