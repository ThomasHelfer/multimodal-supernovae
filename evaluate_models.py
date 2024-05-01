# Standard library imports
import math
import os
import sys

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from einops import rearrange
from PIL import Image
from ruamel.yaml import YAML
from tqdm import tqdm
from torchvision.transforms import RandomRotation
from torch.utils.data import DataLoader, TensorDataset, random_split

# Local application imports
from src.dataloader import (
    load_images,
    load_lightcurves,
    plot_lightcurve_and_images,
    load_spectras,
    load_data,
    NoisyDataLoader,
    load_redshifts,
)
from src.loss import sigmoid_loss, clip_loss
from src.loss import sigmoid_loss_multimodal, clip_loss_multimodal
from src.models_multimodal import (
    ConvMixer,
    TransformerWithTimeEmbeddings,
    LightCurveImageCLIP,
    load_model,
)
from src.transformer_utils import Transformer
from src.utils import (
    get_valid_dir,
    set_seed,
    LossTrackingCallback,
    plot_loss_history,
    get_embs,
    find_indices_in_arrays,
    get_AUC,
    LossTrackingCallback,
    cosine_similarity,
    plot_ROC_curves,
    get_ROC_data,
)

# Typing imports
from typing import Dict, Optional, Tuple

# Additional imports
from IPython.display import Image as IPImage


device = "cuda" if torch.cuda.is_available() else "cpu"

# Load models
set_seed(0)

paths = ["analysis/1fscimab/radiant-sweep-1/epoch=1995-step=39920.ckpt"]
labels = ["test_model_1"]
models = []
for i, path in enumerate(paths):
    print(f"loading {labels[i]}")
    models.append(load_model(path))

print("finished loading models")


# Data preprocessing

data_dirs = [
    "/home/thelfer1/scr4_tedwar42/thelfer1/ZTFBTS/",
    "ZTFBTS/",
    "/ocean/projects/phy230064p/shared/ZTFBTS/",
    "/n/home02/gemzhang/repos/Multimodal-hackathon-2024/data/ZTFBTS/",
]
data_dir = get_valid_dir(data_dirs)

data_dirs = [
    "ZTFBTS_spectra/",
    "/n/home02/gemzhang/Storage/multimodal/ZTFBTS_spectra/",
]
spectra_dir = get_valid_dir(data_dirs)

max_spectral_data_len = 1000

# Spectral data is cut to this length
dataset, nband, _ = load_data(
    data_dir,
    spectra_dir,
    max_data_len_spec=max_spectral_data_len,
    combinations=["host_galaxy", "lightcurve", "spectral"],
)

# Default to 1 if the environment variable is not set
cpus_per_task = int(os.getenv("SLURM_CPUS_PER_TASK", 1))

# Assuming you want to leave one CPU for overhead
num_workers = max(1, cpus_per_task - 1)
print(f"Using {num_workers} workers for data loading", flush=True)

for output, label in zip(models, labels):
    model, combinations, regression, cfg, cfg_extra_args = output

    set_seed(cfg["seed"])

    # Making sure that spectral lengths are the same
    assert max_spectral_data_len == cfg_extra_args["max_spectral_data_len"]

    val_fraction = cfg_extra_args["val_fraction"]
    # Iterate over data
    number_of_samples = len(dataset)
    n_samples_val = int(val_fraction * number_of_samples)
    dataset_train, dataset_val = random_split(
        dataset, [number_of_samples - n_samples_val, n_samples_val]
    )

    val_loader_no_aug = NoisyDataLoader(
        dataset_val,
        batch_size=cfg["batchsize"],
        noise_level_img=0,
        noise_level_mag=0,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        combinations=["host_galaxy", "lightcurve", "spectral"],
    )

    model = model.to(device)

    y_true_val = []
    y_pred_val = []

    for batch in val_loader_no_aug:
        x_img, x_lc, t_lc, mask_lc, x_sp, t_sp, mask_sp, redshift = batch

        x_img, x_lc, t_lc, mask_lc, x_sp, t_sp, mask_sp = (
            x_img.to(device),
            x_lc.to(device),
            t_lc.to(device),
            mask_lc.to(device),
            x_sp.to(device),
            t_sp.to(device),
            mask_sp.to(device),
        )
        x = model(x_img, x_lc, t_lc, mask_lc, x_sp, t_sp, mask_sp)

        if regression:
            y_pred_val.append(x.detach().cpu().flatten())
            y_true_val.append(redshift)

    # Calculate R2

    y_true = torch.cat(y_true_val, dim=0)
    y_pred = torch.cat(y_pred_val, dim=0)
    r2 = 1 - (y_true - y_pred).pow(2).sum() / (y_true - y_true.mean()).pow(2).sum()

    print(f"We find our model as an R2 value of {r2}")
