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
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

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
    load_model,
)
from src.transformer_utils import Transformer
from src.utils import (
    get_valid_dir,
    set_seed,
    get_linearR2,
    get_linear_predictions,
    get_knn_predictions,
    get_knnR2,
    get_embs,
)

# Typing imports
from typing import Dict, Optional, Tuple

# Additional imports
from IPython.display import Image as IPImage

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load models
set_seed(0)

paths = [
    "models/unimodal_sp/stoic-sweep-1/epoch=397-step=56118.ckpt",
    "models/unimodal_lc/honest-sweep-1/epoch=964-step=17370.ckpt",
    "models/bimodal_clip_lcsp/skilled-sweep-1/epoch=377-step=52542.ckpt",
]  # "models/spectra-lightcurve/flowing-sweep-36/epoch=472-s dtep=70950.ckpt",
labels = ["unimodal_sp", "unimodal_lc", "bimodal_clip_lcsp"]
combs_list = [["spectra-lightcurve", "spectral"], ["spectral"], ["lightcurve"]]
regressions = [False, True]
models = []
for i, path in enumerate(paths):
    print(f"loading {labels[i]}")
    models.append(load_model(path))  # , combs_list[i], regressions[i]))

print("finished loading models")


# Data preprocessing

data_dirs = [
    "/home/thelfer1/scr4_tedwar42/thelfer1/ZTFBTS/",
    "ZTFBTS/",
    "/ocean/projects/phy230064p/shared/ZTFBTS/",
    "data/ZTFBTS/",
]
data_dir = get_valid_dir(data_dirs)

data_dirs = [
    "ZTFBTS_spectra/",
    "data/ZTFBTS_spectra/",
    # "/n/home02/gemzhang/Storage/multimodal/ZTFBTS_spectra/",
    # "/n/home02/gemzhang/Storage/multimodal/ZTFBTS_spectra/",
]
spectra_dir = get_valid_dir(data_dirs)


# Default to 1 if the environment variable is not set
cpus_per_task = int(os.getenv("SLURM_CPUS_PER_TASK", 1))

# Assuming you want to leave one CPU for overhead
num_workers = max(1, cpus_per_task - 1)
print(f"Using {num_workers} workers for data loading", flush=True)

for output, label in zip(models, labels):
    model, combinations, regression, cfg, cfg_extra_args = output

    set_seed(cfg["seed"])

    # Spectral data is cut to this length
    dataset, nband, _ = load_data(
        data_dir,
        spectra_dir,
        max_data_len_spec=cfg_extra_args["max_spectral_data_len"],
        combinations=cfg_extra_args["combinations"],
        spectral_rescalefactor=cfg_extra_args["spectral_rescalefactor"],
    )

    val_fraction = cfg_extra_args.get("val_fraction", cfg_extra_args["val_fraction"])
    # Iterate over data
    number_of_samples = len(dataset)
    n_samples_val = int(val_fraction * number_of_samples)
    dataset_train, dataset_val = random_split(
        dataset,
        [number_of_samples - n_samples_val, n_samples_val],
        torch.Generator().manual_seed(cfg["seed"]),
    )

    train_loader_no_aug = NoisyDataLoader(
        dataset_train,
        batch_size=cfg["batchsize"],
        noise_level_img=0,
        noise_level_mag=0,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        combinations=cfg_extra_args["combinations"],
    )

    val_loader_no_aug = NoisyDataLoader(
        dataset_val,
        batch_size=cfg["batchsize"],
        noise_level_img=0,
        noise_level_mag=0,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        combinations=cfg_extra_args["combinations"],
    )

    model = model.to(device)
    model.eval()

    y_true_val = []
    y_pred_val = []

    for batch in val_loader_no_aug:
        # Send them all existing tensors to the device
        (
            x_img,
            x_lc,
            t_lc,
            mask_lc,
            x_sp,
            t_sp,
            mask_sp,
            redshift,
            classification,
        ) = batch

        if regression:
            if "host_galaxy" in cfg_extra_args["combinations"]:
                x_img = x_img.to(device)
            if "lightcurve" in cfg_extra_args["combinations"]:
                x_lc = x_lc.to(device)
                t_lc = t_lc.to(device)
                mask_lc = mask_lc.to(device)
            if "spectral" in cfg_extra_args["combinations"]:
                x_sp = x_sp.to(device)
                t_sp = t_sp.to(device)
                mask_sp = mask_sp.to(device)
            x = model(x_img, x_lc, t_lc, mask_lc, x_sp, t_sp, mask_sp)
            y_pred_val.append(x.detach().cpu().flatten())

        y_true_val.append(redshift)

    y_true = torch.cat(y_true_val, dim=0)

    y_true_train = []
    for batch in train_loader_no_aug:
        (
            x_img,
            x_lc,
            t_lc,
            mask_lc,
            x_sp,
            t_sp,
            mask_sp,
            redshift,
            classification,
        ) = batch
        y_true_train.append(redshift)

    y_true_train = torch.cat(y_true_train, dim=0)
    print("===============================")
    print(f"Model: {label}")
    print(f"Using data modalities: {cfg_extra_args['combinations']}")
    if regression:
        y_pred = torch.cat(y_pred_val, dim=0)
        # Calculating L1 value
        print(
            f"Model has an L1 value of {(torch.mean(torch.abs(y_true - y_pred))):.5f}"
        )
    else:
        embs_list, combs = get_embs(
            model, val_loader_no_aug, cfg_extra_args["combinations"], ret_combs=True
        )
        embs_list_train = get_embs(model, train_loader_no_aug, combinations)
        for i in range(len(embs_list)):
            # print(f"Train set linear regression R2 value for {combs[i]}: {get_linearR2(embs_list_train[i], y_true_train)}")
            print(f"---- {combs[i]} input ---- ")
            y_pred = get_linear_predictions(
                embs_list_train[i], y_true_train, embs_list[i], y_true
            )
            print(
                f"    Linear regression L1: {(torch.mean(torch.abs(y_true - y_pred))):.5f}"
            )
            y_pred = get_knn_predictions(
                embs_list_train[i], y_true_train, embs_list[i], y_true
            )
            print(f"    KNN L1 : {(torch.mean(torch.abs(y_true - y_pred))):.5f}")

        # for concatenated pairs of modalities
        for i in range(len(embs_list)):
            for j in range(i + 1, len(embs_list)):
                emb_concat = torch.cat([embs_list[i], embs_list[j]], dim=1)
                emb_train = torch.cat([embs_list_train[i], embs_list_train[j]], dim=1)
                print(f"---- {combs[i]} and {combs[j]} input ---- ")
                y_pred = get_linear_predictions(
                    emb_train, y_true_train, emb_concat, y_true
                )
                print(
                    f"    Linear regression L1: {(torch.mean(torch.abs(y_true - y_pred))):.5f}"
                )
                y_pred = get_knn_predictions(
                    emb_train, y_true_train, emb_concat, y_true
                )
                print(f"    KNN L1 : {(torch.mean(torch.abs(y_true - y_pred))):.5f}")

    print("===============================")
