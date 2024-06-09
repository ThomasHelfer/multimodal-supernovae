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

import pandas as pd

# Typing imports
from typing import Dict, Optional, Tuple

# Additional imports
from IPython.display import Image as IPImage

device = "cuda" if torch.cuda.is_available() else "cpu"


def calculate_metrics(y_true, y_pred, label, combination):
    """
    Calculate L1 and L2 norms to assess the accuracy of predictions against true values.

    Parameters:
    - y_true (torch.Tensor): The true values against which predictions are evaluated.
    - y_pred (torch.Tensor): The predicted values to be evaluated.
    - label (str): Label describing the model or configuration being evaluated.
    - combination (str): Description of the data or feature combination used for the model.

    Returns:
    - dict: A dictionary containing the calculated metrics including L1 and L2 norms for the model predictions. Each key describes the metric:
            - 'Model': The label of the model or configuration.
            - 'Combination': Description of the feature or data combination.
            - 'L1': The L1 norm (mean absolute error) of the prediction error.
            - 'L2': The L2 norm (root mean squared error) of the prediction error.
    """
    # Calculate L1 and L2 norms for the predictions
    l1 = torch.mean(torch.abs(y_true - y_pred)).item()
    l2 = torch.sqrt(torch.mean((y_true - y_pred) ** 2)).item()
    R2 = 1 - (torch.sum((y_true - y_pred) ** 2) / torch.sum((y_true - torch.mean(y_true)) ** 2)).item()

    # Calculate the residuals
    delta_z = y_true - y_pred


    # Outliers based on a fixed threshold
    outliers = torch.abs(delta_z) > 0.07
    non_outliers = ~outliers

    # calulate the fraction of outliers
    OLF = torch.mean(outliers.float()).item()


    # Compile the results into a metrics dictionary
    metrics = {
        'Model': label,
        'Combination': combination,
        'L1': l1,
        'L2': l2,
        'R2': R2,
        'OLF': OLF,
    }
    
    return metrics

# Load models
set_seed(0)

paths = [
    "models/unimodal_sp/stoic-sweep-1/epoch=397-step=56118.ckpt",
    "models/unimodal_lc/honest-sweep-1/epoch=964-step=17370.ckpt",
    "models/bimodal_clip_lcsp/skilled-sweep-1/epoch=377-step=52542.ckpt",
]  # "models/spectra-lightcurve/flowing-sweep-36/epoch=472-s dtep=70950.ckpt",
labels = ["ENDtoEND", "ENDtoEND", "CLIP"]
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

#Keeping track of all metrics
metrics_list = []


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

    def format_combinations(combinations):
        if len(combinations) > 1:
            return ', '.join(combinations[:-1]) + ' and ' + combinations[-1]
        elif combinations:
            return combinations[0]
        return ''

    if regression:
        y_pred = torch.cat(y_pred_val, dim=0)

        metrics = calculate_metrics(y_true, y_pred, label, format_combinations(cfg_extra_args['combinations']))        
        
        metrics_list.append(metrics)

    else:
        embs_list, combs = get_embs(
            model, val_loader_no_aug, cfg_extra_args["combinations"], ret_combs=True
        )
        embs_list_train = get_embs(model, train_loader_no_aug, combinations)
        for i in range(len(embs_list)):
            # print(f"Train set linear regression R2 value for {combs[i]}: {get_linearR2(embs_list_train[i], y_true_train)}")
            print(f"---- {combs[i]} input ---- ")
            y_pred_linear = get_linear_predictions(
                embs_list_train[i], y_true_train, embs_list[i], y_true
            )

            y_pred_knn = get_knn_predictions(
                embs_list_train[i], y_true_train, embs_list[i], y_true
            )

            metrics = calculate_metrics(y_true, y_pred_linear, label+ '+Linear', combs[i])        
            metrics_list.append(metrics)

            metrics = calculate_metrics(y_true, y_pred_knn, label+ '+KNN', combs[i])        
            metrics_list.append(metrics)

        # for concatenated pairs of modalities
        for i in range(len(embs_list)):
            for j in range(i + 1, len(embs_list)):
                emb_concat = torch.cat([embs_list[i], embs_list[j]], dim=1)
                emb_train = torch.cat([embs_list_train[i], embs_list_train[j]], dim=1)
                print(f"---- {combs[i]} and {combs[j]} input ---- ")
                y_pred_linear = get_linear_predictions(
                    emb_train, y_true_train, emb_concat, y_true
                )

                y_pred_knn = get_knn_predictions(
                    emb_train, y_true_train, emb_concat, y_true
                )

                metrics = calculate_metrics(y_true, y_pred_linear, label+ '+Linear', combs[i] + ' and ' + combs[j])        
                metrics_list.append(metrics)

                metrics = calculate_metrics(y_true, y_pred_knn, label+ '+KNN', combs[i] + ' and ' + combs[j])        
                metrics_list.append(metrics)
    print("===============================")

# Convert metrics list to a DataFrame
metrics_df = pd.DataFrame(metrics_list)

# Save the DataFrame to a CSV file
metrics_df.to_csv('model_metrics.csv', index=False)

# Define formatters for the float columns to format them to three decimal places
float_formatter = lambda x: f"{x:.4f}"
float_formatter_R2 = lambda x: f"{x:.2f}"
formatters = {
    'L1': float_formatter,
    'L2': float_formatter,
    'R2': float_formatter_R2,
    'OLF': float_formatter,
    'MAD': float_formatter,
}

latex_code = metrics_df.to_latex(index=False,formatters=formatters, escape=False, na_rep='')

print(latex_code)