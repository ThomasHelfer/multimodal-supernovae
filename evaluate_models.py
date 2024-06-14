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
from sklearn.metrics import f1_score, average_precision_score, accuracy_score, recall_score, balanced_accuracy_score

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
    is_subset,
)

import pandas as pd

# Typing imports
from typing import Dict, Optional, Tuple

# Additional imports
from IPython.display import Image as IPImage

device = "cuda" if torch.cuda.is_available() else "cpu"


def calculate_metrics(y_true, y_pred, label, combination, task='redshift'):
    """
    Calculates performance metrics (for both classification and redshift estimation) to assess the accuracy of predictions against true values.

    Parameters:
    - y_true (torch.Tensor): The true values against which predictions are evaluated.
    - y_pred (torch.Tensor): The predicted values to be evaluated.
    - label (str): Label describing the model or configuration being evaluated.
    - combination (str): Description of the data or feature combination used for the model.
    - task (str): the downstream task being done; can be 'redshift' or 'classification'.

    Returns:
    - dict: A dictionary containing the calculated metrics. Each key describes the metric.
            - 'Model': The label of the model or configuration.
            - 'Combination': Description of the feature or data combination.
         For redshift regression:
            - 'L1': The L1 norm (mean absolute error) of the prediction error.
            - 'L2': The L2 norm (root mean squared error) of the prediction error.
            - 'R2': The coefficient of determination of the prediction error.
            - 'OLF': The outlier fraction of the prediction error.
        For 3- or 5-way classification:
            - 'micro-f1': The micro-averaged f1-score (NOT balanced across classes).
            - 'micro-precision': The micro-averaged precision (true positives / (true positives + false positives), NOT balanced across classes).
            - 'micro-recall': The micro-averaged precision (true positives / (true positives + false negatives), NOT balanced across classes).
            - 'micro-acc': The micro-averaged accuracy (averaged across all points, NOT balanced across classes).

            - 'macro-f1': The macro-averaged f1-score (balanced across classes).
            - 'macro-precision': The macro-averaged precision (true positives / (true positives + false positives), balanced across classes).
            - 'macro-recall': The macro-averaged precision (true positives / (true positives + false negatives), balanced across classes).
            - 'macro-acc': The macro-averaged accuracy (balanced across classes).
    """
    if task == 'redshift':
        # Calculate L1 and L2 norms for the predictions
        l1 = torch.mean(torch.abs(y_true - y_pred)).item()
        l2 = torch.sqrt(torch.mean((y_true - y_pred) ** 2)).item()
        R2 = (
            1
            - (
                torch.sum((y_true - y_pred) ** 2)
                / torch.sum((y_true - torch.mean(y_true)) ** 2)
            ).item()
        )

        # Calculate the residuals
        delta_z = y_true - y_pred

        # Outliers based on a fixed threshold
        outliers = torch.abs(delta_z) > 0.07
        non_outliers = ~outliers

        # calulate the fraction of outliers
        OLF = torch.mean(outliers.float()).item()

        # Compile the results into a metrics dictionary
        metrics = {
            "Model": label,
            "Combination": combination,
            "L1": l1,
            "L2": l2,
            "R2": R2,
            "OLF": OLF,
        }
    elif task == 'classification':
        y_true = y_true.cpu().numpy()
        y_pred = y_pred.cpu().numpy()

        #micro f1-score
        micF1 = f1_score(y_true, y_pred, average='micro')

        #micro precision
        micPrec = precision_score(y_true, y_pred, average='micro')

        #micro recall
        micRec = recall_score(y_true, y_pred, average='micro')

        #micro accuracy
        micAcc = accuracy_score(y_true, y_pred, normalize=True)

        #macro f1-score
        micF1 = f1_score(y_true, y_pred, average='macro')

        #macro precision
        macPrec = precision_score(y_true, y_pred, average='macro')

        #macro recall
        macRec = recall_score(y_true, y_pred, average='macro')

        #macro accuracy
        macAcc = balanced_accuracy_score(y_true, y_pred)

        # Compile the results into a metrics dictionary
        metrics = {
            "Model": label,
            "Combination": combination,
            "micro-f1": micF1,
            "micro-precision": micPrec,
            "micro-recall": micRec,
            "micro-acc": micAcc,
            "macro-f1": macF1,
            "macro-precision": macPrec,
            "macro-recall": macRec,
            "macro-acc": macAcc,
        }

    else:
        raise ValueError("Could not understand the task! Please set task to 'redshift' or 'classification'.")

    return metrics


# Load models
set_seed(0)

paths = [
   # "/home/thelfer1/data_tedwar42/thelfer1/Multimodal-hackathon-2024/models/unimodal_lc/radiant-sweep-1/epoch=27-step=51968.ckpt",
    "/home/thelfer1/data_tedwar42/thelfer1/Multimodal-hackathon-2024/models/pretrained_sim_lc_finetuned_lc/pleasant-sweep-1/epoch=2024-step=2025.ckpt",
    "/home/thelfer1/data_tedwar42/thelfer1/Multimodal-hackathon-2024/models/clip-real/swept-sweep-1/epoch=347-step=48372.ckpt",
    "/home/thelfer1/data_tedwar42/thelfer1/Multimodal-hackathon-2024/models/clip-simpretrain-clipreal/daily-sweep-7/epoch=35-step=5004.ckpt"
]  #"ENDtoEND",
labels = [ "masked-lc-pretraining", "clip-real","clip-simpretrain-clipreal"]
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

# Keeping track of all metrics
metrics_list = []


for output, label in zip(models, labels):
    (
        model,
        combinations,
        regression,
        cfg,
        cfg_extra_args,
        train_filenames,
        val_filenames,
    ) = output

    set_seed(cfg["seed"])

    # Spectral data is cut to this length
    dataset_train, nband, filenames_read = load_data(
        data_dir,
        spectra_dir,
        max_data_len_spec=cfg_extra_args["max_spectral_data_len"],
        combinations=cfg_extra_args["combinations"],
        spectral_rescalefactor=cfg_extra_args["spectral_rescalefactor"],
        filenames=train_filenames,
    )

    # Check that the filenames read are a subset of the training filenames from the already trained models
    assert is_subset(filenames_read, train_filenames)

    dataset_val, nband, filenames_read = load_data(
        data_dir,
        spectra_dir,
        max_data_len_spec=cfg_extra_args["max_spectral_data_len"],
        combinations=cfg_extra_args["combinations"],
        spectral_rescalefactor=cfg_extra_args["spectral_rescalefactor"],
        filenames=val_filenames,
    )

    # Check that the filenames read are a subset of the training filenames from the already trained models
    assert is_subset(filenames_read, val_filenames)

    # val_fraction = cfg_extra_args.get("val_fraction", cfg_extra_args["val_fraction"])
    ## Iterate over data
    # number_of_samples = len(dataset)
    # n_samples_val = int(val_fraction * number_of_samples)

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
            return ", ".join(combinations[:-1]) + " and " + combinations[-1]
        elif combinations:
            return combinations[0]
        return ""

    if regression:
        y_pred = torch.cat(y_pred_val, dim=0)

        metrics = calculate_metrics(
            y_true, y_pred, label, format_combinations(cfg_extra_args["combinations"])
        )

        metrics_list.append(metrics)
    elif classification:
        y_pred = torch.cat(y_pred_val, dim=0)
        metrics = calculate_metrics(
            y_true, y_pred, label, format_combinations(cfg_extra_args["combinations"], task='classification')
        )
        metrics_list.append(metrics)

    else:
        embs_list, combs = get_embs(
            model, val_loader_no_aug, cfg_extra_args["combinations"], ret_combs=True
        )
        embs_list_train = get_embs(model, train_loader_no_aug, combinations)
        for i in range(len(embs_list)):
            # print(f"Train set linear regression R2 value for {combs[i]}: {get_linearR2(embs_list_train[i], y_true_train)}")
            print(f"---- {combs[i]} input ---- ")
            for task in ['regression', 'classification']:
                y_pred_linear = get_linear_predictions(embs_list_train[i], y_true_train, embs_list[i], y_true, task=task)
                y_pred_knn = get_knn_predictions(embs_list_train[i], y_true_train, embs_list[i], y_true, task=task)

                metrics = calculate_metrics(y_true, y_pred_linear, label + "+Linear", combs[i], task=task)
                metrics_list.append(metrics)

                metrics = calculate_metrics(y_true, y_pred_knn, label + "+KNN", combs[i], task=task)
                metrics_list.append(metrics)

        # for concatenated pairs of modalities
        for i in range(len(embs_list)):
            for j in range(i + 1, len(embs_list)):
                emb_concat = torch.cat([embs_list[i], embs_list[j]], dim=1)
                emb_train = torch.cat([embs_list_train[i], embs_list_train[j]], dim=1)
                print(f"---- {combs[i]} and {combs[j]} input ---- ")
                for task in ['regression', 'classification']:
                    y_pred_linear = get_linear_predictions(emb_train, y_true_train, emb_concat, y_true, task=task)

                    y_pred_knn = get_knn_predictions(emb_train, y_true_train, emb_concat, y_true, task=task)

                    metrics = calculate_metrics(y_true, y_pred_linear, label + "+Linear", combs[i] + " and " + combs[j], task=task)
                    metrics_list.append(metrics)

                    metrics = calculate_metrics(y_true, y_pred_knn, label + "+KNN", combs[i] + " and " + combs[j], task=task)
                    metrics_list.append(metrics)
    print("===============================")

# Convert metrics list to a DataFrame
metrics_df = pd.DataFrame(metrics_list)

# Save the DataFrame to a CSV file
metrics_df.to_csv("model_metrics.csv", index=False)

# Define formatters for the float columns to format them to three decimal places
float_formatter = lambda x: f"{x:.3f}"
formatters = {}

#iterate over all metrics and truncate to 3 decimal places
for colname in list(set(metrics_df.columns.values) - set(['Combination', 'Model'])):
    formatters[colname] = float_formatter

latex_code = metrics_df.to_latex(
    index=False, formatters=formatters, escape=False, na_rep=""
)

print(latex_code)
