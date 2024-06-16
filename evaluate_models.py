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
from PIL import Image
from tqdm import tqdm


# Local application imports
from src.dataloader import (
    load_data,
    NoisyDataLoader,
)
from src.models_multimodal import (
    load_model,
)
from src.transformer_utils import Transformer
from src.utils import (
    get_valid_dir,
    set_seed,
    get_linear_predictions,
    get_knn_predictions,
    get_embs,
    is_subset,
    process_data_loader,
    print_metrics_in_latex,
    calculate_metrics,
    get_checkpoint_paths,
)

import pandas as pd

# Typing imports
from typing import Dict, Optional, Tuple

# Additional imports
from IPython.display import Image as IPImage

device = "cuda" if torch.cuda.is_available() else "cpu"

# Specify the root directory of the sweeps
root_directory = (
    "/path/to/your/sweep/folder"  # Change this to your specific root directory path
)
smallest_ckpts = get_checkpoint_paths(root_directory)

# Printing the results
for sweep, path in smallest_ckpts.items():
    print(f"Sweep: {sweep}, Path: {path}")


# Load models
set_seed(0)

paths = [
    # "models/pretrained_sim_lc_finetuned_lc/pleasant-sweep-1//epoch=27-step=51968.ckpt",
    # "models/pretrained_sim_lc_finetuned_lc/pleasant-sweep-1/epoch=19817-step=19818.ckpt",
    "models/clip-real/swept-sweep-1/epoch=347-step=48372.ckpt",
    "models/clip-simpretrain-clipreal/daily-sweep-7/epoch=35-step=5004.ckpt",
]  # "ENDtoEND",
labels = ["masked-lc-pretraining", "clip-real", "clip-simpretrain-clipreal"]
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
regression_metrics_list = []
classification_metrics_list = []


for output, label in zip(models, labels):
    (
        model,
        combinations,
        regression,
        classification,
        n_classes,
        cfg,
        cfg_extra_args,
        train_filenames,
        val_filenames,
    ) = output

    set_seed(cfg["seed"])

    # Spectral data is cut to this length
    dataset_train, nband, filenames_read, _ = load_data(
        data_dir,
        spectra_dir,
        max_data_len_spec=cfg_extra_args["max_spectral_data_len"],
        combinations=cfg_extra_args["combinations"],
        spectral_rescalefactor=cfg_extra_args["spectral_rescalefactor"],
        filenames=train_filenames,
    )

    # Check that the filenames read are a subset of the training filenames from the already trained models
    assert is_subset(filenames_read, train_filenames)

    dataset_val, nband, filenames_read, _ = load_data(
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

    y_true, y_true_label, y_pred = process_data_loader(
        val_loader_no_aug,
        regression,
        classification,
        device,
        model,
        combinations=cfg_extra_args["combinations"],
    )
    y_true_train, y_true_train_label, _ = process_data_loader(
        train_loader_no_aug,
        regression,
        classification,
        device,
        model,
        combinations=cfg_extra_args["combinations"],
    )

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
        metrics = calculate_metrics(
            y_true, y_pred, label, format_combinations(cfg_extra_args["combinations"])
        )

        regression_metrics_list.append(metrics)
        """x
        elif classification:
            y_pred = torch.cat(y_pred_val, dim=0)
            metrics = calculate_metrics(
                y_true, y_pred, label, format_combinations(cfg_extra_args["combinations"], task='classification')
            )
            classification_metrics_list.append(metrics)
        """
    else:
        embs_list, combs = get_embs(
            model, val_loader_no_aug, cfg_extra_args["combinations"], ret_combs=True
        )
        embs_list_train = get_embs(model, train_loader_no_aug, combinations)
        for i in range(len(embs_list)):
            # print(f"Train set linear regression R2 value for {combs[i]}: {get_linearR2(embs_list_train[i], y_true_train)}")
            print(f"---- {combs[i]} input ---- ")
            for task in ["regression", "classification"]:
                if task == "regression":
                    y_pred_linear = get_linear_predictions(
                        embs_list_train[i],
                        y_true_train,
                        embs_list[i],
                        y_true,
                        task=task,
                    )
                    y_pred_knn = get_knn_predictions(
                        embs_list_train[i],
                        y_true_train,
                        embs_list[i],
                        y_true,
                        task=task,
                    )
                    metrics = calculate_metrics(
                        y_true, y_pred_linear, label + "+Linear", combs[i], task=task
                    )
                    regression_metrics_list.append(metrics)
                    metrics = calculate_metrics(
                        y_true, y_pred_knn, label + "+KNN", combs[i], task=task
                    )
                    regression_metrics_list.append(metrics)

                elif task == "classification":
                    y_pred_linear = get_linear_predictions(
                        embs_list_train[i],
                        y_true_train_label,
                        embs_list[i],
                        y_true_label,
                        task=task,
                    )
                    y_pred_knn = get_knn_predictions(
                        embs_list_train[i],
                        y_true_train_label,
                        embs_list[i],
                        y_true_label,
                        task=task,
                    )
                    metrics = calculate_metrics(
                        y_true_label,
                        y_pred_linear,
                        label + "+Linear",
                        combs[i],
                        task=task,
                    )
                    classification_metrics_list.append(metrics)
                    metrics = calculate_metrics(
                        y_true_label, y_pred_knn, label + "+KNN", combs[i], task=task
                    )
                    classification_metrics_list.append(metrics)

        # for concatenated pairs of modalities
        for i in range(len(embs_list)):
            for j in range(i + 1, len(embs_list)):
                emb_concat = torch.cat([embs_list[i], embs_list[j]], dim=1)
                emb_train = torch.cat([embs_list_train[i], embs_list_train[j]], dim=1)
                print(f"---- {combs[i]} and {combs[j]} input ---- ")
                for task in ["regression", "classification"]:
                    if task == "regression":
                        y_pred_linear = get_linear_predictions(
                            embs_list_train[i],
                            y_true_train,
                            embs_list[i],
                            y_true,
                            task=task,
                        )
                        y_pred_knn = get_knn_predictions(
                            embs_list_train[i],
                            y_true_train,
                            embs_list[i],
                            y_true,
                            task=task,
                        )
                        metrics = calculate_metrics(
                            y_true,
                            y_pred_linear,
                            label + "+Linear",
                            combs[i] + " and " + combs[j],
                            task=task,
                        )
                        regression_metrics_list.append(metrics)
                        metrics = calculate_metrics(
                            y_true,
                            y_pred_knn,
                            label + "+KNN",
                            combs[i] + " and " + combs[j],
                            task=task,
                        )
                        regression_metrics_list.append(metrics)
                    elif task == "classification":
                        y_pred_linear = get_linear_predictions(
                            embs_list_train[i],
                            y_true_train_label,
                            embs_list[i],
                            y_true_label,
                            task=task,
                        )
                        y_pred_knn = get_knn_predictions(
                            embs_list_train[i],
                            y_true_train_label,
                            embs_list[i],
                            y_true_label,
                            task=task,
                        )
                        metrics = calculate_metrics(
                            y_true_label,
                            y_pred_linear,
                            label + "+Linear",
                            combs[i] + " and " + combs[j],
                            task=task,
                        )
                        classification_metrics_list.append(metrics)
                        metrics = calculate_metrics(
                            y_true_label,
                            y_pred_knn,
                            label + "+KNN",
                            combs[i] + " and " + combs[j],
                            task=task,
                        )
                        classification_metrics_list.append(metrics)
    print("===============================")

# Convert metrics list to a DataFrame

print_metrics_in_latex(classification_metrics_list)
print_metrics_in_latex(regression_metrics_list)
