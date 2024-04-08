import torch
import pytorch_lightning as pl
import argparse
import os
from ruamel.yaml import YAML

import numpy as np
from torch.utils.data import TensorDataset, DataLoader, random_split
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from src.models_multimodal import LightCurveImageCLIP
from src.utils import (
    get_valid_dir,
    find_indices_in_arrays,
    LossTrackingCallback,
    plot_loss_history,
    get_embs,
    plot_ROC_curves,
    get_savedir,
    set_seed,
)
from src.dataloader import (
    load_images,
    load_lightcurves,
    load_spectras,
    load_data,
    plot_lightcurve_and_images,
    NoisyDataLoader,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process arguments")
    parser.add_argument(
        "--ckpt_path", type=str, default=None, help="Path to the checkpoint directory"
    )
    parser.add_argument(
        "--runname", 
        type=str, 
        default=None, 
        help="Name of the run for saving; if None, will be set to the next available number"
        )
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="Path for config if not resuming from checkpoint",
    )
    args = parser.parse_args()

    save_dir, cfg = get_savedir(args)

    set_seed(cfg["seed"])

    # Data preprocessing
    data_dirs = [
        "/home/thelfer1/scr4_tedwar42/thelfer1/ZTFBTS/",
        "ZTFBTS/",
        "/ocean/projects/phy230064p/shared/ZTFBTS/",
        "/n/home02/gemzhang/repos/Multimodal-hackathon-2024/data/ZTFBTS/",
    ]

    combinations = cfg["combinations"]
    print(f"Using combinations: {combinations}")

    # Get the first valid directory
    data_dir = get_valid_dir(data_dirs)

    # Check if the config file has a spectra key
    if "spectral" in combinations:
        data_dirs = ["ZTFBTS_spectra/", "data/ZTFBTS_spectra/"]
        spectra_dir = get_valid_dir(data_dirs)
    else:
        spectra_dir = None

    max_spectral_data_len = cfg["max_spectral_data_len"]  # Spectral data is cut to this length
    dataset, nband = load_data(
        data_dir, spectra_dir, max_data_len_spec = max_spectral_data_len, combinations=combinations
    )

    number_of_samples = len(dataset)

    val_fraction = 0.05
    batch_size = cfg["batchsize"]
    n_samples_val = int(val_fraction * number_of_samples)

    dataset_train, dataset_val = random_split(
        dataset, [number_of_samples - n_samples_val, n_samples_val]
    )

    train_loader_no_aug = NoisyDataLoader(
        dataset_train,
        batch_size=batch_size,
        noise_level_img=0,
        noise_level_mag=0,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        combinations=combinations,
    )
    val_loader_no_aug = NoisyDataLoader(
        dataset_val,
        batch_size=batch_size,
        noise_level_img=0,
        noise_level_mag=0,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        combinations=combinations,
    )

    # Define the noise levels for images and magnitude (multiplied by magerr)
    noise_level_img = 1  # Adjust as needed
    noise_level_mag = 1  # Adjust as needed

    val_noise = 0

    # Create custom noisy data loaders
    train_loader = NoisyDataLoader(
        dataset_train,
        batch_size=batch_size,
        noise_level_img=noise_level_img,
        noise_level_mag=noise_level_mag,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
        combinations=combinations,
    )
    val_loader = NoisyDataLoader(
        dataset_val,
        batch_size=batch_size,
        noise_level_img=val_noise,
        noise_level_mag=val_noise,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        combinations=combinations,
    )

    transformer_kwargs = {
        "n_out": cfg["n_out"],
        "emb": cfg["emb"],
        "heads": cfg["heads"],
        "depth": cfg["transformer_depth"],
        "dropout": cfg["dropout"],
        "time_norm": cfg["time_norm"],
        "agg": cfg["agg"],
    }

    transformer_spectral_kwargs = {
        "n_out": cfg["n_out"],
        "emb": cfg["emb_spectral"],
        "heads": cfg["heads_spectral"],
        "depth": cfg["transformer_depth_spectral"],
        "dropout": cfg["dropout"],
        "time_norm": cfg["time_norm_spectral"],
        "agg": cfg["agg_spectral"],
    }

    conv_kwargs = {
        "dim": 32,
        "depth": cfg["cnn_depth"],
        "channels": 3,
        "kernel_size": 5,
        "patch_size": 10,
        "n_out": cfg["n_out"],
        "dropout_prob": cfg["dropout"],
    }

    optimizer_kwargs = {"weight_decay": cfg["weight_decay"]}

    clip_model = LightCurveImageCLIP(
        logit_scale=cfg['logit_scale'],
        lr=cfg["lr"],
        nband=nband,
        loss="softmax",
        transformer_kwargs=transformer_kwargs,
        transformer_spectral_kwargs=transformer_spectral_kwargs,
        conv_kwargs=conv_kwargs,
        optimizer_kwargs=optimizer_kwargs,
        combinations=combinations,
    )

    # Custom call back for tracking loss
    loss_tracking_callback = LossTrackingCallback()

    device = "gpu" if torch.cuda.is_available() else "cpu"

    checkpoint_callback = ModelCheckpoint(
        dirpath=save_dir, save_top_k=2, monitor="val_loss"
    )

    early_stop_callback = EarlyStopping(
            monitor="val_loss", min_delta=0.00, patience=100, verbose=False, mode="min"
        )
    
    trainer = pl.Trainer(
        max_epochs=cfg["epochs"],
        accelerator=device,
        callbacks=[loss_tracking_callback, checkpoint_callback, early_stop_callback],
    )
    trainer.fit(
        model=clip_model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=args.ckpt_path,
    )

    plot_loss_history(
        loss_tracking_callback.train_loss_history,
        loss_tracking_callback.val_loss_history,
        path_base=save_dir,
    )

    # Get embeddings for all images and light curves
    embs_train = get_embs(
        clip_model, train_loader_no_aug, combinations
    )
    embs_val = get_embs(
        clip_model, val_loader_no_aug, combinations
    )

    plot_ROC_curves(
        embs_train, 
        embs_val,
        combinations, 
        path_base=save_dir,
    )
