import os, sys
import wandb
from ruamel.yaml import YAML
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

import numpy as np

import torch
import pytorch_lightning as pl

from torch.utils.data import TensorDataset, DataLoader, random_split

from src.models_multimodal import LightCurveImageCLIP
from src.utils import (
    set_seed,
    get_valid_dir,
    LossTrackingCallback,
    plot_ROC_curves,
    plot_loss_history,
    get_embs,
)
from src.dataloader import (
    load_images,
    load_lightcurves,
    load_data,
    plot_lightcurve_and_images,
    NoisyDataLoader,
)
from src.wandb_utils import schedule_sweep

def train_sweep(config=None):
    with wandb.init(config=config) as run:
        print(f"run name: {run.name}", flush=True)
        path_run = os.path.join(model_path, run.name)
        os.makedirs(path_run, exist_ok=True)

        cfg = wandb.config
        set_seed(cfg.seed)

        # Default to 1 if the environment variable is not set
        cpus_per_task = int(os.getenv("SLURM_CPUS_PER_TASK", 1))

        # Assuming you want to leave one CPU for overhead
        num_workers = max(1, cpus_per_task - 1)
        print(f"Using {num_workers} workers for data loading", flush=True)

        train_loader_no_aug = DataLoader(
            dataset_train,
            batch_size=cfg.batchsize,
            num_workers=num_workers,
            pin_memory=True,
            shuffle=True,
        )
        val_loader_no_aug = DataLoader(
            dataset_val,
            batch_size=cfg.batchsize,
            num_workers=num_workers,
            pin_memory=True,
            shuffle=False,
        )

        # Create custom noisy data loaders
        train_loader = NoisyDataLoader(
            dataset_train,
            batch_size=cfg.batchsize,
            noise_level_img=noise_level_img,
            noise_level_mag=noise_level_mag,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        val_loader = NoisyDataLoader(
            dataset_val,
            batch_size=cfg.batchsize,
            noise_level_img=val_noise,
            noise_level_mag=val_noise,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        transformer_kwargs = {
            "n_out": 32,
            "emb": cfg.emb,
            "heads": 2,
            "depth": cfg.transformer_depth,
            "dropout": cfg.dropout,
        }
        conv_kwargs = {
            "dim": 32,
            "depth": 2,
            "channels": 3,
            "kernel_size": 5,
            "patch_size": 10,
            "n_out": 32,
            "dropout_prob": cfg.dropout,
        }
        optimizer_kwargs = {"weight_decay": cfg.weight_decay}

        clip_model = LightCurveImageCLIP(
            logit_scale=20.0,
            lr=cfg.lr,
            nband=nband,
            loss="softmax",
            transformer_kwargs=transformer_kwargs,
            conv_kwargs=conv_kwargs,
            optimizer_kwargs=optimizer_kwargs,
        )

        # Custom call back for tracking loss
        loss_tracking_callback = LossTrackingCallback()

        device = "gpu" if torch.cuda.is_available() else "cpu"

        wandb_logger = WandbLogger()
        checkpoint_callback = ModelCheckpoint(
            dirpath=path_run, save_top_k=2, monitor="val_loss"
        )

        trainer = pl.Trainer(
            max_epochs=cfg.epochs,
            accelerator=device,
            callbacks=[loss_tracking_callback, checkpoint_callback],
            logger=wandb_logger,
            enable_progress_bar=False,
        )
        trainer.fit(
            model=clip_model, train_dataloaders=train_loader, val_dataloaders=val_loader
        )

        wandb.run.summary["best_val_loss"] = np.min(
            loss_tracking_callback.val_loss_history
        )
        plot_loss_history(
            loss_tracking_callback.train_loss_history,
            loss_tracking_callback.val_loss_history,
            path_base=path_run,
        )

        # Get embeddings for all images and light curves
        embs_curves_train, embs_images_train = get_embs(clip_model, train_loader_no_aug)
        embs_curves_val, embs_images_val = get_embs(clip_model, val_loader_no_aug)

        plot_ROC_curves(
            embs_curves_train,
            embs_images_train,
            embs_curves_val,
            embs_images_val,
            path_base=path_run,
        )

        config_dict = {k: v for k, v in cfg.items()}
        with open(os.path.join(path_run, "config.yaml"), "w") as f:
            YAML().dump(config_dict, f)

        wandb.finish()


if __name__ == "__main__":
    wandb.login()

    config = sys.argv[
        1
    ]  # '/n/home02/gemzhang/repos/Multimodal-hackathon-2024/sweep_configs/config_grid.yaml'
    
    analysis_path = "./analysis/"

    sweep_id, model_path, cfg = schedule_sweep(config, analysis_path)
    print("model path: " + model_path, flush=True)

    # define constants
    val_fraction = 0.05
    # Define the noise levels for images and magnitude (multiplied by magerr)
    noise_level_img = 1  # Adjust as needed
    noise_level_mag = 1  # Adjust as needed

    val_noise = 0

    # Data preprocessing

    data_dirs = [
        "/home/thelfer1/scr4_tedwar42/thelfer1/ZTFBTS/",
        "ZTFBTS/",
        "/ocean/projects/phy230064p/shared/ZTFBTS/",
        "/n/home02/gemzhang/repos/Multimodal-hackathon-2024/ZTFBTS/",
    ]

    # Get the first valid directory
    data_dir = get_valid_dir(data_dirs)
    # Check if the config file has a spectra key
    if "spectral" == cfg["data"]:
        data_dirs = ["ZTFBTS_spectra/"]
        spectra_dir = get_valid_dir(data_dirs)
    else:
        spectra_dir = None


    max_data_len = 1000  # Spectral data is cut to this length
    dataset, nband = load_data(data_dir, spectra_dir, max_data_len)

    number_of_samples = len(dataset)

    val_fraction = 0.05
    n_samples_val = int(val_fraction * number_of_samples)

    dataset_train, dataset_val = random_split(
        dataset, [number_of_samples - n_samples_val, n_samples_val]
    )

    wandb.agent(sweep_id=sweep_id, function=train_sweep)
