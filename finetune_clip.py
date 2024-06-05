import os, sys
import wandb
from ruamel.yaml import YAML
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import numpy as np

import torch
import pytorch_lightning as pl

from torch.utils.data import TensorDataset, DataLoader, random_split

from src.models_multimodal import LightCurveImageCLIP, load_pretrain_clip_model, initialize_model, ClipMLP
from src.utils import (
    set_seed,
    get_valid_dir,
    LossTrackingCallback,
    plot_ROC_curves,
    plot_loss_history,
    get_embs,
)
from src.dataloader import (
    load_data,
    NoisyDataLoader,
)
from src.wandb_utils import continue_sweep, schedule_sweep


def train_sweep(config=None):
    with wandb.init(config=config) as run:
        print(f"run name: {run.name}", flush=True)
        path_run = os.path.join(model_path, run.name)
        os.makedirs(path_run, exist_ok=True)

        cfg = wandb.config
        set_seed(cfg.seed)

        number_of_samples = len(dataset)

        n_samples_val = int(val_fraction * number_of_samples)

        dataset_train, dataset_val = random_split(
            dataset,
            [number_of_samples - n_samples_val, n_samples_val],
            generator=torch.Generator().manual_seed(cfg.seed),
        )

        # dump config
        config_dict = {k: v for k, v in cfg.items()}
        with open(os.path.join(path_run, "config.yaml"), "w") as f:
            YAML().dump(config_dict, f)

        # Default to 1 if the environment variable is not set
        cpus_per_task = int(os.getenv("SLURM_CPUS_PER_TASK", 1))

        # Assuming you want to leave one CPU for overhead
        num_workers = max(1, cpus_per_task - 1)
        print(f"Using {num_workers} workers for data loading", flush=True)

        train_loader_no_aug = NoisyDataLoader(
            dataset_train,
            batch_size=cfg.batchsize,
            noise_level_img=0,
            noise_level_mag=0,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            combinations=combinations,
        )
        val_loader_no_aug = NoisyDataLoader(
            dataset_val,
            batch_size=cfg.batchsize,
            noise_level_img=0,
            noise_level_mag=0,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            combinations=combinations,
        )

        # Create custom noisy data loaders
        train_loader = NoisyDataLoader(
            dataset_train,
            batch_size=cfg.batchsize,
            noise_level_img=1,
            noise_level_mag=1,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            combinations=combinations,
        )
        val_loader = NoisyDataLoader(
            dataset_val,
            batch_size=cfg.batchsize,
            noise_level_img=0,
            noise_level_mag=0,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            combinations=combinations,
        )

        optimizer_kwargs = {"weight_decay": cfg.weight_decay}
        mlp_kwargs = {"hidden_dim": cfg.hidden_dim, 
                      "output_dim": 1,
                      "num_layers": cfg.num_layers,
                      "dropout": cfg.dropout}
        
        clip_model,_,_,_,_  = initialize_model(pretrain_path, 
                                                optimizer_kwargs=optimizer_kwargs, 
                                                combinations=combinations)

        # Loading up pretrained models
        load_pretrain_clip_model(pretrain_path, clip_model, freeze_backbone)

        model = ClipMLP(clip_model, 
                        mlp_kwargs, 
                        optimizer_kwargs,
                        cfg.lr, 
                        combinations, 
                        regression=regression, 
                        classification=classification, 
                        n_classes=n_classes)

        # Custom call back for tracking loss
        loss_tracking_callback = LossTrackingCallback()

        device = "gpu" if torch.cuda.is_available() else "cpu"
        if device == "gpu":  # Set float32 matmul precision for A100 GPUs
            cuda_name = torch.cuda.get_device_name(torch.cuda.current_device())
            if cuda_name.startswith("NVIDIA A100-SXM4"):
                torch.set_float32_matmul_precision("high")

        wandb_logger = WandbLogger()
        checkpoint_callback = ModelCheckpoint(
            dirpath=path_run, save_top_k=2, monitor="val_loss"
        )
        early_stop_callback = EarlyStopping(
            monitor="val_loss",
            min_delta=0.00,
            patience=cfg.patience,
            verbose=False,
            mode="min",
        )

        trainer = pl.Trainer(
            max_epochs=cfg.epochs,
            accelerator=device,
            callbacks=[
                loss_tracking_callback,
                checkpoint_callback,
                early_stop_callback,
            ],
            logger=wandb_logger,
            enable_progress_bar=False,
        )
        
        trainer.fit(
            model=model, train_dataloaders=train_loader, val_dataloaders=val_loader
        )

        wandb.finish()


if __name__ == "__main__":
    wandb.login()

    arg = sys.argv[1]
    analysis_path = "./analysis/"

    if arg.endswith(".yaml"):
        config = arg
        sweep_id, model_path, cfg = schedule_sweep(config, analysis_path)
    else:
        sweep_id = os.path.basename(arg)
        model_path = os.path.join(analysis_path, sweep_id)
        cfg = continue_sweep(model_path)

    print("model path: " + model_path, flush=True)

    set_seed(0)
    # define constants
    val_fraction = cfg["extra_args"]["val_fraction"]

    # Data preprocessing

    data_dirs = [
        "/home/thelfer1/scr4_tedwar42/thelfer1/ZTFBTS/",
        "ZTFBTS/",
        "data/ZTFBTS/",
        "/ocean/projects/phy230064p/shared/ZTFBTS/",
        "/n/home02/gemzhang/repos/Multimodal-hackathon-2024/data/ZTFBTS/",
    ]

    # Get the first valid directory
    data_dir = get_valid_dir(data_dirs)

    # Get what data combinations are used
    combinations = cfg["extra_args"]["combinations"]
    regression = cfg["extra_args"]["regression"]
    classification = cfg["extra_args"]["classification"]

    if classification:
        n_classes = cfg["extra_args"]["n_classes"]
    else:
        n_classes = 5

    pretrain_path = cfg["extra_args"].get("pretrain_path")
    freeze_backbone = cfg["extra_args"].get("freeze_backbone")

    # Check if the config file has a spectra key
    if "spectral" in combinations:
        data_dirs = ["ZTFBTS_spectra/", "data/ZTFBTS_spectra/"]
        spectra_dir = get_valid_dir(data_dirs)
    else:
        spectra_dir = None

    max_spectral_data_len = cfg["extra_args"][
        "max_spectral_data_len"
    ]  # Spectral data is cut to this length
    dataset, nband, _ = load_data(
        data_dir,
        spectra_dir,
        max_data_len_spec=max_spectral_data_len,
        combinations=combinations,
        n_classes=n_classes,
        spectral_rescalefactor=1,
    )

    wandb.agent(
        sweep_id=sweep_id,
        entity=cfg["entity"],
        project=cfg["project"],
        function=train_sweep,
        count=cfg["extra_args"]["nruns"],
    )
