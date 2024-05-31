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
    SimulationDataset,
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
            dataset, [number_of_samples - n_samples_val, n_samples_val], generator=torch.Generator().manual_seed(cfg.seed)
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

        train_loader = DataLoader(
            dataset_train,
            batch_size=cfg.batchsize,
            num_workers=num_workers,
            pin_memory=True,
            shuffle=True,
        )
        val_loader = DataLoader(
            dataset_val,
            batch_size=cfg.batchsize,
            num_workers=num_workers,
            pin_memory=True,
            shuffle=False,
        )


        transformer_kwargs = {
            "n_out": cfg.n_out,
            "emb": cfg.emb,
            "heads": cfg.heads,
            "depth": cfg.transformer_depth,
            "dropout": cfg.dropout,
            "time_norm": cfg.time_norm,
            "agg": cfg.agg,
        }
        transformer_spectral_kwargs = {
            "n_out": cfg.n_out,
            "emb": cfg.emb_spectral,
            "heads": cfg.heads_spectral,
            "depth": cfg.transformer_depth_spectral,
            "dropout": cfg.dropout,
            "time_norm": cfg.time_norm_spectral,
            "agg": cfg.agg_spectral,
        }
        conv_kwargs = {
            "dim": cfg.cnn_dim, 
            "depth": cfg.cnn_depth,
            "channels": cfg.cnn_channels,
            "kernel_size": cfg.cnn_kernel_size, 
            "patch_size": cfg.cnn_patch_size, 
            "n_out": cfg.n_out,
            "dropout_prob": cfg.dropout,
        }
        optimizer_kwargs = {"weight_decay": cfg.weight_decay}

        clip_model = LightCurveImageCLIP(
            logit_scale=cfg.logit_scale,
            lr=cfg.lr,
            nband=2,
            loss="softmax",
            transformer_kwargs=transformer_kwargs,
            transformer_spectral_kwargs=transformer_spectral_kwargs,
            conv_kwargs=conv_kwargs,
            optimizer_kwargs=optimizer_kwargs,
            combinations=combinations,
            regression=False,
            classification=False,
        )

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
        if len(combinations) == 2:
            wandb.define_metric("AUC_val", summary="max")

        trainer.fit(
            model=clip_model, train_dataloaders=train_loader, val_dataloaders=val_loader
        )

        
        wandb.run.summary["best_auc"] = np.max(
            loss_tracking_callback.auc_val_history
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
        embs_train = get_embs(clip_model, train_loader, combinations)
        embs_val = get_embs(clip_model, val_loader, combinations)

        plot_ROC_curves(
            embs_train,
            embs_val,
            combinations,
            path_base=path_run,
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
        "data/sim_data/",
    ]

    # Get the first valid directory
    data_dir = get_valid_dir(data_dirs)

    # Get what data combinations are used
    combinations = cfg["extra_args"]["combinations"]

    max_spectral_data_len = cfg["extra_args"][
        "max_spectral_data_len"
    ]  # Spectral data is cut to this length
    dataset = SimulationDataset(
        data_dir + 'ZTF_Pretrain_5Class.hdf5',
        bands=['r', 'g'], 
        n_max_obs_spec=max_spectral_data_len,
        combinations=combinations, 
    )

    wandb.agent(
        sweep_id=sweep_id,
        entity=cfg["entity"],
        project=cfg["project"],
        function=train_sweep,
        count=cfg["extra_args"]["nruns"],
    )
