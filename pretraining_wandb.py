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
    load_data,
    NoisyDataLoader,
    SimulationLightcurveDataset,
)

from src.models_pretraining import (
    MaskedLightCurveEncoder,
    plot_masked_pretraining_model,
)

from src.wandb_utils import continue_sweep, schedule_sweep


def train_sweep(config=None):
    with wandb.init(config=config) as run:
        print(f"run name: {run.name}", flush=True)
        path_run = os.path.join(model_path, run.name)
        os.makedirs(path_run, exist_ok=True)

        cfg = wandb.config
        set_seed(cfg.seed)

        # For the moment hardcoded to a single band
        bands = [ "r","g"]
        nband = len(bands)
        n_max_obs = 80
        mask_ratio = 0.15

        dataset = SimulationLightcurveDataset(
            "./sim_data/scotch_z3.hdf5",
            transient_types=None,
            bands=bands,
            n_max_obs=n_max_obs,
        )

        number_of_samples = len(dataset)
        n_samples_val = int(val_fraction * number_of_samples)

        # dump config
        config_dict = {k: v for k, v in cfg.items()}
        with open(os.path.join(path_run, "config.yaml"), "w") as f:
            YAML().dump(config_dict, f)

        # Default to 1 if the environment variable is not set
        cpus_per_task = int(os.getenv("SLURM_CPUS_PER_TASK", 1))

        # Assuming you want to leave one CPU for overhead
        num_workers = max(1, cpus_per_task - 1)
        print(f"Using {num_workers} workers for data loading", flush=True)

        dataset_train, dataset_val = random_split(
            dataset, [len(dataset) - n_samples_val, n_samples_val]
        )
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
            "n_out": n_max_obs,
            "emb": cfg.emb,
            "heads": cfg.heads,
            "depth": cfg.transformer_depth,
            "dropout": cfg.dropout,
            "time_norm": cfg.time_norm,
            "agg": cfg.agg,
        }

        optimizer_kwargs = {"weight_decay": cfg.weight_decay}

        model = MaskedLightCurveEncoder(
            f_mask=mask_ratio,
            lr=cfg.lr,
            transformer_kwargs=transformer_kwargs,
            optimizer_kwargs=optimizer_kwargs,
            nband = nband,
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
        print("made it to fit")
        trainer.fit(
            model=model, train_dataloaders=train_loader, val_dataloaders=val_loader
        )

        plot_loss_history(
            loss_tracking_callback.train_loss_history,
            loss_tracking_callback.val_loss_history,
            path_base=path_run,
        )

        plot_masked_pretraining_model(
            model, val_loader, path_run + "/masked_pretraining.png"
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

    wandb.agent(
        sweep_id=sweep_id,
        entity=cfg["entity"],
        project=cfg["project"],
        function=train_sweep,
        count=cfg["extra_args"]["nruns"],
    )
