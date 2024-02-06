from ruamel.yaml import YAML
import wandb
import os
from typing import Tuple


def schedule_sweep(config: str, analysis_path: str) -> Tuple[str, str]:
    """
    Schedules a sweep with Weights & Biases (wandb) and saves the sweep configuration
    to a specified analysis path.

    Args:
        config: Path to the YAML configuration file for the sweep.
        analysis_path: Directory path where the sweep's configuration and model path will be saved.

    Returns:
        A tuple containing the sweep ID and the model path directory.
    """
    print("config file : ", config, flush=True)
    yaml = YAML(typ="rt")
    cfg = yaml.load(open(f"{config}"))

    sweep_id = wandb.sweep(sweep=YAML(typ="safe").load(open(f"{config}")))
    print("Schedule sweep with id : ", sweep_id, flush=True)
    cfg["sweep"] = {"id": sweep_id}

    model_path = f"{analysis_path}/{sweep_id}/"
    config_path = f"{model_path}/sweep_config.yaml"

    os.makedirs(model_path, exist_ok=True)
    with open(config_path, "w") as outfile:
        yaml.dump(cfg, outfile)

    print(f"config path saved at:\n{config_path}\n", flush=True)

    return sweep_id, model_path
