import os
from typing import List
from pytorch_lightning.callbacks import Callback
import torch
import numpy as np
from torch.utils.data import DataLoader
from typing import Tuple, List
from matplotlib import pyplot as plt
from ruamel.yaml import YAML


def get_savedir(args) -> str:
    """
    Return config dict and path to save new plots and models based on
    whether to continue from checkpoint or not

    Args:
    args: argparse.ArgumentParser object

    Returns:
    str: path to save new plots and models
    cfg: dict: configuration dictionary
    """

    if args.ckpt_path:
        cfg = YAML(typ="safe").load(
            open(os.path.join(os.path.dirname(args.ckpt_path), "config.yaml"))
        )
        save_dir = os.path.join(os.path.dirname(args.ckpt_path), "resume/")
        os.makedirs(save_dir, exist_ok=True)
    else:
        yaml = YAML(typ="rt")
        cfg = yaml.load(open(args.config_path))
        if args.runname:
            save_dir = f"./analysis/runs/{args.runname}/"
        else:
            dirlist = [
                int(item)
                for item in os.listdir("./analysis/runs/")
                if os.path.isdir(os.path.join("./analysis/runs/", item))
                and item.isnumeric()
            ]
            dirname = str(max(dirlist) + 1) if len(dirlist) > 0 else "0"
            save_dir = os.path.join("./analysis/runs/", dirname)

        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "config.yaml"), "w") as outfile:
            yaml.dump(cfg, outfile)

    return save_dir, cfg


def set_seed(seed: int = 0) -> None:
    """
    set seed so that results are fully reproducible
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed: {seed}")


def get_valid_dir(data_dirs: List[str]) -> str:
    """
    Returns the first valid directory in the list of directories.

    Args:
    data_dirs (List[str]): A list of directory paths to check.

    Returns:
    str: The first valid directory path found in the list.

    Raises:
    ValueError: If no valid directory is found in the list.
    """
    for data_dir in data_dirs:
        if os.path.isdir(data_dir):
            return data_dir
    raise ValueError("No valid data directory found")


class LossTrackingCallback(Callback):
    def __init__(self):
        self.train_loss_history = []
        self.val_loss_history = []
        self.epoch_train_loss = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Accumulate training loss for each batch
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs
        self.epoch_train_loss.append(loss.detach().item())

    def on_train_epoch_end(self, trainer, pl_module):
        # Append average training loss after each epoch
        epoch_loss = sum(self.epoch_train_loss) / len(self.epoch_train_loss)
        self.train_loss_history.append(epoch_loss)
        # Reset the list for the next epoch
        self.epoch_train_loss = []

    def on_validation_epoch_end(self, trainer, pl_module):
        # Append validation loss after each validation epoch
        val_loss = trainer.callback_metrics.get("val_loss")
        if val_loss is not None:
            self.val_loss_history.append(val_loss.detach().item())


def plot_loss_history(train_loss_history, val_loss_history, path_base="./"):
    """
    Plots the training and validation loss histories.

    Args:
    train_loss_history (list): A list of training loss values.
    val_loss_history (list): A list of validation loss values.
    """
    # Create a figure and a set of subplots
    plt.figure(figsize=(10, 6))

    # Plot training loss
    plt.plot(
        train_loss_history,
        label="Training Loss",
        color="blue",
        linestyle="-",
        marker="o",
    )

    # Plot validation loss
    plt.plot(
        val_loss_history,
        label="Validation Loss",
        color="red",
        linestyle="-",
        marker="x",
    )

    # Adding title and labels
    plt.title("Training and Validation Loss Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")

    # Adding a legend
    plt.legend()

    # Show grid
    plt.grid(True)

    # Show the plot
    plt.savefig(os.path.join(path_base, "loss_history.png"))


def cosine_similarity(a, b, temperature=1):
    a_norm = a / a.norm(dim=-1, keepdim=True)
    b_norm = b / b.norm(dim=-1, keepdim=True)

    logits = a_norm @ b_norm.T * temperature
    return logits.squeeze()


def get_embs(
    clip_model: torch.nn.Module, dataloader: DataLoader
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes and concatenates embeddings for light curves and images from a DataLoader.

    Args:
    clip_model (torch.nn.Module): The model used for generating embeddings.
    dataloader (DataLoader): DataLoader providing batches of data (images, magnitudes, times, masks).

    Returns:
    Tuple[torch.Tensor, torch.Tensor]: Tuple of two tensors containing concatenated embeddings
    for light curves and images, respectively.
    """

    clip_model.eval()

    embs_curves = []
    embs_images = []

    # Iterate through the DataLoader
    for batch in dataloader:
        img, mag, time, mask, _ = batch

        # Compute embeddings and detach from the computation graph
        emb_host = clip_model.lightcurve_embeddings_with_projection(
            mag.detach(), time.detach(), mask.detach()
        ).detach()
        emb_src = clip_model.image_embeddings_with_projection(img.detach()).detach()

        # Append the results to the lists
        embs_curves.append(emb_host)
        embs_images.append(emb_src)

    # Concatenate all embeddings into single tensors
    embs_curves = torch.cat(embs_curves, dim=0)
    embs_images = torch.cat(embs_images, dim=0)
    return embs_curves, embs_images


def get_ROC_data(
    embs_curves: torch.Tensor, embs_images: torch.Tensor
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate ROC-like data by evaluating the cosine similarity between two sets of embeddings.

    Args:
    embs_curves (torch.Tensor): Tensor of embeddings for light curves.
    embs_images (torch.Tensor): Tensor of embeddings for images.

    Returns:
    Tuple[np.ndarray, np.ndarray]: A tuple containing an array of thresholds and an array of the fraction of correct predictions at each threshold.
    """
    thresholds = np.linspace(0, 1, 100)
    imgs = []

    # Iterate through image embeddings and calculate cosine similarity with curve embeddings
    for idx, emb_src in enumerate(embs_images):
        cos_sim = cosine_similarity(embs_curves, emb_src)
        idx_sorted = torch.argsort(cos_sim, descending=True)

        # Calculate the number of correct predictions for each threshold
        num_right = [
            idx in idx_sorted[: int(threshold * len(idx_sorted))]
            for threshold in thresholds
        ]
        imgs.append(num_right)

    # Calculate the fraction of correct predictions at each threshold
    fraction_correct = np.sum(imgs, axis=0) / len(embs_images)

    return thresholds, fraction_correct


def get_AUC(
    embs_curves: torch.Tensor,
    embs_images: torch.Tensor,
) -> Tuple[float, float]:
    """
    Calculate the area under the ROC curve for training and validation datasets.
    Args:
    embs_curves (torch.Tensor): Embeddings for light curves in the training set.
    embs_images (torch.Tensor): Embeddings for images in the training set.
    """
    thresholds, fraction_correct = get_ROC_data(embs_curves, embs_images)
    auc = np.trapz(fraction_correct, thresholds)
    return auc


def plot_ROC_curves(
    embs_curves_train: torch.Tensor,
    embs_images_train: torch.Tensor,
    embs_curves_val: torch.Tensor,
    embs_images_val: torch.Tensor,
    path_base: str = "./",
) -> None:
    """
    Plots ROC-like curves for training and validation datasets based on embeddings.

    Args:
    embs_curves_train (torch.Tensor): Embeddings for light curves in the training set.
    embs_images_train (torch.Tensor): Embeddings for images in the training set.
    embs_curves_val (torch.Tensor): Embeddings for light curves in the validation set.
    embs_images_val (torch.Tensor): Embeddings for images in the validation set.
    path_base (str) : path to save the plot
    """
    thresholds, fraction_correct_train = get_ROC_data(
        embs_curves_train, embs_images_train
    )
    thresholds, fraction_correct_val = get_ROC_data(embs_curves_val, embs_images_val)

    # Set overall figure size and title
    plt.figure(figsize=(12, 6))
    plt.suptitle("Fraction of Correct Predictions vs. Threshold")

    # Plot for validation data
    plt.subplot(1, 2, 1)
    plt.plot(thresholds, fraction_correct_val, color="blue", lw=2, label="Validation")
    plt.plot(thresholds, thresholds, linestyle="--", color="gray", label="Random")
    plt.title("Validation Data")
    plt.xlabel("Threshold")
    plt.ylabel("Fraction Correct")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)

    # Plot for training data
    plt.subplot(1, 2, 2)
    plt.plot(thresholds, fraction_correct_train, color="green", lw=2, label="Training")
    plt.plot(thresholds, thresholds, linestyle="--", color="gray", label="Random")
    plt.title("Training Data")
    plt.xlabel("Threshold")
    plt.ylabel("Fraction Correct")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)

    # Adjust layout to prevent overlapping
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(path_base, "ROC_curves.png"))
