import os
from typing import List
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback
import torch
import numpy as np
from torch.utils.data import DataLoader
from typing import Tuple, List
from matplotlib import pyplot as plt
from ruamel.yaml import YAML


def filter_files(filenames_avail, filenames_to_filter, data_to_filter=None):
    """
    Function to filter filenames and data based on the filenames_avail

    Args:
    filenames_avail (list): List of filenames available
    filenames_to_filter (list): List of filenames to filter
    data_to_filter (List[np.ndarray]): Data to filter based on filenames_to_filter

    Returns:
    inds_filt (np.ndarray): Indices of filtered filenames in filenames_to_filter
    filenames_to_filter (list): List of filtered filenames
    data_to_filter (np.ndarray): Filtered data
    """
    # Check which each filenames_to_filter are available in filenames_avail
    inds_filt = np.isin(filenames_to_filter, filenames_avail)
    if data_to_filter:
        for i in range(len(data_to_filter)):
            data_to_filter[i] = data_to_filter[i][inds_filt]

    filenames_to_filter = np.array(filenames_to_filter)[inds_filt]

    return inds_filt, filenames_to_filter, data_to_filter


def find_indices_in_arrays(st1, st2):
    """
    Find indices of where elements of st1 appear in st2 and indices in st1 of those elements.

    Parameters:
    - st1 (list or array): The list of strings to find in st2.
    - st2 (list or array): The list of strings to search within.

    Returns:
    - tuple of two lists:
        - The first list contains indices indicating where each element of st1 is found in st2.
        - The second list contains the indices in st1 for elements that were found in st2.
    """
    indices_in_st2 = []
    indices_in_st1 = []
    for idx, item in enumerate(st1):
        try:
            index_in_st2 = st2.index(item)  # Find the index of item in st2
            indices_in_st2.append(index_in_st2)
            indices_in_st1.append(idx)
        except ValueError:
            # Item not found in st2, optionally handle it
            continue  # Simply skip if not found
    return indices_in_st2, indices_in_st1


def get_savedir(args) -> str:
    """
    Return config dict and path to save new plots and models based on
    whether to continue from checkpoint or not; dump config file in savedir path

    Args:
    args: argparse.ArgumentParser object

    Returns:
    str: path to save new plots and models
    cfg: dict: configuration dictionary
    """
    # Create directory to save new plots and checkpoints
    import os

    if not os.path.exists("analysis"):
        os.makedirs("analysis")
        os.makedirs("analysis/runs")
    if not os.path.exists("analysis/runs"):
        os.makedirs("analysis/runs")

    # save in checkpoint directory if resuming from checkpoint
    # else save in numbered directory if not given runname
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
        self.auc_val_history = []

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

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        try: 
            auc_val = trainer.callback_metrics.get("AUC_val")
            if auc_val is None:
                auc_val = (
                    sum(
                        [
                            trainer.callback_metrics.get(f"AUC_val{i}").detach().item()
                            for i in range(1, 4)
                        ]
                    )
                    / 3
                )
            else:
                auc_val = auc_val.detach().item()
            self.auc_val_history.append(auc_val)
        except:
            pass


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
    """
    Compute cosine similarity between two tensors.

    Args:
    a (torch.Tensor): First tensor.
    b (torch.Tensor): Second tensor.
    temperature (float): Temperature parameter for scaling the cosine similarity; default is 1.

    Returns:
    torch.Tensor: Cosine similarity between the two tensors.
    """
    a_norm = a / a.norm(dim=-1, keepdim=True)
    b_norm = b / b.norm(dim=-1, keepdim=True)

    logits = a_norm @ b_norm.T * temperature
    return logits.squeeze()


def get_embs(
    clip_model: torch.nn.Module, dataloader: DataLoader, combinations: List[str]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes and concatenates embeddings for light curves and images from a DataLoader.

    Args:
    clip_model (torch.nn.Module): The model used for generating embeddings.
    dataloader (DataLoader): DataLoader providing batches of data (images, magnitudes, times, masks).
    combinations (List[str]): List of combinations of modalities to use for embeddings.

    Returns:
    List[torch.Tensor]: List of concatenated embeddings for each item in combinations.
    """

    clip_model.eval()
    # getting device of model
    device = next(clip_model.parameters()).device

    embs_list = [[] for i in range(len(combinations))]

    # Iterate through the DataLoader
    for batch in dataloader:
        x_img, x_lc, t_lc, mask_lc, x_sp, t_sp, mask_sp,_ = batch
        x_img, x_lc, t_lc, mask_lc, x_sp, t_sp, mask_sp = (
            x_img.to(device),
            x_lc.to(device),
            t_lc.to(device),
            mask_lc.to(device),
            x_sp.to(device),
            t_sp.to(device),
            mask_sp.to(device),
        )

        # Compute embeddings and detach from the computation graph
        with torch.no_grad():
            x = []
            if "host_galaxy" in combinations:
                x.append(clip_model.image_embeddings_with_projection(x_img))
            if "lightcurve" in combinations:
                x.append(
                    clip_model.lightcurve_embeddings_with_projection(
                        x_lc, t_lc, mask_lc
                    )
                )
            if "spectral" in combinations:
                x.append(
                    clip_model.spectral_embeddings_with_projection(x_sp, t_sp, mask_sp)
                )

        # Append the results to the lists
        for i in range(len(x)):
            embs_list[i].append(x[i])

    # Concatenate all embeddings into single tensors
    for i in range(len(embs_list)):
        embs_list[i] = torch.cat(embs_list[i], dim=0)
    return embs_list


def get_ROC_data(
    embs1: torch.Tensor, embs2: torch.Tensor
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate ROC-like data by evaluating the cosine similarity between two sets of embeddings.

    Args:
    embs1 (torch.Tensor): Tensor of first set of embeddings.
    embs2 (torch.Tensor): Tensor of second set of embeddings.

    Returns:
    Tuple[np.ndarray, np.ndarray]: A tuple containing an array of thresholds and an array of the fraction of correct predictions at each threshold.
    """
    thresholds = np.linspace(0, 1, 100)
    imgs = []

    # Iterate through image embeddings and calculate cosine similarity with curve embeddings
    for idx, emb_src in enumerate(embs2):
        cos_sim = cosine_similarity(embs1, emb_src)
        idx_sorted = torch.argsort(cos_sim, descending=True)

        # Calculate the number of correct predictions for each threshold
        num_right = [
            idx in idx_sorted[: int(threshold * len(idx_sorted))]
            for threshold in thresholds
        ]
        imgs.append(num_right)

    # Calculate the fraction of correct predictions at each threshold
    fraction_correct = np.sum(imgs, axis=0) / len(embs2)

    return thresholds, fraction_correct


def get_AUC(
    embs1: torch.Tensor,
    embs2: torch.Tensor,
) -> Tuple[float, float]:
    """
    Calculate the area under the ROC curve for training and validation datasets.
    Args:
    embs1 (torch.Tensor): Embeddings for first modality.
    embs2 (torch.Tensor): Embeddings for second modality.
    """
    thresholds, fraction_correct = get_ROC_data(embs1, embs2)
    auc = np.trapz(fraction_correct, thresholds)
    return auc


def plot_ROC_curves(
    embs_train: List[torch.Tensor],
    embs_val: List[torch.Tensor],
    combinations: List[str],
    path_base: str = "./",
) -> None:
    """
    Plots ROC-like curves for training and validation datasets based on embeddings.

    Args:
    embs_train (List[torch.Tensor]): List of embeddings for training data.
    embs_val (List[torch.Tensor]): List of embeddings for validation data.
    combinations (List[str]): List of combinations of modalities to use for embeddings.
    path_base (str) : path to save the plot
    """

    combinations = sorted(combinations)

    fractions_train, fractions_val, labels = [], [], []
    for i in range(len(embs_train) - 1):
        for j in range(i + 1, len(embs_train)):
            thresholds, fraction_correct_train = get_ROC_data(
                embs_train[i], embs_train[j]
            )
            thresholds, fraction_correct_val = get_ROC_data(embs_val[i], embs_val[j])
            fractions_train.append(fraction_correct_train)
            fractions_val.append(fraction_correct_val)
            labels.append(f"{combinations[i]} and {combinations[j]}")

    # Set overall figure size and title
    plt.figure(figsize=(12, 6))
    plt.suptitle("Fraction of Correct Predictions vs. Threshold")

    # Plot for validation data
    plt.subplot(1, 2, 1)
    for i, f_val in enumerate(fractions_val):
        plt.plot(thresholds, f_val, lw=2, label=labels[i])
    plt.plot(thresholds, thresholds, linestyle="--", color="gray", label="Random")
    plt.title("Validation Data")
    plt.xlabel("Threshold")
    plt.ylabel("Fraction Correct")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)

    # Plot for training data
    plt.subplot(1, 2, 2)
    for i, f_train in enumerate(fractions_train):
        plt.plot(thresholds, f_train, lw=2, label=labels[i])
    plt.plot(thresholds, thresholds, linestyle="--", color="gray", label="Random")
    plt.title("Training Data")
    plt.xlabel("Threshold")
    plt.ylabel("Fraction Correct")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)

    # Adjust layout to prevent overlapping
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(path_base, "ROC_curves.png"))
