import os
import numpy as np
import torch
from PIL import Image
from typing import List
from einops import rearrange
from tqdm import tqdm
import pandas as pd
from typing import Tuple
from torchvision.transforms import RandomRotation
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


# Custom data loader with noise augmentation using magerr
class NoisyDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, noise_level_img, noise_level_mag, shuffle=True, **kwargs):
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)
        self.max_noise_intensity = noise_level_img
        self.noise_level_mag = noise_level_mag

    def __iter__(self):
        for batch in super().__iter__():
            # Add random noise to images and time-magnitude tensors
            host_imgs, mag, time, mask, magerr = batch

            # Calculate the range for the random noise based on the max_noise_intensity
            noise_range = self.max_noise_intensity * torch.std(host_imgs)

            # Generate random noise within the specified range
            noisy_imgs = host_imgs + (2 * torch.rand_like(host_imgs) - 1) * noise_range

            # Add Gaussian noise to mag using magerr
            noisy_mag = mag + torch.randn_like(mag) * magerr * self.noise_level_mag

            # Randomly apply rotation by multiples of 90 degrees
            rotation_angle = torch.randint(0, 4, (noisy_imgs.size(0),)) * 90
            rotated_imgs = []

            # Apply rotation to each image
            for i in range(noisy_imgs.size(0)):
                rotated_img = RandomRotation([rotation_angle[i], rotation_angle[i]])(noisy_imgs[i])
                rotated_imgs.append(rotated_img)

            # Stack the rotated images back into a tensor
            rotated_imgs = torch.stack(rotated_imgs)
            
            # Return the noisy batch
            yield noisy_imgs, noisy_mag, time, mask
            
            

def load_images(data_dir: str) -> torch.Tensor:
    """
    Load and preprocess images from a specified directory.

    Args:
    data_dir (str): The directory path where images are stored.

    Returns:
    torch.Tensor: A tensor containing the preprocessed images.
    """
    print("Loading images...")

    dir_host_imgs = f"{data_dir}/hostImgs/"
    host_imgs = []

    # Iterate through the directory and load images
    for filename in tqdm(os.listdir(dir_host_imgs)):
        file_path = os.path.join(dir_host_imgs, filename)
        if file_path.endswith(".png"):
            # Load image, convert to RGB, and then to a NumPy array
            host_img = Image.open(file_path).convert("RGB")
            host_img = np.asarray(host_img)
            host_imgs.append(host_img)

    # Convert the list of images to a NumPy array
    host_imgs = np.array(host_imgs)

    # Convert the NumPy array to a PyTorch tensor and rearrange dimensions
    host_imgs = torch.from_numpy(host_imgs).float()
    host_imgs = rearrange(host_imgs, "b h w c -> b c h w")

    # Normalize the images
    host_imgs /= 255.0

    return host_imgs


def load_lightcurves(
    data_dir: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Load light curves from CSV files in the specified directory.

    Args:
    data_dir (str): Directory path containing light curve CSV files.

    Returns:
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]: A tuple containing:
        - time_ary: Numpy array of time observations.
        - mag_ary: Numpy array of magnitude observations.
        - magerr_ary: Numpy array of magnitude error observations.
        - mask_ary: Numpy array indicating the presence of an observation.
        - nband: Number of observation bands.
    """

    print("Loading light curves...")
    dir_light_curves = f"{data_dir}/light-curves/"

    def open_light_curve_csv(filename: str) -> pd.DataFrame:
        """Helper function to open a light curve CSV file."""
        file_path = os.path.join(dir_light_curves, filename)
        return pd.read_csv(file_path)

    bands = ["R", "g"]
    nband = len(bands)
    n_max_obs = 100
    lightcurve_files = os.listdir(dir_light_curves)

    mask_list, mag_list, magerr_list, time_list = [], [], [], []

    for filename in tqdm(lightcurve_files):
        if filename.endswith(".csv"):
            light_curve_df = open_light_curve_csv(filename)

            if not all(
                col in light_curve_df.columns
                for col in ["time", "mag", "magerr", "band"]
            ):
                continue

            time_concat, mag_concat, magerr_concat, mask_concat = [], [], [], []
            for band in bands:
                df_band = light_curve_df[light_curve_df["band"] == band]

                if len(df_band["mag"]) > n_max_obs:
                    # Sample n_max_obs observations randomly
                    indices = np.random.choice(len(df_band["mag"]), n_max_obs)
                    mask = np.ones(n_max_obs, dtype=bool)
                else:
                    # Pad the arrays with zeros and create a mask
                    indices = np.arange(len(df_band["mag"]))
                    mask = np.zeros(n_max_obs, dtype=bool)
                    mask[: len(indices)] = True

                time = np.pad(
                    df_band["time"].iloc[indices],
                    (0, n_max_obs - len(indices)),
                    "constant",
                )
                mag = np.pad(
                    df_band["mag"].iloc[indices],
                    (0, n_max_obs - len(indices)),
                    "constant",
                )
                magerr = np.pad(
                    df_band["magerr"].iloc[indices],
                    (0, n_max_obs - len(indices)),
                    "constant",
                )

                time_concat += list(time)
                mag_concat += list(mag)
                magerr_concat += list(magerr)
                mask_concat += list(mask)

            mask_list.append(mask_concat)
            time_list.append(time_concat)
            mag_list.append(mag_concat)
            magerr_list.append(magerr_concat)

    time_ary = np.array(time_list)
    mag_ary = np.array(mag_list)
    magerr_ary = np.array(magerr_list)
    mask_ary = np.array(mask_list)

    return time_ary, mag_ary, magerr_ary, mask_ary, nband



def plot_lightcurve_and_images(
    host_imgs: torch.Tensor,
    time_ary: np.ndarray,
    mag_ary: np.ndarray,
    magerr_ary: np.ndarray,
    mask_ary: np.ndarray,
    nband: int,
    path_base : str = './', 
) -> None:
    """
    Plots host images and corresponding light curves.

    Args:
    host_imgs (torch.Tensor): A tensor containing host images.
    time_ary (np.ndarray): Numpy array of time observations for light curves.
    mag_ary (np.ndarray): Numpy array of magnitude observations for light curves.
    magerr_ary (np.ndarray): Numpy array of magnitude error observations for light curves.
    mask_ary (np.ndarray): Numpy array indicating the presence of an observation.
    nband (int): Number of observation bands.
    """

    # Plot host images and light curves in a grid layout
    n_rows = 5
    n_cols = 2
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(10, 20))
    l = len(time_ary[0]) // nband

    for i in range(n_rows):
        axs[i, 0].imshow(host_imgs[i].permute(1, 2, 0))
        axs[i, 0].set_title("Host Image")
        for j in range(nband):
            t, m, mag, e = (
                time_ary[i][j * l : (j + 1) * l],
                mask_ary[i][j * l : (j + 1) * l],
                mag_ary[i][j * l : (j + 1) * l],
                magerr_ary[i][j * l : (j + 1) * l],
            )
            axs[i, 1].errorbar(t[m], mag[m], yerr=e[m], fmt="o")
        axs[i, 1].set_title("Light Curve")

    # Save the first plot as a separate file
    plt.savefig(os.path.join(path_base, "lightcurves_and_host_images.png")) 

    # Plot banner images
    colors = ["firebrick", "dodgerblue"]
    n_pairs_per_row = 3
    n_cols = n_pairs_per_row * 2  # Two columns per pair

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(50, 30))

    for i in range(n_rows):
        for j in range(n_pairs_per_row):
            index = i * n_pairs_per_row + j

            img_col = j * 2  # Column index for image
            axs[i, img_col].imshow(host_imgs[index].permute(1, 2, 0))
            axs[i, img_col].axis("off")  # Turn off axis for images

            lc_col = img_col + 1  # Column index for light curve
            for nb in range(nband):
                t, m, mag, e = (
                    time_ary[index][nb * l : (nb + 1) * l],
                    mask_ary[index][nb * l : (nb + 1) * l],
                    mag_ary[index][nb * l : (nb + 1) * l],
                    magerr_ary[index][nb * l : (nb + 1) * l],
                )
                axs[i, lc_col].errorbar(
                    t[m], mag[m], yerr=e[m], fmt="o", ms=14, color=colors[nb]
                )
            axs[i, lc_col].set_xticklabels([])
            axs[i, lc_col].set_yticklabels([])
            for spine in axs[i, lc_col].spines.values():
                spine.set_linewidth(2.5)

    plt.tight_layout()
    plt.savefig(os.path.join(path_base, "banner.png"))
