import os
import torch
import h5py
import numpy as np
from typing import List, Tuple, Optional
from PIL import Image

import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from einops import rearrange

from torchvision.transforms import RandomRotation
from torch.utils.data import DataLoader, Dataset
from astropy.cosmology import Planck15 as cosmo  # Using Planck15 cosmology by default
from typing import Tuple, List, Optional
from torch.utils.data import TensorDataset
from src.utils import filter_files, find_indices_in_arrays


# Custom data loader with noise augmentation using magerr
class NoisyDataLoader(DataLoader):
    """
    A custom DataLoader that adds noise to images, magnitudes, and spectra in a dataset.

    Attributes:
    - dataset (Dataset): The dataset to load data from.
    - batch_size (int): The size of the batch to load.
    - noise_level_img (float): The level of noise to add to the images.
    - noise_level_mag (float): The level of noise to add to the magnitudes and spectra.
    - combinations (List[str]): Contains the modalities we are working with, the options are
        'host_galaxy','spectral' or 'lightcurve'
    - shuffle (bool): Whether to shuffle the dataset at every epoch. Defaults to True.
    - **kwargs: Additional keyword arguments for the DataLoader.

    This DataLoader is designed to work with datasets that have different modalities,
    specifically targeting host galaxy images and spectral/light_curve data.
    Noise is added to the data based on specified noise levels, and random rotations
    are applied to the images.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        noise_level_img: float,
        noise_level_mag: float,
        combinations: Optional[List[str]] = ["host_galaxy", "spectral"],
        shuffle: bool = True,
        **kwargs,
    ):
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)
        self.max_noise_intensity = noise_level_img
        self.noise_level_mag = noise_level_mag
        self.combinations = set(combinations)

        # Checking if we get the right output
        if len(next(iter(dataset))) == 10:
            assert "lightcurve" in self.combinations
            assert "host_galaxy" not in self.combinations
            assert "spectral" in self.combinations
        elif len(next(iter(dataset))) == 11:
            assert "lightcurve" in self.combinations
            assert "host_galaxy" in self.combinations
            assert "spectral" in self.combinations
        elif len(next(iter(dataset))) == 7:
            assert "lightcurve" in self.combinations or "spectral" in self.combinations
            assert "host_galaxy" in self.combinations
        elif len(next(iter(dataset))) == 3:
            assert "host_galaxy" in self.combinations and len(self.combinations) == 1
        elif len(next(iter(dataset))) == 6:
            assert (
                "lightcurve" in self.combinations or "spectral" in self.combinations
            ) and len(self.combinations) == 1
        else:
            raise ValueError(
                "Input dataloader has the wrong dimensions; has dimension {} which is unexpected".format(
                    len(next(iter(dataset)))
                )
            )

    def __iter__(self):
        for batch in super().__iter__():
            if self.combinations == set(["host_galaxy"]):
                # Add random noise to images
                host_imgs, redshift, classification = batch

                # Calculate the range for the random noise based on the max_noise_intensity
                noise_range = self.max_noise_intensity * torch.std(host_imgs)

                # Generate random noise within the specified range
                noisy_imgs = (
                    host_imgs + (2 * torch.rand_like(host_imgs) - 1) * noise_range
                )

                # Randomly apply rotation by multiples of 90 degrees
                rotation_angle = torch.randint(0, 4, (noisy_imgs.size(0),)) * 90
                rotated_imgs = []

                # Apply rotation to each image
                for i in range(noisy_imgs.size(0)):
                    rotated_img = RandomRotation(
                        [rotation_angle[i], rotation_angle[i]]
                    )(noisy_imgs[i])
                    rotated_imgs.append(rotated_img)

                # Stack the rotated images back into a tensor
                rotated_imgs = torch.stack(rotated_imgs)

                # Return the noisy batch (and Nones to keep outputlength the same)
                yield rotated_imgs, None, None, None, None, None, None, redshift, classification

            elif self.combinations == set(["lightcurve"]):
                # Add random noise to time-magnitude tensors
                mag, time, mask, magerr, redshift, classification = batch

                # Add Gaussian noise to mag using magerr
                noisy_mag = mag + torch.randn_like(mag) * magerr * self.noise_level_mag

                # Return the noisy batch (and Nones to keep outputlength the same)
                yield None, noisy_mag, time, mask, None, None, None, redshift, classification

            elif self.combinations == set(["spectral"]):
                # Add random noise to spectra tensors
                spec, freq, maskspec, specerr, redshift, classification = batch

                # Add Gaussian noise to spec using specerr
                noisy_spec = (
                    spec + torch.randn_like(spec) * specerr * self.noise_level_mag
                )

                # Return the noisy batch (and Nones to keep outputlength the same)
                yield None, None, None, None, noisy_spec, freq, maskspec, redshift, classification

            elif self.combinations == set(["host_galaxy", "lightcurve"]):
                # Add random noise to images and time-magnitude tensors
                host_imgs, mag, time, mask, magerr, redshift, classification = batch

                # Calculate the range for the random noise based on the max_noise_intensity
                noise_range = self.max_noise_intensity * torch.std(host_imgs)

                # Generate random noise within the specified range
                noisy_imgs = (
                    host_imgs + (2 * torch.rand_like(host_imgs) - 1) * noise_range
                )

                # Add Gaussian noise to mag using magerr
                noisy_mag = mag + torch.randn_like(mag) * magerr * self.noise_level_mag

                # Randomly apply rotation by multiples of 90 degrees
                rotation_angle = torch.randint(0, 4, (noisy_imgs.size(0),)) * 90
                rotated_imgs = []

                # Apply rotation to each image
                for i in range(noisy_imgs.size(0)):
                    rotated_img = RandomRotation(
                        [rotation_angle[i], rotation_angle[i]]
                    )(noisy_imgs[i])
                    rotated_imgs.append(rotated_img)

                # Stack the rotated images back into a tensor
                rotated_imgs = torch.stack(rotated_imgs)

                # Return the noisy batch (and Nones to keep outputlength the same)
                yield rotated_imgs, noisy_mag, time, mask, None, None, None, redshift, classification

            elif self.combinations == set(["host_galaxy", "spectral"]):
                # Add random noise to images and time-magnitude tensors
                (
                    host_imgs,
                    spec,
                    freq,
                    maskspec,
                    specerr,
                    redshift,
                    classification,
                ) = batch

                # Calculate the range for the random noise based on the max_noise_intensity
                noise_range = self.max_noise_intensity * torch.std(host_imgs)

                # Generate random noise within the specified range
                noisy_imgs = (
                    host_imgs + (2 * torch.rand_like(host_imgs) - 1) * noise_range
                )

                # Add Gaussian noise to spec using specerr
                noisy_spec = (
                    spec + torch.randn_like(spec) * specerr * self.noise_level_mag
                )

                # Randomly apply rotation by multiples of 90 degrees
                rotation_angle = torch.randint(0, 4, (noisy_imgs.size(0),)) * 90
                rotated_imgs = []

                # Apply rotation to each image
                for i in range(noisy_imgs.size(0)):
                    rotated_img = RandomRotation(
                        [rotation_angle[i], rotation_angle[i]]
                    )(noisy_imgs[i])
                    rotated_imgs.append(rotated_img)

                # Stack the rotated images back into a tensor
                rotated_imgs = torch.stack(rotated_imgs)

                # Return the noisy batch (and Nones to keep outputlength the same)
                yield rotated_imgs, None, None, None, noisy_spec, freq, maskspec, redshift, classification

            elif self.combinations == set(["spectral", "lightcurve"]):
                # Add random noise to images and time-magnitude tensors
                (
                    mag,
                    time,
                    mask,
                    magerr,
                    spec,
                    freq,
                    maskspec,
                    specerr,
                    redshift,
                    classification,
                ) = batch

                # Add Gaussian noise to mag using magerr
                noisy_mag = mag + torch.randn_like(mag) * magerr * self.noise_level_mag

                # Add Gaussian noise to spec using specerr
                noisy_spec = (
                    spec + torch.randn_like(spec) * specerr * self.noise_level_mag
                )

                # Return the noisy batch (and Nones to keep outputlength the same)
                yield None, noisy_mag, time, mask, noisy_spec, freq, maskspec, redshift, classification

            elif self.combinations == set(["host_galaxy", "spectral", "lightcurve"]):
                # Add random noise to images and time-magnitude tensors
                (
                    host_imgs,
                    mag,
                    time,
                    mask,
                    magerr,
                    spec,
                    freq,
                    maskspec,
                    specerr,
                    redshift,
                    classification,
                ) = batch

                # Calculate the range for the random noise based on the max_noise_intensity
                noise_range = self.max_noise_intensity * torch.std(host_imgs)

                # Generate random noise within the specified range
                noisy_imgs = (
                    host_imgs + (2 * torch.rand_like(host_imgs) - 1) * noise_range
                )

                # Add Gaussian noise to mag using magerr
                noisy_mag = mag + torch.randn_like(mag) * magerr * self.noise_level_mag

                # Add Gaussian noise to spec using specerr
                noisy_spec = (
                    spec + torch.randn_like(spec) * specerr * self.noise_level_mag
                )

                # Randomly apply rotation by multiples of 90 degrees
                rotation_angle = torch.randint(0, 4, (noisy_imgs.size(0),)) * 90
                rotated_imgs = []

                # Apply rotation to each image
                for i in range(noisy_imgs.size(0)):
                    rotated_img = RandomRotation(
                        [rotation_angle[i], rotation_angle[i]]
                    )(noisy_imgs[i])
                    rotated_imgs.append(rotated_img)

                # Stack the rotated images back into a tensor
                rotated_imgs = torch.stack(rotated_imgs)

                yield rotated_imgs, noisy_mag, time, mask, noisy_spec, freq, maskspec, redshift, classification


def load_images(data_dir: str, filenames: List[str] = None) -> torch.Tensor:
    """
    Load and preprocess images from a specified directory.

    Args:
    data_dir (str): The directory path where images are stored.

    Returns:
    torch.Tensor: A tensor containing the preprocessed images.
    List[str]: List of filenames corresponding to the loaded data.
    """
    print("Loading images...")

    dir_host_imgs = f"{data_dir}/hostImgs/"
    host_imgs, filenames_valid = [], []

    if filenames is None:
        filenames = sorted(os.listdir(dir_host_imgs))
    else:  # If filenames are provided, filter the filenames
        _, filenames, _ = filter_files(
            sorted(os.listdir(dir_host_imgs)), [f + ".host.png" for f in filenames]
        )

    # Iterate through the directory and load images
    for filename in tqdm(filenames):
        file_path = os.path.join(dir_host_imgs, filename)
        if file_path.endswith(".png"):
            # Load image, convert to RGB, and then to a NumPy array
            host_img = Image.open(file_path).convert("RGB")
            host_img = np.asarray(host_img)
            host_imgs.append(host_img)
            filenames_valid.append(filename.replace(".host.png", ""))

    # Convert the list of images to a NumPy array
    host_imgs = np.array(host_imgs)

    # Convert the NumPy array to a PyTorch tensor and rearrange dimensions
    host_imgs = torch.from_numpy(host_imgs).float()
    host_imgs = rearrange(host_imgs, "b h w c -> b c h w")

    # Normalize the images
    host_imgs /= 255.0

    return host_imgs, filenames_valid


def load_redshifts(data_dir: str, filenames: List[str] = None) -> np.ndarray:
    """
    Load redshift values from a CSV file in the specified directory.

    Args:
    data_dir (str): Directory path containing the redshift CSV file.
    filenames (List[str]): List of filenames corresponding to the loaded data; default is None.

    Returns:
    np.ndarray: Array of redshift values.
    filenames (List[str]): List of filenames corresponding to the returned data.
    """
    print("Loading redshifts...")

    # Load values from the CSV file
    df = pd.read_csv(f"{data_dir}/ZTFBTS_TransientTable.csv")
    df["redshift"] = pd.to_numeric(df["redshift"], errors="coerce")
    df = df.dropna(subset=["redshift"])

    if filenames is None:
        redshifts = df["redshift"].values
        filenames_redshift = df["ZTFID"].values
    else:
        # Filter redshifts based on the filenames
        redshifts = df[df["ZTFID"].isin(filenames)]["redshift"].values

        filenames_redshift = df[df["ZTFID"].isin(filenames)]["ZTFID"].values

    print("Finished loading redshift")
    return redshifts, filenames_redshift


def load_classes(
    data_dir: str, n_classes: int = 5, filenames: List[str] = None
) -> np.ndarray:
    """
    Load classification values from a CSV file in the specified directory.

    Args:
    data_dir (str): Directory path containing the class CSV file.
    filenames (List[str]): List of filenames corresponding to the loaded data; default is None.

    Returns:
    np.ndarray: Array of transient classes values.
    filenames (List[str]): List of filenames corresponding to the returned data.
    """
    print("Loading transient classes...")

    # Load values from the CSV file
    df = pd.read_csv(f"{data_dir}/ZTFBTS_TransientTable.csv")
    df = df.dropna(subset=["type"])

    # only consider five-way classification
    df.loc[df["type"] == "SN Ib", "type"] = "SN Ibc"
    df.loc[df["type"] == "SN Ic", "type"] = "SN Ibc"
    df.loc[df["type"] == "SN Ib/c", "type"] = "SN Ibc"
    df.loc[df["type"] == "SN IIP", "type"] = "SN II"

    if n_classes == 5:
        df = df[df["type"].isin(["SN Ia", "SN Ibc", "SLSN-I", "SN II", "SN IIn"])]
    elif n_classes == 3:
        df = df[df["type"].isin(["SN Ia", "SN Ibc", "SN II"])]

    # Use the Series to map the names to types
    class_types = df["type"].values
    df["type_factorized"] = pd.factorize(class_types, sort=True)[0]
    # factorized types for 5-class will be as follows:
    #    'SLSN-I', 'SN II', 'SN IIn', 'SN Ia', 'SN Ibc'
    # factorized types for 3-class will be:
    #    'SN II', 'SN Ia', 'SN Ibc'

    if filenames is None:
        classifications = df["type_factorized"].values
        filenames_class = df["ZTFID"].values
    else:
        # Filter classifications based on the filenames
        classifications = df[df["ZTFID"].isin(filenames)]["type_factorized"].values
        filenames_class = df[df["ZTFID"].isin(filenames)]

    print("Finished loading transient classes.")
    return classifications, filenames_class


def make_padding_mask(n_obs: int, n_max_obs: int) -> np.ndarray:
    """
    Args:
    n_obs (int): number of observations in the light curve
    n_max_obs (int): maximum number of observations to pad to/filter

    Returns:
    Tuple[np.ndarray, np.ndarray]: indices of the observations to
                    keep and mask for the observations

    """

    if n_obs > n_max_obs:
        # Sample n_max_obs observations randomly (note order doesn't matter and the replace flag guarantees no double datapoints)
        indices = np.random.choice(n_obs, n_max_obs, replace=False)
        mask = np.ones(n_max_obs, dtype=bool)
    else:
        # Pad the arrays with zeros and create a mask
        indices = np.arange(n_obs)
        mask = np.zeros(n_max_obs, dtype=bool)
        mask[: len(indices)] = True

    return indices, mask


def load_lightcurves(
    data_dir: str,
    abs_mag: bool = False,
    n_max_obs: int = 100,
    filenames: List[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Load light curves from CSV files in the specified directory; load files that are available if
    filenames are provided.

    Args:
    data_dir (str): Directory path containing light curve CSV files.
    abs_mag (bool): If True, convert apparent magnitude to absolute magnitude.
    n_max_obs (int): Maximum number of data points per lightcurve.
    filenames (List[str], optional): List of filenames to load. If None, all files are loaded.

    Returns:
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]: A tuple containing:
        - time_ary: Numpy array of time observations.
        - mag_ary: Numpy array of magnitude observations.
        - magerr_ary: Numpy array of magnitude error observations.
        - mask_ary: Numpy array indicating the presence of an observation.
        - nband: Number of observation bands.
        - filenames_loaded: List of filenames corresponding to the loaded data.
    """

    print("Loading light curves...")
    dir_light_curves = f"{data_dir}/light-curves/"

    def open_light_curve_csv(filename: str) -> pd.DataFrame:
        """Helper function to open a light curve CSV file."""
        file_path = os.path.join(dir_light_curves, filename)
        return pd.read_csv(file_path)

    bands = ["R", "g"]
    nband = len(bands)
    if filenames is None:
        filenames = sorted(os.listdir(dir_light_curves))  # Sort file names
    else:  # If filenames are provided, filter the filenames
        _, filenames, _ = filter_files(
            sorted(os.listdir(dir_light_curves)), [f + ".csv" for f in filenames]
        )

    mask_list, mag_list, magerr_list, time_list, filenames_loaded = [], [], [], [], []

    for filename in tqdm(filenames):
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

                indices, mask = make_padding_mask(len(df_band["mag"]), n_max_obs)

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

                # Normalise time if there is anything to normalise
                if sum(mask) != 0:
                    time[mask] = time[mask] - np.min(time[mask])

                time_concat += list(time)
                mag_concat += list(mag)
                magerr_concat += list(magerr)
                mask_concat += list(mask)

            mask_list.append(mask_concat)
            time_list.append(time_concat)
            mag_list.append(mag_concat)
            magerr_list.append(magerr_concat)
            filenames_loaded.append(filename.replace(".csv", ""))

    time_ary = np.array(time_list)
    mag_ary = np.array(mag_list)
    magerr_ary = np.array(magerr_list)
    mask_ary = np.array(mask_list)

    if abs_mag:
        print("Converting to absolute magnitude...", flush=True)

        zs = load_redshifts(data_dir, filenames)
        inds = ~np.isnan(zs)

        # Convert from apparent magnitude to absolute magnitude
        mag_ary -= cosmo.distmod(zs).value[:, None]

        time_ary = time_ary[inds]
        mag_ary = mag_ary[inds]
        magerr_ary = magerr_ary[inds]
        mask_ary = mask_ary[inds]

        filenames = np.array(filenames)[inds]

    return time_ary, mag_ary, magerr_ary, mask_ary, nband, filenames_loaded


def load_spectras(
    data_dir: str,
    n_max_obs: int = 5000,
    zero_pad_missing_error: bool = True,
    rescalefactor: int = 1e14,
    filenames: List[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Load spectra data from CSV files in the specified directory; load files that are available if
    filneames are provided.

    Args:
        data_dir (str): Path to the directory containing the CSV files.
        n_max_obs (int) default 5000: maximum length of data, shorter data is padded and masked and longer data is shorted by randomly choosing points
        zero_pad_missing_error (bool) default True: if there is missing error in a file, pad the error with zero, otherwise it will be removed
        rescalefactor (int) default 1e14: factor to rescale the spectrum data
        filenames (List[str], optional): List of filenames to load. If None, all files are loaded.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]: A tuple containing
        arrays of time, magnitude, magnitude error, mask, and filenames.
        - time (np.ndarray): Array of frequency values for each observation.
        - spec (np.ndarray): Array of spectrum values for each observation.
        - specerr (np.ndarray): Array of spectrum error values for each observation.
        - mask (np.ndarray): Array indicating which observations are not padding.
        - filenames_loaded (List[str]): List of filenames corresponding to the loaded data.
    """

    print("Loading spectra ...")
    dir_data = f"{data_dir}"

    def open_spectra_csv(filename: str) -> pd.DataFrame:
        """Helper function to open a light curve CSV file."""
        file_path = os.path.join(dir_data, filename)
        return pd.read_csv(file_path, header=None)

    if filenames is None:
        # Getting filenames
        filenames = sorted(os.listdir(dir_data))
    else:
        _, filenames, _ = filter_files(
            sorted(os.listdir(dir_data)), [f + ".csv" for f in filenames]
        )

    mask_list, spec_list, specerr_list, freq_list, filenames_loaded = [], [], [], [], []

    for filename in tqdm(filenames):
        if filename.endswith(".csv") and not filename.startswith('.'):
            spectra_df = open_spectra_csv(filename)
            max_columns = spectra_df.shape[1]

            # Checking size and naming dependent on that
            # Note: not all spectra have errors
            if max_columns == 2:
                spectra_df.columns = ["freq", "spec"]
            elif max_columns == 3:
                spectra_df.columns = ["freq", "spec", "specerr"]
                # Fill missing data with zeros
                if zero_pad_missing_error:
                    spectra_df["specerr"] = spectra_df["specerr"].fillna(0)
                # If no zero-pad remove whole colums with missing data
                else:
                    spectra_df.dropna(subset=["specerr"], inplace=True)
            else:
                ValueError("spectra csv should have 2 or three columns only")

            indices, mask = make_padding_mask(len(spectra_df["spec"]), n_max_obs)

            # Pad time and mag
            freq = np.pad(
                spectra_df["freq"].iloc[indices],
                (0, n_max_obs - len(indices)),
                "constant",
            )
            spec = rescalefactor * np.pad(
                spectra_df["spec"].iloc[indices],
                (0, n_max_obs - len(indices)),
                "constant",
            )

            # If there is no error, then just give an empty array with zeros
            if max_columns == 3:
                specerr = rescalefactor * np.pad(
                    spectra_df["specerr"].iloc[indices],
                    (0, n_max_obs - len(indices)),
                    "constant",
                )
            else:
                specerr = np.zeros_like(spec)

            mask_list.append(mask)
            freq_list.append(freq)
            spec_list.append(spec)
            specerr_list.append(specerr)
            filenames_loaded.append(filename.replace(".csv", ""))

    freq_ary = np.array(freq_list)
    spec_ary = np.array(spec_list)
    specerr_ary = np.array(specerr_list)
    mask_ary = np.array(mask_list)

    return freq_ary, spec_ary, specerr_ary, mask_ary, filenames_loaded


def plot_lightcurve_and_images(
    host_imgs: torch.Tensor,
    time_ary: np.ndarray,
    mag_ary: np.ndarray,
    magerr_ary: np.ndarray,
    mask_ary: np.ndarray,
    nband: int,
    path_base: str = "./",
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
    path_base (str): Path to save the plot; default is the current directory.
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


def load_data(
    data_dir: str,
    spectra_dir: str = None,
    max_data_len_lc: int = 100,
    max_data_len_spec: int = 1000,
    combinations: List = ["host_galaxy", "lightcurve"],
    n_classes: int = 5,
    spectral_rescalefactor: float = 1e14,
) -> Tuple[TensorDataset, int]:
    """
    Load data from specified directories, handling both images and light curves or spectra.

    Args:
    data_dir (str): Directory containing images and possibly light curves.
    spectra_dir (str, optional): Directory containing spectra data. If None, loads light curves instead.
    max_data_len_lc (int, optional): Maximum length of the light curve arrays to load. Default is 100.
    max_data_len_spec (int, optional): Maximum length of the spectra arrays to load. Default is 1000.
    combinations (List[str], optional): List of modalities to load. Default is ["host_galaxy", "lightcurve"].
    spectral_rescalefactor (int) default 1e14: factor to rescale the spectrum data


    Returns:
    dataset (TensorDataset): A TensorDataset containing the loaded data.
    nband (int): Number of bands in the light curve data, or 1 if spectra are loaded.
    filenames (List[str]): List of filenames corresponding to the loaded data.

    The function loads images, light curves, and/or spectra data from the specified directories.
    It ensures that the filenames between images and light curves or spectra match,
    filtering out unmatched data. The resulting dataset is suitable for machine learning models.
    """

    # Decision based on whether spectra data is available
    if spectra_dir is None:
        spectra_dir = data_dir

    data, filenames = [], None
    nband = 1  # Default number of bands for spectra data

    if "host_galaxy" in combinations:
        # Load images from data_dir
        host_imgs, filenames_host = load_images(data_dir)

        data.append(host_imgs)
        filenames = filenames_host

    if "lightcurve" in combinations:
        # Load light curves from data_dir if no spectra directory is provided
        (
            time_ary,
            mag_ary,
            magerr_ary,
            mask_ary,
            nband,
            filenames_lightcurves,
        ) = load_lightcurves(data_dir, n_max_obs=max_data_len_lc, filenames=filenames)

        # Ensuring that filenames between images and light curves match if we loaded images
        if filenames is None:
            filenames = filenames_lightcurves
        else:
            _, filenames, data = filter_files(filenames_lightcurves, filenames, data)

        # Prepare dataset with light curve data
        time = torch.from_numpy(time_ary).float()
        mag = torch.from_numpy(mag_ary).float()
        mask = torch.from_numpy(mask_ary).bool()
        magerr = torch.from_numpy(magerr_ary).float()

        data += [mag, time, mask, magerr]

    if "spectral" in combinations:
        # Load spectra from spectra_dir if provided
        (
            freq_ary,
            spec_ary,
            specerr_ary,
            maskspec_ary,
            filenames_spectra,
        ) = load_spectras(
            spectra_dir,
            n_max_obs=max_data_len_spec,
            rescalefactor=spectral_rescalefactor,
            filenames=filenames,
        )

        if filenames is not None:
            _, filenames, data = filter_files(filenames_spectra, filenames, data)
        else:
            filenames = filenames_spectra

        assert (
            list(filenames) == filenames_spectra
        ), "Filtered filenames between modalities must match."

        freq = torch.from_numpy(freq_ary).float()
        spec = torch.from_numpy(spec_ary).float()
        maskspec = torch.from_numpy(maskspec_ary).bool()
        specerr = torch.from_numpy(specerr_ary).float()
        data += [spec, freq, maskspec, specerr]

    # Always load the redshift
    redshifts, filenames_redshift = load_redshifts(f"{data_dir}", filenames)
    _, filenames, data = filter_files(filenames_redshift, filenames, data)

    assert (
        list(filenames) == filenames_redshift
    ).all(), "Filtered filenames between modalities must match."

    # Prepare dataset with spectra data
    redshifts = torch.from_numpy(redshifts).float()
    data += [redshifts]

    # Load transient types
    classifications, filenames_classifications = load_classes(
        f"{data_dir}", n_classes, filenames
    )
    _, filenames, data = filter_files(filenames_classifications, filenames, data)

    # Prepare dataset with spectra data
    classifications = torch.from_numpy(classifications).float()
    data += [classifications]

    data = TensorDataset(*data)

    return data, nband, filenames


class SimulationLightcurveDataset(Dataset):
    """
    A dataset class for handling transient astronomical data stored in HDF5 files.

    Attributes:
        hdf5_path (str): The path to the HDF5 file containing transient data.
        transient_types (Optional[List[str]]): A list of transient event types to load from the file.
            If None, all available types in the file are loaded.
        bands (List[str]): The list of photometric bands to retrieve data for (e.g., ['r', 'g', 'b']).
        index_map (List[Tuple[str, str, int]]): A list of tuples where each tuple contains the
            transient type, model, and entry index for quick data retrieval.

    Methods:
        __len__: Returns the number of entries in the dataset.
        __getitem__: Retrieves a dataset entry by index, loading data from the file as needed.
    """

    def __init__(
        self,
        hdf5_path: str,
        transient_types: Optional[List[str]] = None,
        bands: List[str] = ["r"],
        n_max_obs=100,
    ) -> None:
        """
        Initializes the dataset object by opening the HDF5 file and precalculating indices for quick access.

        Args:
            hdf5_path (str): Path to the HDF5 file.
            transient_types (Optional[List[str]]): List of transient types in the HDF5 file. If None,
                defaults to using all keys available in the file.
            bands (List[str]): List of bands of interest (e.g., ['r', 'g', 'b']).
        """
        self.hdf5_path = hdf5_path
        self.bands = bands
        self.n_max_obs = n_max_obs

        # Open the HDF5 file
        with h5py.File(self.hdf5_path, "r") as file:
            transients = file["TransientTable"]
            # Default to using all keys if no specific types are provided
            if transient_types is None:
                transient_types = list(transients.keys())
            self.transient_types = transient_types

            # Pre-calculate indices for each entry
            self.index_map = []
            for t_type in self.transient_types:
                for model in transients[t_type].keys():
                    num_entries = len(transients[t_type][model]["MJD"])
                    for i in range(num_entries):
                        self.index_map.append((t_type, model, i))

    def __len__(self) -> int:
        """Returns the number of entries in the dataset."""
        return 5000  # len(self.index_map)

    def __getitem__(self, idx: int) -> Tuple[List[float], List[float]]:
        """
        Retrieves an entry from the dataset by its index, dynamically loading data from the HDF5 file.

        Args:
            idx (int): The index of the data entry to retrieve.

        Returns:
            Tuple[List[float], List[float]]: A tuple containing time and magnitude data for the requested entry.
        """
        t_type, model, entry_idx = self.index_map[idx]

        # Access the HDF5 file for each item
        with h5py.File(self.hdf5_path, "r") as file:
            transient_model = file["TransientTable"][t_type][model]

            # Append data from multiple bands
            data = []
            time = []
            mask_concat = []
            for band in self.bands:
                time_data = transient_model["MJD"][entry_idx]
                mag_data = transient_model[f"mag_{band}"][entry_idx]

                time_data = time_data[mag_data < 98]
                mag_data = (mag_data[mag_data < 98] - 23.74)/1.6

                indices, mask = make_padding_mask(len(time_data), self.n_max_obs)

                time_data = np.pad(
                    time_data[indices],
                    (0, self.n_max_obs - len(indices)),
                    "constant",
                )
                mag_data = np.pad(
                    mag_data[indices],
                    (0, self.n_max_obs - len(indices)),
                    "constant",
                )

                # Normalise time if there is anything to normalise
                if sum(mask) != 0:
                    time_data[mask] = time_data[mask] - np.min(time_data[mask])

                data += list(mag_data)
                time += list(time_data)
                mask_concat += list(mask)

        return (
            torch.tensor(time).float(),
            torch.tensor(data).float(),
            torch.tensor(mask_concat).bool(),
        )
