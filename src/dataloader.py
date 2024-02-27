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
from astropy.cosmology import Planck15 as cosmo  # Using Planck15 cosmology by default
from typing import Tuple
from torch.utils.data import TensorDataset
from src.utils import find_indices_in_arrays


# Custom data loader with noise augmentation using magerr
class NoisyDataLoader(DataLoader):
    """
    A custom DataLoader that adds noise to images, magnitudes, and spectra in a dataset.

    Attributes:
    - dataset (Dataset): The dataset to load data from.
    - batch_size (int): The size of the batch to load.
    - noise_level_img (float): The level of noise to add to the images.
    - noise_level_mag (float): The level of noise to add to the magnitudes and spectra.
    - shuffle (bool): Whether to shuffle the dataset at every epoch. Defaults to True.
    - **kwargs: Additional keyword arguments for the DataLoader.

    This DataLoader is designed to work with datasets that have different modalities,
    specifically targeting host galaxy images and spectral/light_curve data.
    Noise is added to the data based on specified noise levels, and random rotations
    are applied to the images.
    """

    def __init__(
        self,
        dataset,
        batch_size,
        noise_level_img,
        noise_level_mag,
        shuffle=True,
        **kwargs,
    ):
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)
        self.max_noise_intensity = noise_level_img
        self.noise_level_mag = noise_level_mag
        # If there are eight outputs we are using the spectral/light_curve modaility
        if len(next(iter(dataset))) == 8:
            self.host_galaxy = False
        elif len(next(iter(dataset))) == 4:
            self.host_galaxy = True
        else:
            raise ValueError("Input dataloader has the wrong dimensions")

    def __iter__(self):
        for batch in super().__iter__():
            if self.host_galaxy:
                # Add random noise to images and time-magnitude tensors
                host_imgs, mag, time, mask, magerr = batch

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

                # Return the noisy batch
                yield noisy_imgs, noisy_mag, time, mask
            else:
                # Add random noise to images and time-magnitude tensors
                mag, time, mask, magerr, spec, freq, maskspec, specerr = batch

                # Add Gaussian noise to mag using magerr
                noisy_mag = mag + torch.randn_like(mag) * magerr * self.noise_level_mag

                # Add Gaussian noise to spec using specerr
                noisy_spec = (
                    spec + torch.randn_like(spec) * specerr * self.noise_level_mag
                )

                # Return the noisy batch
                yield noisy_mag, time, mask, spec, freq, maskspec


def filter_files(filenames_avail, filenames_to_filter, data_to_filter):
    """
    Function to filter filenames and data based on the filenames_avail

    Args:
    filenames_avail (list): List of filenames available
    filenames_to_filter (list): List of filenames to filter
    data_to_filter (np.ndarray): Data to filter based on filenames_to_filter

    Returns:
    filenames_to_filter (list): List of filtered filenames
    data_to_filter (np.ndarray): Filtered data
    """
    # Check which each filenames_to_filter are available in filenames_avail
    inds_filt = np.in1d(filenames_to_filter, filenames_avail)
    data_to_filter = data_to_filter[inds_filt]
    filenames_to_filter = np.array(filenames_to_filter)[inds_filt]

    return filenames_to_filter, data_to_filter


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
    host_imgs, filenames_valid = [], []

    filenames = sorted(os.listdir(dir_host_imgs))
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


def load_redshifts(data_dir: str, filenames: List[str]) -> np.ndarray:
    """
    Load redshift values from a CSV file in the specified directory.

    Args:
    data_dir (str): Directory path containing the redshift CSV file.
    filenames (List[str]): List of filenames corresponding to the loaded data.

    Returns:
    np.ndarray: Array of redshift values.
    """
    print("Loading redshifts...")

    # Load values from the CSV file
    df = pd.read_csv(f"{data_dir}/ZTFBTS_TransientTable.csv")

    # Filter redshifts based on the filenames
    redshifts = pd.to_numeric(
        df[df["ZTFID"].isin(filenames)]["redshift"], errors="coerce"
    ).values

    return redshifts


def load_lightcurves(
    data_dir: str,
    abs_mag: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Load light curves from CSV files in the specified directory.

    Args:
    data_dir (str): Directory path containing light curve CSV files.
    abs_mag (bool): If True, convert apparent magnitude to absolute magnitude.

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
    lightcurve_files = sorted(os.listdir(dir_light_curves))  # Sort file names

    mask_list, mag_list, magerr_list, time_list, filenames = [], [], [], [], []

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
                    # Sample n_max_obs observations randomly (note order doesn't matter and the replace flag guarantees no double datapoints)
                    indices = np.random.choice(
                        len(df_band["mag"]), n_max_obs, replace=False
                    )
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
            filenames.append(filename.replace(".csv", ""))

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

    return time_ary, mag_ary, magerr_ary, mask_ary, nband, filenames


def load_spectras(
    data_dir: str,
    n_max_obs: int = 5000,
    zero_pad_missing_error: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Load spectra data from CSV files in the specified directory.

    Args:
        data_dir (str): Path to the directory containing the CSV files.
        n_max_obs (int) default 5000: maximum length of data, shorter data is padded and masked and longer data is shorted by randomly choosing points
        zero_pad_missing_error (bool) default True: if there is missing error in a file, pad the error with zero, otherwise it will be removed

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]: A tuple containing
        arrays of time, magnitude, magnitude error, mask, and filenames.
        - time (np.ndarray): Array of frequency values for each observation.
        - spec (np.ndarray): Array of spectrum values for each observation.
        - specerr (np.ndarray): Array of spectrum error values for each observation.
        - mask (np.ndarray): Array indicating which observations are not padding.
        - filenames (List[str]): List of filenames corresponding to the loaded data.
    """

    print("Loading spectra ...")
    dir_light_curves = f"{data_dir}"

    def open_spectra_csv(filename: str) -> pd.DataFrame:
        """Helper function to open a light curve CSV file."""
        file_path = os.path.join(dir_light_curves, filename)
        return pd.read_csv(file_path, header=None)

    # Getting filenames
    lightcurve_files = sorted(os.listdir(dir_light_curves))
    mask_list, spec_list, specerr_list, freq_list, filenames = [], [], [], [], []

    for filename in tqdm(lightcurve_files):
        if filename.endswith(".csv"):
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
                    spectra_df["specerr"].fillna(0, inplace=True)
                # If no zero-pad remove whole colums with missing data
                else:
                    spectra_df.dropna(subset=["specerr"], inplace=True)
            else:
                ValueError("spectra csv should have 2 or three columns only")

            if max_columns == 3 and np.sum(np.isnan(spectra_df["specerr"])):
                print(filename)
                print(np.where(np.isnan(spectra_df["specerr"].to_numpy()) == True))

            # Checking if the file is too long
            if len(spectra_df["spec"]) > n_max_obs:
                # Sample n_max_obs observations randomly (note order doesn't matter and the replace flag guarantees no double datapoints)
                indices = np.random.choice(
                    len(spectra_df["spec"]), n_max_obs, replace=False
                )
                mask = np.ones(n_max_obs, dtype=bool)
            else:
                # Pad the arrays with zeros and create a mask
                indices = np.arange(len(spectra_df["spec"]))
                mask = np.zeros(n_max_obs, dtype=bool)
                mask[: len(indices)] = True

            # Pad time and mag
            time = np.pad(
                spectra_df["freq"].iloc[indices],
                (0, n_max_obs - len(indices)),
                "constant",
            )
            spec = np.pad(
                spectra_df["spec"].iloc[indices],
                (0, n_max_obs - len(indices)),
                "constant",
            )

            # If there is no error, then just give an empty array with zeros
            if max_columns == 3:
                specerr = np.pad(
                    spectra_df["specerr"].iloc[indices],
                    (0, n_max_obs - len(indices)),
                    "constant",
                )
            else:
                specerr = np.zeros_like(spec)

            mask_list.append(mask)
            freq_list.append(time)
            spec_list.append(spec)
            specerr_list.append(specerr)
            filenames.append(filename.replace(".csv", ""))

    freq_ary = np.array(freq_list)
    spec_ary = np.array(spec_list)
    specerr_ary = np.array(specerr_list)
    mask_ary = np.array(mask_list)

    return freq_ary, spec_ary, specerr_ary, mask_ary, filenames


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
    max_data_len: int = 1000,
    host_galaxy: bool = True,
) -> Tuple[TensorDataset, int]:
    """
    Load data from specified directories, handling both images and light curves or spectra.

    Parameters:
    - data_dir (str): Directory containing images and possibly light curves.
    - spectra_dir (str, optional): Directory containing spectra data. If None, loads light curves instead.
    - max_data_len (int, optional): Maximum length of the data arrays to load. Default is 1000.
    - host_galaxy (bool,optional): If True choose host galaxy as first modality otherwise It will return lightcurve data

    Returns:
    - dataset (TensorDataset): A TensorDataset containing the loaded data.
    - nband (int): Number of bands in the light curve data, or 1 if spectra are loaded.

    The function loads images and either light curves or spectra from the specified directories.
    It ensures that the filenames between images and light curves or spectra match,
    filtering out unmatched data. The resulting dataset is suitable for machine learning models.
    """

    # Decision based on whether spectra data is available
    if spectra_dir is None:
        print("light curve and host galaxy")
        # Load images from data_dir
        host_imgs, filenames_host = load_images(data_dir)
        # Load light curves from data_dir if no spectra directory is provided
        (
            time_ary,
            mag_ary,
            magerr_ary,
            mask_ary,
            nband,
            filenames_lightcurves,
        ) = load_lightcurves(data_dir)

        # Ensuring that filenames between images and light curves match
        assert (
            filenames_host == filenames_lightcurves
        ), "Filenames between host images and light curves must match."

        # Prepare dataset with light curve data
        time = torch.from_numpy(time_ary).float()
        mag = torch.from_numpy(mag_ary).float()
        mask = torch.from_numpy(mask_ary).bool()
        magerr = torch.from_numpy(magerr_ary).float()
        dataset = TensorDataset(host_imgs, mag, time, mask, magerr)
        return dataset, nband
    else:
        # Decision if you want to combine spectral data with host galaxy or light curve
        if host_galaxy:
            print("spectral and host galaxy")
            # Load images from data_dir
            host_imgs, filenames_host = load_images(data_dir)
            # Load spectra from spectra_dir if provided
            (
                freq_ary,
                spec_ary,
                specerr_ary,
                maskspec_ary,
                filenames_spectra,
            ) = load_spectras(spectra_dir, max_data_len)

            # Finding overlapping filenames between spectra and images
            inds_spec, inds_light = find_indices_in_arrays(
                filenames_spectra, filenames_host
            )

            # Filter data to ensure each spectrum has a corresponding image
            freq_ary, spec_ary, specerr_ary, maskspec_ary, filenames_spectra = [
                arr[inds_light]
                for arr in [
                    freq_ary,
                    spec_ary,
                    specerr_ary,
                    maskspec_ary,
                    np.array(filenames_spectra),
                ]
            ]
            host_imgs, filenames_host = [
                arr[inds_spec] for arr in [host_imgs, np.array(filenames_host)]
            ]

            # Ensuring that filtered filenames match between host images and spectra
            assert (
                filenames_host.tolist() == filenames_spectra.tolist()
            ), "Filtered filenames between host images and spectra must match."

            # Prepare dataset with spectra data
            freq = torch.from_numpy(freq_ary).float()
            spec = torch.from_numpy(spec_ary).float()
            maskspec = torch.from_numpy(maskspec_ary).bool()
            specerr = torch.from_numpy(specerr_ary).float()
            dataset = TensorDataset(host_imgs, freq, spec, maskspec, specerr)
            nband = 1  # Number of bands is set to 1 for spectra
            return dataset, nband

        else:
            print("spectral and light curve")
            # Load light curves from data_dir if no spectra directory is provided
            (
                time_ary,
                mag_ary,
                magerr_ary,
                mask_ary,
                nband,
                filenames_lightcurves,
            ) = load_lightcurves(data_dir)

            # Load spectra from spectra_dir if provided
            (
                freq_ary,
                spec_ary,
                specerr_ary,
                maskspec_ary,
                filenames_spectra,
            ) = load_spectras(spectra_dir, max_data_len)

            # Finding overlapping filenames between spectra and images
            inds_spec, inds_light = find_indices_in_arrays(
                filenames_spectra, filenames_lightcurves
            )

            # Filter data to ensure each spectrum has a corresponding image
            freq_ary, spec_ary, specerr_ary, maskspec_ary, filenames_spectra = [
                arr[inds_light]
                for arr in [
                    freq_ary,
                    spec_ary,
                    specerr_ary,
                    maskspec_ary,
                    np.array(filenames_spectra),
                ]
            ]
            time_ary, mag_ary, magerr_ary, mask_ary, filenames_lightcurves = [
                arr[inds_spec]
                for arr in [
                    time_ary,
                    mag_ary,
                    magerr_ary,
                    mask_ary,
                    np.array(filenames_lightcurves),
                ]
            ]

            # Ensuring that filtered filenames match between host images and spectra
            assert (
                filenames_lightcurves.tolist() == filenames_spectra.tolist()
            ), "Filtered filenames between host images and spectra must match."

            # Prepare dataset with spectra data
            time = torch.from_numpy(time_ary).float()
            mag = torch.from_numpy(mag_ary).float()
            mask = torch.from_numpy(mask_ary).bool()
            magerr = torch.from_numpy(magerr_ary).float()

            freq = torch.from_numpy(freq_ary).float()
            spec = torch.from_numpy(spec_ary).float()
            maskspec = torch.from_numpy(maskspec_ary).bool()
            specerr = torch.from_numpy(specerr_ary).float()
            dataset = TensorDataset(
                mag, time, mask, magerr, spec, freq, maskspec, specerr
            )

            return dataset, nband
