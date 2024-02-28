import torch
import os
import sys
from src.dataloader import (
    load_images,
    load_lightcurves,
    load_spectras,
    load_data,
    plot_lightcurve_and_images,
    NoisyDataLoader,
)


def test_dataloader():
    # Loading 
    max_data_len = 1000
    dataset, nband = load_data("ZTFBTS", "ZTFBTS_spectra/", max_data_len)


if __name__ == "__main__":
    test_dataloader()
