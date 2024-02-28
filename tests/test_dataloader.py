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
    # Loading spectra and galaxy 
    max_data_len = 1000
    load_galaxy = True
    # Note that load_data has internal asserts, checking if the right files are mached 
    dataset, nband = load_data("ZTFBTS", "ZTFBTS_spectra/", max_data_len,load_galaxy)
    assert(next(iter(dataset)) == 5)
    assert(nband == 1)
    # Loading lightcurve and galaxy 
    max_data_len = 1000
    dataset, nband = load_data("ZTFBTS", None, max_data_len)
    assert(next(iter(dataset)) == 5)
    assert(nband == 2)
    host_imgs, mag, time, mask, magerr = next(iter(dataset))
    for t,m in time,mask:
        print(torch.min(t[mask]))
        assert(torch.min(t[mask]) ==0)

if __name__ == "__main__":
    test_dataloader()
