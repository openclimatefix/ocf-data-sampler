import pandas as pd
from xarray import DataArray

from ocf_dataset_alpha.load.nwp import open_nwp


def test_load_ukv(nwp_ukv_zarr_path):
    da = open_nwp(zarr_path=nwp_ukv_zarr_path, provider="ukv")
    assert isinstance(da, DataArray)

def _test_load_ecmwf(ecmwf_nwp_zarr_path):
    da = open_nwp(zarr_path=ecmwf_nwp_zarr_path, provider="ecmwf")
    assert isinstance(da, DataArray)

