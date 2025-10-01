import pandas as pd
import pytest


@pytest.fixture(scope="module")
def concatable_nwp_like_data(ds_nwp_ecmwf):
    """Create two NWP datasets with consecutive init times for concatenation"""
    ds_2 = ds_nwp_ecmwf.copy(deep=True)
    ds_2["init_time"] = pd.date_range(
        start=ds_nwp_ecmwf.init_time.max().values + pd.Timedelta("6h"),
        freq="6h",
        periods=len(ds_nwp_ecmwf.init_time),
    )
    return ds_nwp_ecmwf, ds_2


def _save_nwp_zarr(session_tmp_path, datasets, zarr_format):
    """Save NWP datasets to zarr with specified format"""
    paths = [f"{session_tmp_path}/nwp_like_data_{n}.zarr{zarr_format}"
             for n in range(len(datasets))]
    for ds, path in zip(datasets, paths, strict=False):
        ds.to_zarr(path, zarr_format=zarr_format)
    return paths


@pytest.fixture(scope="module")
def nwp_like_zarr2_paths(session_tmp_path, concatable_nwp_like_data):
    """Save NWP datasets as zarr format 2"""
    return _save_nwp_zarr(session_tmp_path, concatable_nwp_like_data, 2)


@pytest.fixture(scope="module")
def nwp_like_zarr3_paths(session_tmp_path, concatable_nwp_like_data):
    """Save NWP datasets as zarr format 3"""
    return _save_nwp_zarr(session_tmp_path, concatable_nwp_like_data, 3)
