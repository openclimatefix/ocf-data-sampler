import pandas as pd
import pytest
import xarray as xr

from ocf_data_sampler.common.xr_tensorstore import open_zarr, open_zarrs


@pytest.fixture(scope="module")
def concatable_nwp_like_data(ds_nwp_ecmwf):
    """Create two NWP datasets with consecutive init times for concatenation"""
    ds_2 = ds_nwp_ecmwf.copy(deep=True)
    ds_2["init_time_utc"] = pd.date_range(
        start=ds_nwp_ecmwf.init_time_utc.values.max() + pd.Timedelta("6h"),
        freq="6h",
        periods=len(ds_nwp_ecmwf.init_time_utc),
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


def test_open_zarr(nwp_like_zarr2_paths, nwp_like_zarr3_paths):
    # Check function can open zarr2
    ds_ts = open_zarr(nwp_like_zarr2_paths[0])
    # Check tensorstore version returns same results as dask version
    ds_dask = xr.open_zarr(nwp_like_zarr2_paths[0])
    assert ds_ts.compute().equals(ds_dask.compute())

    # Check function can open zarr3
    ds_ts = open_zarr(nwp_like_zarr3_paths[0])
    # Check tensorstore version returns same results as dask version
    ds_dask = xr.open_zarr(nwp_like_zarr3_paths[0])
    assert ds_ts.compute().equals(ds_dask.compute())


def test_open_zarrs(nwp_like_zarr2_paths, nwp_like_zarr3_paths):
    # Check function can open zarr2
    ds_ts = open_zarrs(nwp_like_zarr2_paths, concat_dim="init_time_utc")
    # Check tensorstore version returns same results as dask version
    kwargs = {"concat_dim": "init_time_utc", "combine": "nested", "engine": "zarr"}
    ds_dask = xr.open_mfdataset(nwp_like_zarr2_paths, **kwargs)
    assert ds_ts.compute().equals(ds_dask.compute())

    # Check function can open zarr3
    ds_ts = open_zarrs(nwp_like_zarr3_paths, concat_dim="init_time_utc")
    # Check tensorstore version returns same results as dask version
    ds_dask = xr.open_mfdataset(nwp_like_zarr3_paths, **kwargs)
    assert ds_ts.compute().equals(ds_dask.compute())
