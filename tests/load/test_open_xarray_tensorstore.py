import pandas as pd
import pytest
import xarray as xr

from ocf_data_sampler.load.open_xarray_tensorstore import open_zarr, open_zarrs


@pytest.fixture(scope="module")
def concatable_nwp_like_data(ds_nwp_ecmwf):

    # Make a second NWP-like dataset so we can concat them
    ds_2 = ds_nwp_ecmwf.copy(deep=True)
    ds_2["init_time"] = pd.date_range(
        start=ds_nwp_ecmwf.init_time.max().values + pd.Timedelta("6h"),
        freq=pd.Timedelta("6h"),
        periods=len(ds_nwp_ecmwf.init_time),
    )

    return ds_nwp_ecmwf, ds_2


@pytest.fixture(scope="module")
def nwp_like_zarr2_paths(session_tmp_path, concatable_nwp_like_data):

    data_paths = [
        f"{session_tmp_path}/nwp_like_data_{n}.zarr2" for n in range(len(concatable_nwp_like_data))
    ]

    for ds, path in zip(concatable_nwp_like_data, data_paths, strict=False):
        ds.to_zarr(path, zarr_format=2)

    return data_paths


@pytest.fixture(scope="module")
def nwp_like_zarr3_paths(session_tmp_path, concatable_nwp_like_data):

    data_paths = [
        f"{session_tmp_path}/nwp_like_data_{n}.zarr3" for n in range(len(concatable_nwp_like_data))
    ]

    for ds, path in zip(concatable_nwp_like_data, data_paths, strict=False):
        ds.to_zarr(path, zarr_format=3)

    return data_paths


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
    ds_ts = open_zarrs(nwp_like_zarr2_paths, concat_dim="init_time")
    # Check tensorstore version returns same results as dask version
    kwargs = {"concat_dim": "init_time", "combine": "nested", "engine": "zarr"}
    ds_dask = xr.open_mfdataset(nwp_like_zarr2_paths, **kwargs)
    assert ds_ts.compute().equals(ds_dask.compute())

    # Check function can open zarr3
    ds_ts = open_zarrs(nwp_like_zarr3_paths, concat_dim="init_time")
    # Check tensorstore version returns same results as dask version
    ds_dask = xr.open_mfdataset(nwp_like_zarr3_paths, **kwargs)
    assert ds_ts.compute().equals(ds_dask.compute())
