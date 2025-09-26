import xarray as xr

from ocf_data_sampler.load.open_xarray_tensorstore import open_zarr, open_zarrs


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
