import pandas as pd
import pytest
import xarray as xr

from ocf_data_sampler.common.xr_tensorstore import (
    _extract_tensorstore,
    concat_tensorstore,
    open_zarr_paths,
)

ZARR_FORMATS = [2, 3]


@pytest.fixture(scope="module")
def concatable_nwp_like_data(ds_nwp_ecmwf) -> tuple[xr.Dataset, xr.Dataset]:
    """Create two NWP datasets with consecutive init times for concatenation"""
    ds_2 = ds_nwp_ecmwf.copy(deep=True)
    ds_2["init_time_utc"] = pd.date_range(
        start=ds_nwp_ecmwf.init_time_utc.values.max() + pd.Timedelta("6h"),
        freq="6h",
        periods=len(ds_nwp_ecmwf.init_time_utc),
    )
    return ds_nwp_ecmwf, ds_2


@pytest.fixture(scope="module")
def zarr_paths(session_tmp_path, concatable_nwp_like_data) -> dict[int, list[str]]:
    """Save the NWP datasets to zarr, keyed by zarr format"""
    paths: dict[int, list[str]] = {}
    for zarr_format in (2, 3):
        paths[zarr_format] = [
            f"{session_tmp_path}/nwp_like_data_{n}.zarr{zarr_format}"
            for n in range(len(concatable_nwp_like_data))
        ]
        for ds, path in zip(concatable_nwp_like_data, paths[zarr_format], strict=True):
            ds.to_zarr(path, zarr_format=zarr_format)
    return paths


@pytest.fixture(scope="module")
def tensorstore_datasets(zarr_paths) -> list[xr.Dataset]:
    """The two consecutive datasets, opened as tensorstore-backed datasets"""
    return [open_zarr_paths(path) for path in zarr_paths[2]]


@pytest.mark.parametrize("zarr_format", ZARR_FORMATS)
def test_open_single_zarr(zarr_paths, zarr_format):
    path = zarr_paths[zarr_format][0]
    # Check tensorstore version returns same results as dask version
    assert open_zarr_paths(path).compute().equals(xr.open_zarr(path).compute())


@pytest.mark.parametrize("zarr_format", ZARR_FORMATS)
def test_open_multi_zarrs(zarr_paths, zarr_format):
    paths = zarr_paths[zarr_format]
    ds_ts = open_zarr_paths(paths, concat_dim="init_time_utc")
    # Check tensorstore version returns same results as dask version
    ds_dask = xr.open_mfdataset(
        paths, concat_dim="init_time_utc", combine="nested", engine="zarr",
    )
    assert ds_ts.compute().equals(ds_dask.compute())


def test_open_multi_zarrs_requires_concat_dim(zarr_paths):
    with pytest.raises(ValueError, match="`concat_dim` must be specified"):
        open_zarr_paths(zarr_paths[2])


def test_concat_matches_xr_concat(tensorstore_datasets, zarr_paths):
    ds_ts = concat_tensorstore(tensorstore_datasets, concat_dim="init_time_utc")

    ds_numpy = xr.concat(
        [xr.open_zarr(path, chunks=None) for path in zarr_paths[2]],
        dim="init_time_utc",
    )
    assert ds_ts.compute().equals(ds_numpy)


def test_concat_stays_tensorstore_backed(tensorstore_datasets):
    """The concatenated data variables should still be lazy tensorstores"""
    ds_ts = concat_tensorstore(tensorstore_datasets, concat_dim="init_time_utc")

    # Raises TypeError if the variable has been materialised
    store = _extract_tensorstore(ds_ts["ECMWF_UK"])
    assert store.shape == ds_ts["ECMWF_UK"].shape


def test_concat_requires_multiple_datasets(tensorstore_datasets):
    with pytest.raises(ValueError, match="need at least two datasets"):
        concat_tensorstore(tensorstore_datasets[:1], concat_dim="init_time_utc")


def test_concat_requires_valid_concat_dim(tensorstore_datasets):
    with pytest.raises(ValueError, match="'not_a_dim' is not a dimension"):
        concat_tensorstore(tensorstore_datasets, concat_dim="not_a_dim")


def test_concat_rejects_mismatched_datasets(tensorstore_datasets):
    ds_1, ds_2 = tensorstore_datasets
    with pytest.raises(ValueError, match="data_vars"):
        concat_tensorstore(
            [ds_1, ds_2.rename({"ECMWF_UK": "other"})],
            concat_dim="init_time_utc",
        )


def test_concat_rejects_non_tensorstore_datasets(zarr_paths):
    datasets = [xr.open_zarr(path) for path in zarr_paths[2]]
    with pytest.raises(TypeError, match="expected TensorStore"):
        concat_tensorstore(datasets, concat_dim="init_time_utc")
