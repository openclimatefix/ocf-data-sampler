import os
import pickle

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader, default_collate

from ocf_data_sampler.common.lightarray import LightDataArray
from ocf_data_sampler.config import save_yaml_configuration
from ocf_data_sampler.datasets.pvnet.dataset import PVNetConcurrentDataset, PVNetDataset
from tests.conftest import LOCATION_IDS, save_zarr, to_datetime64


def _assert_feature_dtype(value, name):
    assert isinstance(value, np.ndarray | torch.Tensor), f"{name}: {type(value)}"
    expected_dtype = np.float32 if isinstance(value, np.ndarray) else torch.float32
    assert value.dtype == expected_dtype, f"{name}: {value.dtype}"


def _assert_sample_features(sample, batch_dim=()):
    """Assert a sample from the test config has float32 features with the expected shapes."""
    assert "t0" in sample

    expected_shapes = {
        # 30 minutes of 5-minute data (inclusive), one channel, 2x2 pixels
        "satellite": (7, 1, 2, 2),
        # 1 hour and 2 hours of 30-minute data (inclusive)
        "generation_input": (3,),
        "generation_target": (5,),
        "generation_input_capacity_mwp": (3,),
        "generation_target_capacity_mwp": (5,),
        # 3 periods, each with a sin and cos embedding
        "t0_embedding": (6,),
        # Solar and datetime features: -15 to +15 minutes of 5-minute data (inclusive)
        "solar_azimuth": (7,),
        "solar_elevation": (7,),
        "date_sin": (7,),
        "date_cos": (7,),
        "time_sin": (7,),
        "time_cos": (7,),
    }
    for key, shape in expected_shapes.items():
        value = sample[key]
        # Check the shape of the feature, including any batch dimension
        assert value.shape == (*batch_dim, *shape), f"{key}: {value.shape}"

        _assert_feature_dtype(value, key)

    # 3 hours of 60-minute data (inclusive), one channel, 2x2 pixels
    nwp = sample["nwp"]["ukv"]
    assert nwp.shape == (*batch_dim, 4, 1, 2, 2), f"nwp.ukv: {nwp.shape}"
    _assert_feature_dtype(nwp, "nwp.ukv")


def _assert_samples_equal(sample0, sample1):
    assert sample0.keys() == sample1.keys()
    for k in sample0:
        if isinstance(sample0[k], dict):
            _assert_samples_equal(sample0[k], sample1[k])
        elif isinstance(sample0[k], np.ndarray | torch.Tensor):
            np.testing.assert_array_equal(np.asarray(sample0[k]), np.asarray(sample1[k]))
        else:
            assert sample0[k] == sample1[k]


def _expected_num_locations(dataset, catalog_ids):
    """The catalogued locations which survive the config's exclusion list."""
    return len(catalog_ids) - len(dataset.config.sampling_grid.exclude_location_ids)


def _extract_backing_data(da):
    # Inspect the backing object without triggering a lazy read
    return da.data if isinstance(da, LightDataArray) else da.variable._data


@pytest.mark.parametrize("source", ["satellite", "ukv"])
def test_pvnet_dataset_rejects_oversized_crops(pvnet_config, source):
    config = pvnet_config
    source_config = {"satellite": config.satellite, "ukv": config.nwp["ukv"]}[source]
    # Both fixture grids are narrower than 201 pixels.
    source_config.image_width_pixels = 201

    with pytest.raises(ValueError, match="Cannot slice all locations"):
        PVNetDataset(config)


def test_pvnet_dataset(tmp_path, pvnet_config):
    # Datasets also accept a path to a YAML configuration
    config_path = str(tmp_path / "configuration.yaml")
    save_yaml_configuration(pvnet_config, config_path)
    dataset = PVNetDataset(
        config_path,
        time_periods=[
            ("2023-01-01 06:00", "2023-01-01 07:00"),
            ("2023-01-01 12:00", "2023-01-01 13:00"),
        ],
    )

    # Requested periods include their start and exclude their end
    expected_t0s = to_datetime64(
        ["2023-01-01 06:00", "2023-01-01 06:30", "2023-01-01 12:00", "2023-01-01 12:30"]
    )
    num_locs = _expected_num_locations(dataset, LOCATION_IDS)

    assert len(dataset.locations) == num_locs

    np.testing.assert_array_equal(np.unique(dataset.sample_index.t0), expected_t0s)
    assert len(dataset) == num_locs * len(expected_t0s)

    sample = dataset[0]

    _assert_sample_features(sample)


def test_pvnet_concurrent_dataset(tmp_path, pvnet_config):
    config = pvnet_config

    # Reverse the CSV rows so their order differs from the generation source. This is so we can
    # check that the location ordering comes from the locations CSV as the source of truth.
    locations_path = str(tmp_path / "reordered_locations.csv")

    df_locations = pd.read_csv(
        config.sampling_grid.locations_csv_path, dtype={"location_id": str}
    ).iloc[::-1]
    df_locations.to_csv(locations_path, index=False)

    config.sampling_grid.locations_csv_path = locations_path

    expected_locs = [
        location_id
        for location_id in df_locations["location_id"]
        if location_id not in config.sampling_grid.exclude_location_ids
    ]

    concurrent_dataset = PVNetConcurrentDataset(config, include_extra_metadata=True)
    # Valid t0s run from 03:00 to 22:00 at 30-minute intervals, giving 39 samples:
    # - Start: the first NWP init time is 00:00 and NWP data has a 3-hour delay
    # - End: generation data ends at 00:00 the next day and the target window ends at t0 + 2 hours
    t0s = concurrent_dataset.sample_index.t0
    assert len(concurrent_dataset) == 39
    assert (t0s[0], t0s[-1]) == tuple(to_datetime64(["2023-01-01 03:00", "2023-01-01 22:00"]))

    sample = concurrent_dataset[0]
    _assert_sample_features(sample, (len(expected_locs),))

    # Check locations all present and correctly ordered
    assert sample["location_id"] == expected_locs

    # Check that PVNetConcurrentDataset agrees with PVNetDataset for all locations.
    single_location_dataset = PVNetDataset(config, include_extra_metadata=True)
    t0 = t0s[0]
    location_samples = [
        single_location_dataset.get_sample(t0, location_id) for location_id in expected_locs
    ]
    _assert_samples_equal(sample, default_collate(location_samples))


def test_pvnet_dataset_without_generation(pvnet_config):
    """Test that a dataset can be built with locations/NWP/satellite but no generation at all."""
    config = pvnet_config
    config.generation = None

    dataset = PVNetDataset(config)

    assert len(dataset.locations) == _expected_num_locations(dataset, LOCATION_IDS)

    sample = dataset[0]
    assert "generation_input" not in sample
    assert "generation_target" not in sample
    assert "nwp" in sample
    assert "satellite" in sample


def test_pvnet_concurrent_dataset_minimal_dropout(pvnet_config):
    """Minimal mode samples a t0 where the most recent satellite data is missing"""
    config = pvnet_config
    # Without a target, generation is available for t0s up to 2023-01-02 00:00
    fill_value = -1.0
    config.generation.target = None
    config.satellite.fill_value = fill_value
    config.satellite.reject_nan = False
    config.satellite.dropout_minute_probabilities = {30: 0.3, 15: 0.3, 0: 0.4}

    dataset = PVNetConcurrentDataset(config, dropout_mode="minimal")

    # Satellite data ends at 23:55, but the 30-minute cutoff only needs data up to 23:30
    t0 = np.datetime64("2023-01-02T00:00")
    assert dataset.sample_index.t0[-1] == t0

    satellite = dataset.get_sample(t0)["satellite"].numpy()
    # The satellite data at the first 4 requested timestamps (23:30-23:45) is available
    assert (satellite[:, 0:4] != fill_value).all()
    # Satellite data IS available at the requested times 23:50 - 23:55. However, the minimal
    # configured dropout compatible with the available data is 15 minutes so everything after 23:45
    # is dropped and filled with the fill value.
    assert (satellite[:, 4:] == fill_value).all()


def test_pvnet_concurrent_dataset_minimal_dropout_generation_nan(
    tmp_path, pvnet_config, ds_generation
):
    """Minimal mode treats a NaN generation value at any location as missing data"""
    t0 = np.datetime64("2023-01-01T23:30", "ns")

    # One included location is NaN 60 minutes before t0, while its timestamp remains in the data
    ds_generation = ds_generation.copy(deep=True)
    nan_coords = {"time_utc": t0 - np.timedelta64(60, "m"), "location_id": "loc_1"}
    ds_generation["generation_mw"].loc[nan_coords] = np.nan
    generation_zarr_path = save_zarr(ds_generation, tmp_path, "generation_with_nan.zarr")

    config = pvnet_config
    fill_value = -1.0
    config.generation.zarr_path = generation_zarr_path
    config.generation.target = None
    config.generation.input.interval_start_minutes = -120
    config.generation.input.fill_value = fill_value
    config.generation.input.reject_nan = False
    config.generation.input.dropout_minute_probabilities = {30: 0.5, 90: 0.5}

    dataset = PVNetConcurrentDataset(config, dropout_mode="minimal")
    generation_input = dataset.get_sample(t0)["generation_input"].numpy()

    # Requested times are 21:30-23:30. The NaN at 22:30 rules out the 30-minute dropout, so the
    # 90-minute dropout keeps only 21:30 and 22:00 at every location
    assert (generation_input[:, :2] != fill_value).all()
    assert (generation_input[:, 2:] == fill_value).all()


@pytest.mark.parametrize("dataset_class", [PVNetDataset, PVNetConcurrentDataset])
def test_array_backends_produce_identical_samples(pvnet_config, dataset_class):
    config = pvnet_config
    # Set dropout so there is only one possible dropout choice
    config.satellite.dropout_minute_probabilities = {5: 1.0}
    config.satellite.fill_value = -1.0
    config.satellite.reject_nan = False
    config.generation.input.dropout_minute_probabilities = {30: 1.0}
    config.generation.input.fill_value = -1.0
    config.generation.input.reject_nan = False

    kwargs = {"config": config, "include_extra_metadata": True}
    xarray_dataset = dataset_class(**kwargs, array_backend="xarray")
    lightarray_dataset = dataset_class(**kwargs, array_backend="lightarray")

    # Check equal for the the start index, end index, and a few indices in the middle
    for index in np.linspace(0, len(xarray_dataset) - 1, num=5, dtype=int):
        _assert_samples_equal(xarray_dataset[index], lightarray_dataset[index])


@pytest.mark.parametrize("dataset_class", [PVNetDataset, PVNetConcurrentDataset])
@pytest.mark.parametrize("array_backend", ["xarray", "lightarray"])
def test_sampling_leaves_dataset_sources_unchanged(pvnet_config, dataset_class, array_backend):
    """Sampling must not load the dataset's lazy sources or write to its generation data."""
    dataset = dataset_class(pvnet_config, array_backend=array_backend)
    sources = [dataset.source_data["sat"], *dataset.source_data["nwp"].values()]
    lazy_data = [_extract_backing_data(da) for da in sources]
    generation = dataset.source_data["generation"].values.copy()

    dataset[0]

    # Loading in place through shared arrays would replace the dataset's lazy stores
    assert all(
        _extract_backing_data(da) is data for da, data in zip(sources, lazy_data, strict=True)
    )
    # Generation is in memory, so in-place writes through views would change these values
    np.testing.assert_array_equal(dataset.source_data["generation"].values, generation)


@pytest.mark.parametrize("array_backend", ["xarray", "lightarray"])
def test_pvnet_dataset_dataloader(pvnet_config, array_backend):
    dataset = PVNetDataset(pvnet_config, array_backend=array_backend)
    dataloader = DataLoader(
        dataset, batch_size=2, num_workers=2, multiprocessing_context="forkserver"
    )
    batch = next(iter(dataloader))

    _assert_sample_features(batch, batch_dim=(2,))
    _assert_samples_equal(batch, default_collate([dataset[0], dataset[1]]))


def test_pvnet_dataset_pickle(tmp_path, pvnet_config):
    pickle_path = str(tmp_path / "dataset.pkl")
    dataset = PVNetDataset(pvnet_config)
    sample = dataset[0]

    dataset.presave_pickle(pickle_path)
    # Once presaved, the dataset is pickled by reference to the file only
    pickle_bytes = pickle.dumps(dataset)
    assert dataset.__getstate__() == {"_pickle_path": pickle_path}

    # Check we can reload the object and that the first sample is the same as before
    unpickled_dataset = pickle.loads(pickle_bytes)  # noqa: S301
    _assert_samples_equal(unpickled_dataset[0], sample)

    # Unpickling must fail clearly if the presaved state file has gone missing
    os.remove(pickle_path)
    with pytest.raises(FileNotFoundError, match="Presaved state file not found"):
        _ = pickle.loads(pickle_bytes)  # noqa: S301

    # Check we can still pickle and unpickle if we don't presave
    dataset = PVNetDataset(pvnet_config)
    pickle_bytes = pickle.dumps(dataset)
    unpickled_dataset = pickle.loads(pickle_bytes)  # noqa: S301
    _assert_samples_equal(unpickled_dataset[0], sample)
