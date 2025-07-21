"""Torch dataset for sites."""

import numpy as np
import pandas as pd
import xarray as xr
from torch.utils.data import Dataset
from typing_extensions import override

from ocf_data_sampler.config import Configuration, load_yaml_configuration
from ocf_data_sampler.load.load_dataset import get_dataset_dict
from ocf_data_sampler.numpy_sample import (
    NWPSampleKey,
    convert_nwp_to_numpy_sample,
    convert_satellite_to_numpy_sample,
    convert_site_to_numpy_sample,
    encode_datetimes,
    make_sun_position_numpy_sample,
)
from ocf_data_sampler.numpy_sample.collate import stack_np_samples_into_batch
from ocf_data_sampler.numpy_sample.common_types import NumpySample
from ocf_data_sampler.select import (
    Location,
    fill_time_periods,
    find_contiguous_t0_periods,
    intersection_of_multiple_dataframes_of_periods,
)
from ocf_data_sampler.torch_datasets.utils import (
    channel_dict_to_dataarray,
    find_valid_time_periods,
    slice_datasets_by_space,
    slice_datasets_by_time,
)
from ocf_data_sampler.torch_datasets.utils.merge_and_fill_utils import (
    fill_nans_in_arrays,
    merge_dicts,
)
from ocf_data_sampler.utils import compute, minutes

xr.set_options(keep_attrs=True)


def get_locations(site_xr: xr.Dataset) -> list[Location]:
    """Get list of locations of all sites.

    Args:
        site_xr: xarray Dataset of site data
    """
    locations = []
    for site_id in site_xr.site_id.values:
        site = site_xr.sel(site_id=site_id)
        location = Location(
            id=site_id,
            x=site.longitude.values,
            y=site.latitude.values,
            coordinate_system="lon_lat",
        )
        locations.append(location)

    return locations


class SitesDataset(Dataset):
    """A torch Dataset for creating PVNet Site samples."""

    def __init__(
        self,
        config_filename: str,
        start_time: str | None = None,
        end_time: str | None = None,
    ) -> None:
        """A torch Dataset for creating PVNet Site samples.

        Args:
            config_filename: Path to the configuration file
            start_time: Limit the init-times to be after this
            end_time: Limit the init-times to be before this
        """
        config = load_yaml_configuration(config_filename)
        datasets_dict = get_dataset_dict(config.input_data)

        # Assign config and input data to self
        self.datasets_dict = datasets_dict
        self.config = config

        # get all locations
        self.locations = get_locations(datasets_dict["site"])
        self.location_lookup = {loc.id: loc for loc in self.locations}

        # Get t0 times where all input data is available
        valid_t0_and_site_ids = self.find_valid_t0_and_site_ids(datasets_dict)

        # Filter t0 times to given range
        if start_time is not None:
            valid_t0_and_site_ids = valid_t0_and_site_ids[
                valid_t0_and_site_ids["t0"] >= pd.Timestamp(start_time)
            ]

        if end_time is not None:
            valid_t0_and_site_ids = valid_t0_and_site_ids[
                valid_t0_and_site_ids["t0"] <= pd.Timestamp(end_time)
            ]

        # Assign coords and indices to self
        self.valid_t0_and_site_ids = valid_t0_and_site_ids

    def find_valid_t0_and_site_ids(
        self,
        datasets_dict: dict,
    ) -> pd.DataFrame:
        """Find the t0 times where all of the requested input data is available.

        The idea is to
        1. Get valid time period for nwp and satellite
        2. For each site location, find valid periods for that location

        Args:
            datasets_dict: A dictionary of input datasets
            config: Configuration file
        """
        # Get valid time period for nwp and satellite
        datasets_without_site = {k: v for k, v in datasets_dict.items() if k != "site"}
        valid_time_periods = find_valid_time_periods(datasets_without_site, self.config)

        # Loop over each location in system id and obtain valid periods
        sites = datasets_dict["site"]
        site_ids = sites.site_id.values
        site_config = self.config.input_data.site
        valid_t0_and_site_ids = []
        for site_id in site_ids:
            site = sites.sel(site_id=site_id)
            # Drop NaN values
            site = site.dropna(dim="time_utc")

            # Obtain valid time periods for this location
            time_periods = find_contiguous_t0_periods(
                pd.DatetimeIndex(site["time_utc"]),
                time_resolution=minutes(site_config.time_resolution_minutes),
                interval_start=minutes(site_config.interval_start_minutes),
                interval_end=minutes(site_config.interval_end_minutes),
            )
            valid_time_periods_per_site = intersection_of_multiple_dataframes_of_periods(
                [valid_time_periods, time_periods],
            )

            # Fill out contiguous time periods to get t0 times
            valid_t0_times_per_site = fill_time_periods(
                valid_time_periods_per_site,
                freq=minutes(site_config.time_resolution_minutes),
            )

            valid_t0_per_site = pd.DataFrame(index=valid_t0_times_per_site)
            valid_t0_per_site["site_id"] = site_id
            valid_t0_and_site_ids.append(valid_t0_per_site)

        valid_t0_and_site_ids = pd.concat(valid_t0_and_site_ids)
        valid_t0_and_site_ids.index.name = "t0"
        return valid_t0_and_site_ids.reset_index()

    @override
    def __len__(self) -> int:
        return len(self.valid_t0_and_site_ids)

    @override
    def __getitem__(self, idx: int) -> dict:
        # Get the coordinates of the sample
        t0, site_id = self.valid_t0_and_site_ids.iloc[idx]

        # get location from site id
        location = self.location_lookup[site_id]

        # Generate the sample
        return self._get_sample(t0, location)

    def _get_sample(self, t0: pd.Timestamp, location: Location) -> dict:
        """Generate the PVNet sample for given coordinates.

        Args:
            t0: init-time for sample
            location: location for sample
        """
        sample_dict = slice_datasets_by_space(self.datasets_dict, location, self.config)
        sample_dict = slice_datasets_by_time(sample_dict, t0, self.config)

        sample = self.process_and_combine_site_sample_dict(sample_dict, t0)
        return sample.compute()

    def get_sample(self, t0: pd.Timestamp, site_id: int) -> dict:
        """Generate a sample for a given site id and t0.

        Useful for users to generate samples by t0 and site id

        Args:
            t0: init-time for sample
            site_id: site id as int
        """
        location = self.location_lookup[site_id]

        return self._get_sample(t0, location)

    def process_and_combine_site_sample_dict(
        self,
        dataset_dict: dict,
        t0: pd.Timestamp,
    ) -> xr.Dataset:
        """Normalize and combine data into a single xr Dataset.

        Args:
            dataset_dict: dict containing sliced xr DataArrays
            t0: The initial timestamp of the sample

        Returns:
            xr.Dataset: A merged Dataset with nans filled in.
        """
        data_arrays = []

        if "nwp" in dataset_dict:
            for nwp_key, da_nwp in dataset_dict["nwp"].items():
                provider = self.config.input_data.nwp[nwp_key].provider

                da_channel_means = channel_dict_to_dataarray(
                    self.config.input_data.nwp[nwp_key].channel_means,
                )
                da_channel_stds = channel_dict_to_dataarray(
                    self.config.input_data.nwp[nwp_key].channel_stds,
                )

                da_nwp = (da_nwp - da_channel_means) / da_channel_stds
                data_arrays.append((f"nwp-{provider}", da_nwp))

        if "sat" in dataset_dict:
            da_sat = dataset_dict["sat"]

            da_channel_means = channel_dict_to_dataarray(
                self.config.input_data.satellite.channel_means,
            )
            da_channel_stds = channel_dict_to_dataarray(
                self.config.input_data.satellite.channel_stds,
            )

            da_sat = (da_sat - da_channel_means) / da_channel_stds
            data_arrays.append(("satellite", da_sat))

        if "site" in dataset_dict:
            da_sites = dataset_dict["site"]
            da_sites = da_sites / da_sites.capacity_kwp
            data_arrays.append(("site", da_sites))

        combined_sample_dataset = self.merge_data_arrays(data_arrays)

        # add datetime features
        datetimes = pd.DatetimeIndex(combined_sample_dataset.site__time_utc.values)
        datetime_features = encode_datetimes(datetimes=datetimes)
        combined_sample_dataset = combined_sample_dataset.assign_coords(
            {k: ("site__time_utc", v) for k, v in datetime_features.items()},
        )

        # Only add solar position if explicitly configured
        has_solar_config = (
            hasattr(self.config.input_data, "solar_position")
            and self.config.input_data.solar_position is not None
        )

        if has_solar_config:
            solar_config = self.config.input_data.solar_position

            # Datetime range - solar config params
            solar_datetimes = pd.date_range(
                t0 + minutes(solar_config.interval_start_minutes),
                t0 + minutes(solar_config.interval_end_minutes),
                freq=minutes(solar_config.time_resolution_minutes),
            )

            # Calculate sun position features
            sun_position_features = make_sun_position_numpy_sample(
                datetimes=solar_datetimes,
                lon=combined_sample_dataset.site__longitude.values,
                lat=combined_sample_dataset.site__latitude.values,
            )

            # Use existing dimension for solar positions
            # TODO decouple this as a separate data varaible
            solar_dim_name = "site__time_utc"

            # Assign solar position values
            for key, values in sun_position_features.items():
                combined_sample_dataset = combined_sample_dataset.assign_coords(
                    {key: (solar_dim_name, values)},
                )

        # TODO include t0_index in xr dataset?

        # Fill any nan values
        return combined_sample_dataset.fillna(0.0)

    def merge_data_arrays(
        self,
        normalised_data_arrays: list[tuple[str, xr.DataArray]],
    ) -> xr.Dataset:
        """Combine a list of DataArrays into a single Dataset with unique naming conventions.

        Args:
            normalised_data_arrays: List of tuples where each tuple contains:
                - A string (key name).
                - An xarray.DataArray.

        Returns:
            xr.Dataset: A merged Dataset with uniquely named variables, coordinates, and dimensions.
        """
        datasets = []

        for key, data_array in normalised_data_arrays:
            # Ensure all attributes are strings for consistency
            data_array = data_array.assign_attrs(
                {attr_key: str(attr_value) for attr_key, attr_value in data_array.attrs.items()},
            )

            # Convert DataArray to Dataset with the variable name as the key
            dataset = data_array.to_dataset(name=key)

            # Prepend key name to all dimension and coordinate names for uniqueness
            dataset = dataset.rename(
                {dim: f"{key}__{dim}" for dim in dataset.dims if dim not in dataset.coords},
            )
            dataset = dataset.rename(
                {coord: f"{key}__{coord}" for coord in dataset.coords},
            )

            # Handle concatenation dimension if applicable
            concat_dim = (
                f"{key}__target_time_utc"
                if f"{key}__target_time_utc" in dataset.coords
                else f"{key}__time_utc"
            )

            if f"{key}__init_time_utc" in dataset.coords:
                init_coord = f"{key}__init_time_utc"
                if dataset[init_coord].ndim == 0:  # Check if scalar
                    expanded_init_times = [dataset[init_coord].values] * len(dataset[concat_dim])
                    dataset = dataset.assign_coords({init_coord: (concat_dim, expanded_init_times)})

            datasets.append(dataset)

        # Ensure all datasets are valid xarray.Dataset objects
        for ds in datasets:
            if not isinstance(ds, xr.Dataset):
                raise ValueError(f"Object is not an xr.Dataset: {type(ds)}")

        # Merge all prepared datasets
        combined_dataset = xr.merge(datasets)

        return combined_dataset


class SitesDatasetConcurrent(Dataset):
    """A torch Dataset for creating PVNet Site batches with samples for all sites."""

    def __init__(
        self,
        config_filename: str,
        start_time: str | None = None,
        end_time: str | None = None,
    ) -> None:
        """A torch Dataset for creating PVNet Site samples.

        Args:
            config_filename: Path to the configuration file
            start_time: Limit the init-times to be after this
            end_time: Limit the init-times to be before this
        """
        config = load_yaml_configuration(config_filename)
        datasets_dict = get_dataset_dict(config.input_data)

        # Assign config and input data to self
        self.datasets_dict = datasets_dict
        self.config = config

        # get all locations
        self.locations = get_locations(datasets_dict["site"])

        # Get t0 times where all input data is available
        valid_t0s = self.find_valid_t0s(datasets_dict)

        # Filter t0 times to given range
        if start_time is not None:
            valid_t0s = valid_t0s[
                valid_t0s >= pd.Timestamp(start_time)
            ]

        if end_time is not None:
            valid_t0s = valid_t0s[
                valid_t0s <= pd.Timestamp(end_time)
            ]

        # Assign coords and indices to self
        self.valid_t0s = valid_t0s

    @staticmethod
    def process_and_combine_datasets(
        dataset_dict: dict,
        config: Configuration,
        t0: pd.Timestamp,
    ) -> NumpySample:
        """Normalise and convert data to numpy arrays.

        Args:
            dataset_dict: Dictionary of xarray datasets
            config: Configuration object
            t0: init-time for sample
        """
        numpy_modalities = []

        if "nwp" in dataset_dict:
            nwp_numpy_modalities = {}

            for nwp_key, da_nwp in dataset_dict["nwp"].items():
                # Standardise and convert to NumpyBatch

                da_channel_means = channel_dict_to_dataarray(
                    config.input_data.nwp[nwp_key].channel_means,
                )
                da_channel_stds = channel_dict_to_dataarray(
                    config.input_data.nwp[nwp_key].channel_stds,
                )

                da_nwp = (da_nwp - da_channel_means) / da_channel_stds

                nwp_numpy_modalities[nwp_key] = convert_nwp_to_numpy_sample(da_nwp)

            # Combine the NWPs into NumpyBatch
            numpy_modalities.append({NWPSampleKey.nwp: nwp_numpy_modalities})

        if "sat" in dataset_dict:
            da_sat = dataset_dict["sat"]

            # Standardise and convert to NumpyBatch
            da_channel_means = channel_dict_to_dataarray(config.input_data.satellite.channel_means)
            da_channel_stds = channel_dict_to_dataarray(config.input_data.satellite.channel_stds)

            da_sat = (da_sat - da_channel_means) / da_channel_stds

            numpy_modalities.append(convert_satellite_to_numpy_sample(da_sat))

        if "site" in dataset_dict:
            da_sites = dataset_dict["site"]
            da_sites = da_sites / da_sites.capacity_kwp

            # Convert to NumpyBatch
            numpy_modalities.append(convert_site_to_numpy_sample(da_sites))

        # Only add solar position if explicitly configured
        has_solar_config = (
            hasattr(config.input_data, "solar_position")
            and config.input_data.solar_position is not None
        )

        if has_solar_config:
            solar_config = config.input_data.solar_position

            # Create datetime range for solar position calculation
            datetimes = pd.date_range(
                t0 + minutes(solar_config.interval_start_minutes),
                t0 + minutes(solar_config.interval_end_minutes),
                freq=minutes(solar_config.time_resolution_minutes),
            )

            # Calculate solar positions and add to modalities
            numpy_modalities.append(
                make_sun_position_numpy_sample(
                    datetimes, da_sites.longitude.values, da_sites.latitude.values,
                ),
            )

        # Combine all the modalities and fill NaNs
        combined_sample = merge_dicts(numpy_modalities)
        combined_sample = fill_nans_in_arrays(combined_sample)

        return combined_sample


    def find_valid_t0s(
        self,
        datasets_dict: dict,
    ) -> pd.DataFrame:
        """Find the t0 times where all of the requested input data is available.

        The idea is to
        1. Get valid time period for nwp and satellite
        2. For the first site location, find valid periods for that location
        Note there is an assumption that all sites have the same t0 data available

        Args:
            datasets_dict: A dictionary of input datasets
        """
        # Get valid time period for nwp and satellite
        datasets_without_site = {k: v for k, v in datasets_dict.items() if k != "site"}
        valid_time_periods = find_valid_time_periods(datasets_without_site, self.config)
        sites = datasets_dict["site"]

        # Taking just the first site value, assume t0s the same for all of them
        site_id = sites.site_id.values[0]
        site_config = self.config.input_data.site
        site = sites.sel(site_id=site_id)
        # Drop NaN values
        site = site.dropna(dim="time_utc")

        # Obtain valid time periods for this location
        time_periods = find_contiguous_t0_periods(
            pd.DatetimeIndex(site["time_utc"]),
            time_resolution=minutes(site_config.time_resolution_minutes),
            interval_start=minutes(site_config.interval_start_minutes),
            interval_end=minutes(site_config.interval_end_minutes),
        )
        valid_time_periods_per_site = intersection_of_multiple_dataframes_of_periods(
            [valid_time_periods, time_periods],
        )

        # Fill out contiguous time periods to get t0 times
        valid_t0_times_per_site = fill_time_periods(
            valid_time_periods_per_site,
            freq=minutes(site_config.time_resolution_minutes),
        )

        return valid_t0_times_per_site

    @override
    def __len__(self) -> int:
        return len(self.valid_t0s)

    @override
    def __getitem__(self, idx: int) -> dict:
        # Get the coordinates of the sample
        t0 = self.valid_t0s[idx]

        return self._get_batch(t0)

    def _get_batch(self, t0: pd.Timestamp) -> dict:
        """Generate the PVNet batch for given coordinates.

        Args:
            t0: init-time for sample
        """
        # slice by time first as we want to keep all site id info
        sample_dict = slice_datasets_by_time(self.datasets_dict, t0, self.config)
        sample_dict = compute(sample_dict)

        site_samples = []

        for location in self.locations:
            site_sample_dict = slice_datasets_by_space(sample_dict, location, self.config)
            site_numpy_sample = self.process_and_combine_datasets(
                site_sample_dict,
                self.config,
                t0,
            )
            site_samples.append(site_numpy_sample)

        return stack_np_samples_into_batch(site_samples)


# ----- functions to load presaved samples ------


def convert_netcdf_to_numpy_sample(ds: xr.Dataset) -> dict:
    """Convert a netcdf dataset to a numpy sample.

    Args:
        ds: xarray Dataset
    """
    # convert the single dataset to a dict of arrays
    sample_dict = convert_from_dataset_to_dict_datasets(ds)

    if "satellite" in sample_dict:
        # rename satellite to sat # TODO this could be improved
        sample_dict["sat"] = sample_dict.pop("satellite")

    # process and combine the datasets
    sample = convert_to_numpy_and_combine(dataset_dict=sample_dict)

    # Add solar coord and datetime features
    keys = ["solar_azimuth", "solar_elevation", "date_sin", "date_cos", "time_sin", "time_cos"]
    for key in keys:
        if key in ds.coords:
            sample[key] = ds.coords[key].values

    # TODO think about normalization:
    # * maybe its done not in sample creation, maybe its done afterwards,
    #   to allow it to be flexible

    return sample


def convert_from_dataset_to_dict_datasets(combined_dataset: xr.Dataset) -> dict[str, xr.DataArray]:
    """Convert a combined sample dataset to a dict of datasets for each input.

    Args:
        combined_dataset: The combined NetCDF dataset

    Returns:
        The uncombined datasets as a dict of xr.Datasets
    """
    # Split into datasets by splitting by the prefix added in combine_to_netcdf
    datasets: dict[str, xr.DataArray] = {}

    # Go through each data variable and split it into a dataset
    for key, dataset in combined_dataset.items():
        # If 'key__' doesn't exist in a dim or coordinate, remove it
        for dim in list(dataset.coords):
            if f"{key}__" not in dim:
                dataset = dataset.drop_vars(dim)
        dataset = dataset.rename(
            {dim: dim.split(f"{key}__")[1] for dim in dataset.dims if dim not in dataset.coords},
        )
        dataset = dataset.rename(
            {coord: coord.split(f"{key}__")[1] for coord in dataset.coords},
        )
        # Split the dataset by the prefix
        datasets[key] = dataset

    # Unflatten any NWP data
    return nest_nwp_source_dict(datasets, sep="-")


def nest_nwp_source_dict(
    dataset_dict: dict[xr.Dataset],
    sep: str = "-",
) -> dict[str, xr.Dataset | dict[xr.Dataset]]:
    """Re-nest a dictionary where the NWP values are nested under keys 'nwp-<key>'.

    Args:
        dataset_dict: Dictionary of datasets
        sep: Separator to use to nest NWP keys
    """
    nwp_prefix = f"nwp{sep}"
    new_dict = {k: v for k, v in dataset_dict.items() if not k.startswith(nwp_prefix)}
    nwp_keys = [k for k in dataset_dict if k.startswith(nwp_prefix)]
    if len(nwp_keys) > 0:
        nwp_subdict = {k.removeprefix(nwp_prefix): dataset_dict[k] for k in nwp_keys}
        new_dict["nwp"] = nwp_subdict
    return new_dict


def convert_to_numpy_and_combine(dataset_dict: dict[xr.Dataset]) -> NumpySample:
    """Convert input data in a dict to numpy arrays.

    Args:
        dataset_dict: Dictionary of xarray Datasets
    """
    numpy_modalities = []

    if "nwp" in dataset_dict:
        nwp_numpy_modalities = {}
        for nwp_key, da_nwp in dataset_dict["nwp"].items():
            # Convert to NumpySample
            nwp_numpy_modalities[nwp_key] = convert_nwp_to_numpy_sample(da_nwp)

        # Combine the NWPs into NumpySample
        numpy_modalities.append({NWPSampleKey.nwp: nwp_numpy_modalities})

    if "sat" in dataset_dict:
        # Satellite is already in the range [0-1] so no need to standardise
        da_sat = dataset_dict["sat"]

        # Convert to NumpySample
        numpy_modalities.append(convert_satellite_to_numpy_sample(da_sat))

    if "site" in dataset_dict:
        da_sites = dataset_dict["site"]

        numpy_modalities.append(convert_site_to_numpy_sample(da_sites))

    # Combine all the modalities and fill NaNs
    combined_sample = merge_dicts(numpy_modalities)
    return fill_nans_in_arrays(combined_sample)


def coarsen_data(xr_data: xr.Dataset, coarsen_to_deg: float = 0.1) -> xr.Dataset:
    """Coarsen the data to a specified resolution in degrees.

    Args:
        xr_data: xarray dataset to coarsen
        coarsen_to_deg: resolution to coarsen to in degrees
    """
    if "latitude" in xr_data.coords and "longitude" in xr_data.coords:
        step = np.abs(xr_data.latitude.values[1] - xr_data.latitude.values[0])
        step = np.round(step, 4)
        coarsen_factor = int(coarsen_to_deg / step)
        if coarsen_factor > 1:
            xr_data = xr_data.coarsen(
                latitude=coarsen_factor,
                longitude=coarsen_factor,
                boundary="pad",
                coord_func="min",
            ).mean()

    return xr_data
