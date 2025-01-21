"""Torch dataset for sites"""
import logging

import pandas as pd
import xarray as xr
from typing import Tuple

from torch.utils.data import Dataset

from ocf_data_sampler.config import Configuration, load_yaml_configuration
from ocf_data_sampler.load.load_dataset import get_dataset_dict
from ocf_data_sampler.select import (
    Location,
    fill_time_periods,
    find_contiguous_t0_periods,
    intersection_of_multiple_dataframes_of_periods,
    slice_datasets_by_time, slice_datasets_by_space
)
from ocf_data_sampler.utils import minutes
from ocf_data_sampler.torch_datasets.utils.valid_time_periods import find_valid_time_periods
from ocf_data_sampler.torch_datasets.utils.merge_and_fill_utils import merge_dicts, fill_nans_in_arrays
from ocf_data_sampler.numpy_sample import (
    convert_site_to_numpy_sample, 
    convert_satellite_to_numpy_sample, 
    convert_nwp_to_numpy_sample,
    make_datetime_numpy_dict,
    make_sun_position_numpy_sample,
)
from ocf_data_sampler.numpy_sample import NWPSampleKey
from ocf_data_sampler.constants import NWP_MEANS, NWP_STDS

xr.set_options(keep_attrs=True)

class SitesDataset(Dataset):
    def __init__(
        self,
        config_filename: str,
        start_time: str | None = None,
        end_time: str | None = None,
    ):
        """A torch Dataset for creating PVNet Site samples

        Args:
            config_filename: Path to the configuration file
            start_time: Limit the init-times to be after this
            end_time: Limit the init-times to be before this
        """

        config: Configuration = load_yaml_configuration(config_filename)
        datasets_dict = get_dataset_dict(config)

        # Assign config and input data to self
        self.datasets_dict = datasets_dict
        self.config = config

        # get all locations
        self.locations = self.get_locations(datasets_dict['site'])

        # Get t0 times where all input data is available
        valid_t0_and_site_ids = self.find_valid_t0_and_site_ids(datasets_dict)

        # Filter t0 times to given range
        if start_time is not None:
            valid_t0_and_site_ids \
                = valid_t0_and_site_ids[valid_t0_and_site_ids['t0'] >= pd.Timestamp(start_time)]

        if end_time is not None:
            valid_t0_and_site_ids \
                = valid_t0_and_site_ids[valid_t0_and_site_ids['t0'] <= pd.Timestamp(end_time)]

        # Assign coords and indices to self
        self.valid_t0_and_site_ids = valid_t0_and_site_ids

    def __len__(self):
        return len(self.valid_t0_and_site_ids)
    
    def __getitem__(self, idx):

        # Get the coordinates of the sample
        t0, site_id = self.valid_t0_and_site_ids.iloc[idx]

        # get location from site id
        location = self.get_location_from_site_id(site_id)

        # Generate the sample
        return self._get_sample(t0, location)

    def _get_sample(self, t0: pd.Timestamp, location: Location) -> dict:
        """Generate the PVNet sample for given coordinates

        Args:
            t0: init-time for sample
            location: location for sample
        """
        sample_dict = slice_datasets_by_space(self.datasets_dict, location, self.config)
        sample_dict = slice_datasets_by_time(sample_dict, t0, self.config)

        sample = self.process_and_combine_site_sample_dict(sample_dict)
        sample = sample.compute()
        return sample

    def get_sample(self, t0: pd.Timestamp, site_id: int) -> dict:
        """Generate a sample for a given site id and t0.

        Useful for users to generate samples by t0 and site id

        Args:
            t0: init-time for sample
            site_id: site id as int
        """

        location = self.get_location_from_site_id(site_id)

        return self._get_sample(t0, location)
    
    def get_location_from_site_id(self, site_id):
        """Get location from system id"""

        locations = [loc for loc in self.locations if loc.id == site_id]
        if len(locations) == 0:
            raise ValueError(f"Location not found for site_id {site_id}")

        if len(locations) > 1:
            logging.warning(f"Multiple locations found for site_id {site_id}, but will take the first")

        return locations[0]

    def find_valid_t0_and_site_ids(
        self,
        datasets_dict: dict,
    ) -> pd.DataFrame:
        """Find the t0 times where all of the requested input data is available

        The idea is to
        1. Get valid time period for nwp and satellite
        2. For each site location, find valid periods for that location

        Args:
            datasets_dict: A dictionary of input datasets
            config: Configuration file
        """

        # 1. Get valid time period for nwp and satellite
        datasets_without_site = {k:v for k, v in datasets_dict.items() if k!="site"}
        valid_time_periods = find_valid_time_periods(datasets_without_site, self.config)

        # 2. Now lets loop over each location in system id and find the valid periods
        # Should we have a different option if there are not nans
        sites = datasets_dict["site"]
        site_ids = sites.site_id.values
        site_config = self.config.input_data.site
        valid_t0_and_site_ids = []
        for site_id in site_ids:
            site = sites.sel(site_id=site_id)

            # drop any nan values
            # not sure this is right?
            site = site.dropna(dim='time_utc')

            # Get the valid time periods for this location
            time_periods = find_contiguous_t0_periods(
                pd.DatetimeIndex(site["time_utc"]),
                sample_period_duration=minutes(site_config.time_resolution_minutes),
                interval_start=minutes(site_config.interval_start_minutes),
                interval_end=minutes(site_config.interval_end_minutes),
            )
            valid_time_periods_per_site = intersection_of_multiple_dataframes_of_periods(
                [valid_time_periods, time_periods]
            )

            # Fill out the contiguous time periods to get the t0 times
            valid_t0_times_per_site = fill_time_periods(
                valid_time_periods_per_site,
                freq=minutes(site_config.time_resolution_minutes)
            )

            valid_t0_per_site = pd.DataFrame(index=valid_t0_times_per_site)
            valid_t0_per_site['site_id'] = site_id
            valid_t0_and_site_ids.append(valid_t0_per_site)

        valid_t0_and_site_ids = pd.concat(valid_t0_and_site_ids)
        valid_t0_and_site_ids.index.name = 't0'
        valid_t0_and_site_ids.reset_index(inplace=True)

        return valid_t0_and_site_ids


    def get_locations(self, site_xr: xr.Dataset):
        """Get list of locations of all sites"""

        locations = []
        for site_id in site_xr.site_id.values:
            site = site_xr.sel(site_id=site_id)
            location = Location(
                id=site_id,
                x=site.longitude.values,
                y=site.latitude.values,
                coordinate_system="lon_lat"
            )
            locations.append(location)

        return locations

    def process_and_combine_site_sample_dict(
        self,
        dataset_dict: dict,
    ) -> xr.Dataset:
        """
        Normalize and combine data into a single xr Dataset

        Args:
            dataset_dict: dict containing sliced xr DataArrays
            config: Configuration for the model

        Returns:
            xr.Dataset: A merged Dataset with nans filled in.
        
        """

        data_arrays = []

        if "nwp" in dataset_dict:
            for nwp_key, da_nwp in dataset_dict["nwp"].items():
                # Standardise
                provider = self.config.input_data.nwp[nwp_key].provider
                da_nwp = (da_nwp - NWP_MEANS[provider]) / NWP_STDS[provider]
                data_arrays.append((f"nwp-{provider}", da_nwp))
            
        if "sat" in dataset_dict:
            # TODO add some satellite normalisation
            da_sat = dataset_dict["sat"]
            data_arrays.append(("satellite", da_sat))

        if "site" in dataset_dict:
            # site_config = config.input_data.site
            da_sites = dataset_dict["site"]
            da_sites = da_sites / da_sites.capacity_kwp
            data_arrays.append(("site", da_sites))

        combined_sample_dataset = self.merge_data_arrays(data_arrays)

        # add datetime features
        datetimes = pd.DatetimeIndex(combined_sample_dataset.site__time_utc.values)
        datetime_features = make_datetime_numpy_dict(datetimes=datetimes, key_prefix="site")
        datetime_features_xr = xr.Dataset(datetime_features, coords={"site__time_utc": datetimes})
        combined_sample_dataset = xr.merge([combined_sample_dataset, datetime_features_xr])

        # add sun features
        sun_position_features = make_sun_position_numpy_sample(
            datetimes=datetimes,
            lon=combined_sample_dataset.site__longitude.values,
            lat=combined_sample_dataset.site__latitude.values,
            key_prefix="site",
        )
        sun_position_features_xr = xr.Dataset(
            sun_position_features, coords={"site__time_utc": datetimes}
        )
        combined_sample_dataset = xr.merge([combined_sample_dataset, sun_position_features_xr])

        # TODO include t0_index in xr dataset? 

        # Fill any nan values
        return combined_sample_dataset.fillna(0.0)


    def merge_data_arrays(self, normalised_data_arrays: list[Tuple[str, xr.DataArray]]) -> xr.Dataset:
        """
        Combine a list of DataArrays into a single Dataset with unique naming conventions.

        Args:
            list_of_arrays: List of tuples where each tuple contains:
                - A string (key name).
                - An xarray.DataArray.

        Returns:
            xr.Dataset: A merged Dataset with uniquely named variables, coordinates, and dimensions.
        """
        datasets = []

        for key, data_array in normalised_data_arrays:
            # Ensure all attributes are strings for consistency
            data_array = data_array.assign_attrs(
                {attr_key: str(attr_value) for attr_key, attr_value in data_array.attrs.items()}
            )

            # Convert DataArray to Dataset with the variable name as the key
            dataset = data_array.to_dataset(name=key)

            # Prepend key name to all dimension and coordinate names for uniqueness
            dataset = dataset.rename(
                {dim: f"{key}__{dim}" for dim in dataset.dims if dim not in dataset.coords}
            )
            dataset = dataset.rename(
                {coord: f"{key}__{coord}" for coord in dataset.coords}
            )

            # Handle concatenation dimension if applicable
            concat_dim = (
                f"{key}__target_time_utc" if f"{key}__target_time_utc" in dataset.coords
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
            assert isinstance(ds, xr.Dataset), f"Object is not an xr.Dataset: {type(ds)}"

        # Merge all prepared datasets
        combined_dataset = xr.merge(datasets)

        return combined_dataset

# ----- functions to load presaved samples ------

def convert_netcdf_to_numpy_sample(ds: xr.Dataset) -> dict:
    """Convert a netcdf dataset to a numpy sample"""

    # convert the single dataset to a dict of arrays
    sample_dict = convert_from_dataset_to_dict_datasets(ds) 

    if "satellite" in sample_dict:
        # rename satellite to satellite actual # TODO this could be improves
        sample_dict["sat"] = sample_dict.pop("satellite")

    # process and combine the datasets
    sample = convert_to_numpy_and_combine(
        dataset_dict=sample_dict,
    )

    # TODO think about normalization, maybe its done not in sample creation, maybe its done afterwards,
    #  to allow it to be flexible

    return sample

def convert_from_dataset_to_dict_datasets(combined_dataset: xr.Dataset) -> dict[str, xr.DataArray]:
    """
    Convert a combined sample dataset to a dict of datasets for each input

    Args:
        combined_dataset: The combined NetCDF dataset

    Returns:
        The uncombined datasets as a dict of xr.Datasets
    """
    # Split into datasets by splitting by the prefix added in combine_to_netcdf
    datasets = {}
    # Go through each data variable and split it into a dataset
    for key, dataset in combined_dataset.items():
        # If 'key_' doesn't exist in a dim or coordinate, remove it
        dataset_dims = list(dataset.coords)
        for dim in dataset_dims:
            if f"{key}__" not in dim:
                dataset: xr.Dataset = dataset.drop(dim)
        dataset = dataset.rename(
            {dim: dim.split(f"{key}__")[1] for dim in dataset.dims if dim not in dataset.coords}
        )
        dataset: xr.Dataset = dataset.rename(
            {coord: coord.split(f"{key}__")[1] for coord in dataset.coords}
        )
        # Split the dataset by the prefix
        datasets[key] = dataset

    # Unflatten any NWP data
    datasets = nest_nwp_source_dict(datasets, sep="-")
    return datasets


def nest_nwp_source_dict(d: dict, sep: str = "/") -> dict:
    """Re-nest a dictionary where the NWP values are nested under keys 'nwp/<key>'."""
    nwp_prefix = f"nwp{sep}"
    new_dict = {k: v for k, v in d.items() if not k.startswith(nwp_prefix)}
    nwp_keys = [k for k in d.keys() if k.startswith(nwp_prefix)]
    if len(nwp_keys) > 0:
        nwp_subdict = {k.removeprefix(nwp_prefix): d[k] for k in nwp_keys}
        new_dict["nwp"] = nwp_subdict
    return new_dict

def convert_to_numpy_and_combine(
    dataset_dict: dict,
) -> dict:
    """Convert input data in a dict to numpy arrays"""

    numpy_modalities = []

    if "nwp" in dataset_dict:

        nwp_numpy_modalities = dict()
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

        numpy_modalities.append(
            convert_site_to_numpy_sample(
                da_sites,
            )
        )

    # Combine all the modalities and fill NaNs
    combined_sample = merge_dicts(numpy_modalities)
    combined_sample = fill_nans_in_arrays(combined_sample)

    return combined_sample
