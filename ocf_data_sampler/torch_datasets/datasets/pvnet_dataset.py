"""Torch dataset for UK PVNet."""

import pandas as pd
import xarray as xr
from torch.utils.data import Dataset
from typing_extensions import override

from ocf_data_sampler.config import Configuration, load_yaml_configuration
from ocf_data_sampler.load.gsp import get_gsp_boundaries
from ocf_data_sampler.load.load_dataset import get_dataset_dict
from ocf_data_sampler.numpy_sample import (
    convert_gsp_to_numpy_sample,
    convert_nwp_to_numpy_sample,
    convert_satellite_to_numpy_sample,
    convert_site_to_numpy_sample,
    encode_datetimes,
    make_sun_position_numpy_sample,
)
from ocf_data_sampler.numpy_sample.collate import stack_np_samples_into_batch
from ocf_data_sampler.numpy_sample.common_types import NumpyBatch, NumpySample
from ocf_data_sampler.numpy_sample.gsp import GSPSampleKey
from ocf_data_sampler.numpy_sample.nwp import NWPSampleKey
from ocf_data_sampler.select import (
    Location,
    fill_time_periods,
    find_contiguous_t0_periods,
    intersection_of_multiple_dataframes_of_periods,
)
from ocf_data_sampler.torch_datasets.datasets.picklecache import PickleCacheMixin
from ocf_data_sampler.torch_datasets.utils import (
    add_alterate_coordinate_projections,
    config_normalization_values_to_dicts,
    diff_nwp_data,
    fill_nans_in_arrays,
    find_valid_time_periods,
    merge_dicts,
    slice_datasets_by_space,
    slice_datasets_by_time,
)
from ocf_data_sampler.utils import minutes, tensorstore_compute

xr.set_options(keep_attrs=True)


def get_locations(
    gsp_ids: list[int] | None = None,
    version: str = "20220314",
    location_type: str = "gsp",
    site_dataset: xr.Dataset | None = None,
) -> list[Location]:
    """Get list of locations of all GSPs.

    Args:
        gsp_ids: List of GSP IDs to include. Defaults to all GSPs except national
        version: Version of GSP boundaries to use. Defaults to "20220314"
        location_type: Type of location to get. Options are "gsp" or "site"
        site_dataset: xarray dataset of sites. Required if location_type is "site"
    """
    locations = []

    if location_type == "gsp":
        df_gsp_loc = get_gsp_boundaries(version)

        # Default GSP IDs is all except national (gsp_id=0)
        if gsp_ids is None:
            gsp_ids = df_gsp_loc.index.values
            gsp_ids = gsp_ids[gsp_ids != 0]

        df_gsp_loc = df_gsp_loc.loc[gsp_ids]

        for gsp_id in gsp_ids:
            locations.append(
                Location(
                    x=df_gsp_loc.loc[gsp_id].x_osgb,
                    y=df_gsp_loc.loc[gsp_id].y_osgb,
                    coord_system="osgb",
                    id=int(gsp_id),
                ),
            )
    else:

        for site_id in site_dataset.site_id.values:

            site = site_dataset.sel(site_id=site_id)
            location = Location(
                id=site_id,
                x=site.longitude.values,
                y=site.latitude.values,
                coord_system="lon_lat",
            )
            locations.append(location)

    return locations


class AbstractPVNetDataset(PickleCacheMixin, Dataset):
    """Abstract class for PVNet datasets."""

    def __init__(
        self,
        config_filename: str,
        start_time: str | None = None,
        end_time: str | None = None,
        gsp_ids: list[int] | None = None,
    ) -> None:
        """A torch Dataset for creating PVNet UK samples.

        Args:
            config_filename: Path to the configuration file
            start_time: Limit the init-times to be after this
            end_time: Limit the init-times to be before this
            gsp_ids: List of GSP IDs to create samples for. Defaults to all
        """
        super().__init__()

        config = load_yaml_configuration(config_filename)

        # Get dataset depending on whether GSP or sites
        if config.input_data.gsp:
            datasets_dict = get_dataset_dict(config.input_data, gsp_ids=gsp_ids)
        else:
            datasets_dict = get_dataset_dict(config.input_data)

        # Get t0 times where all input data is available
        if config.input_data.gsp:
            valid_t0_times = self.find_valid_t0_times(datasets_dict, config)
            filter_times = valid_t0_times
        else:
            valid_t0_times = self.find_valid_t0_and_site_ids(datasets_dict, config)
            filter_times = valid_t0_times["t0"]

        # Filter t0 times to given range
        if start_time is not None:
            valid_t0_times = valid_t0_times[filter_times >= pd.Timestamp(start_time)]

        if end_time is not None:
            valid_t0_times = valid_t0_times[filter_times <= pd.Timestamp(end_time)]

        # Construct list of locations to sample from
        if config.input_data.gsp:
            locations = get_locations(
                gsp_ids=gsp_ids,
                version=config.input_data.gsp.boundaries_version,
                )
            primary_coords = "osgb"
        else:
            locations = get_locations(site_dataset=datasets_dict["site"], location_type="site")
            primary_coords = "lon_lat"

        self.locations = add_alterate_coordinate_projections(
            locations,
            datasets_dict,
            primary_coords=primary_coords,
        )

        self.valid_t0_times = valid_t0_times

        # Assign config and input data to self
        self.config = config
        self.datasets_dict = datasets_dict

        # Extract the normalisation values from the config for faster access
        means_dict, stds_dict = config_normalization_values_to_dicts(config)
        self.means_dict = means_dict
        self.stds_dict = stds_dict

    def process_and_combine_datasets(
        self,
        dataset_dict: dict,
        t0: pd.Timestamp,
        location: Location,
    ) -> NumpySample:
        """Normalise and convert data to numpy arrays.

        Args:
            dataset_dict: Dictionary of xarray datasets
            t0: init-time for sample
            location: location of the sample
        """
        numpy_modalities = [{"t0": t0.timestamp()}]

        if "nwp" in dataset_dict:
            nwp_numpy_modalities = {}

            for nwp_key, da_nwp in dataset_dict["nwp"].items():

                # Standardise and convert to NumpyBatch
                channel_means = self.means_dict["nwp"][nwp_key]
                channel_stds = self.stds_dict["nwp"][nwp_key]

                da_nwp = (da_nwp - channel_means) / channel_stds

                nwp_numpy_modalities[nwp_key] = convert_nwp_to_numpy_sample(da_nwp)

            # Combine the NWPs into NumpyBatch
            numpy_modalities.append({NWPSampleKey.nwp: nwp_numpy_modalities})

        if "sat" in dataset_dict:
            da_sat = dataset_dict["sat"]

            # Standardise and convert to NumpyBatch
            channel_means = self.means_dict["sat"]
            channel_stds = self.stds_dict["sat"]

            da_sat = (da_sat - channel_means) / channel_stds

            numpy_modalities.append(convert_satellite_to_numpy_sample(da_sat))

        if "gsp" in dataset_dict:
            gsp_config = self.config.input_data.gsp
            da_gsp = dataset_dict["gsp"]
            da_gsp = da_gsp / da_gsp.effective_capacity_mwp.values

            # Convert to NumpyBatch
            numpy_modalities.append(
                convert_gsp_to_numpy_sample(
                    da_gsp,
                    t0_idx=-gsp_config.interval_start_minutes / gsp_config.time_resolution_minutes,
                ),
            )

            # Add GSP location data

            osgb_x, osgb_y = location.in_coord_system("osgb")

            numpy_modalities.append(
                {
                    GSPSampleKey.gsp_id: location.id,
                    GSPSampleKey.x_osgb: osgb_x,
                    GSPSampleKey.y_osgb: osgb_y,
                },
            )

        if "site" in dataset_dict:
            da_sites = dataset_dict["site"]
            da_sites = da_sites / da_sites.capacity_kwp.values

            # Convert to NumpyBatch
            numpy_modalities.append(convert_site_to_numpy_sample(da_sites))

            # add datetime features
            datetimes = pd.DatetimeIndex(da_sites.time_utc.values)
            datetime_features = encode_datetimes(datetimes=datetimes)

            numpy_modalities.append(datetime_features)

        # Only add solar position if explicitly configured
        if self.config.input_data.solar_position is not None:
            solar_config = self.config.input_data.solar_position

            # Create datetime range for solar position calculation
            datetimes = pd.date_range(
                t0 + minutes(solar_config.interval_start_minutes),
                t0 + minutes(solar_config.interval_end_minutes),
                freq=minutes(solar_config.time_resolution_minutes),
            )

            if self.config.input_data.gsp:
                # Convert OSGB coordinates to lon/lat
                lon, lat = location.in_coord_system("lon_lat")

                # Calculate solar positions and add to modalities
                numpy_modalities.append(make_sun_position_numpy_sample(datetimes, lon, lat))
            else:
                numpy_modalities.append(
                    make_sun_position_numpy_sample(
                        datetimes,
                        da_sites.longitude.values,
                        da_sites.latitude.values,
                        ),
                    )

        # Combine all the modalities and fill NaNs
        combined_sample = merge_dicts(numpy_modalities)
        combined_sample = fill_nans_in_arrays(combined_sample, config=self.config)

        return combined_sample

    @staticmethod
    def find_valid_t0_times(datasets_dict: dict, config: Configuration) -> pd.DatetimeIndex:
        """Find the t0 times where all of the requested input data is available.

        Args:
            datasets_dict: A dictionary of input datasets
            config: Configuration file
        """
        valid_time_periods = find_valid_time_periods(datasets_dict, config)

        # Fill out the contiguous time periods to get the t0 times
        valid_t0_times = fill_time_periods(
            valid_time_periods,
            freq=minutes(config.input_data.gsp.time_resolution_minutes),
        )
        return valid_t0_times

    def find_valid_t0_and_site_ids(
        self,
        datasets_dict: dict,
        config: Configuration,
    ) -> pd.DataFrame:
        """Find the t0 times where all of the requested input data is available for sites.

        The idea is to
        1. Get valid time period for nwp and satellite
        2. For each site location, find valid periods for that location

        Args:
            datasets_dict: A dictionary of input datasets
            config: Configuration file
        """
        # Get valid time period for nwp and satellite
        datasets_without_site = {k: v for k, v in datasets_dict.items() if k != "site"}
        valid_time_periods = find_valid_time_periods(datasets_without_site, config)

        # Loop over each location in system id and obtain valid periods
        sites = datasets_dict["site"]
        site_ids = sites.site_id.values
        site_config = config.input_data.site
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


class PVNetDataset(AbstractPVNetDataset):
    """A torch Dataset for creating PVNet samples."""

    @override
    def __init__(
        self,
        config_filename: str,
        start_time: str | None = None,
        end_time: str | None = None,
        gsp_ids: list[int] | None = None,
    ) -> None:

        super().__init__(config_filename, start_time, end_time, gsp_ids)

        # Construct a lookup for locations - useful for users to construct sample by GSP ID
        location_lookup = {loc.id: loc for loc in self.locations}

        # Assign coords and indices to self
        self.location_lookup = location_lookup

    @override
    def __len__(self) -> int:
        if self.config.input_data.gsp:
            return len(self.locations)*len(self.valid_t0_times)
        # For sites all t0 and site combinations already present
        return len(self.valid_t0_times)

    def _get_sample(self, t0: pd.Timestamp, location: Location) -> NumpySample:
        """Generate the PVNet sample for given coordinates.

        Args:
            t0: init-time for sample
            location: location for sample
        """
        sample_dict = slice_datasets_by_space(self.datasets_dict, location, self.config)
        sample_dict = slice_datasets_by_time(sample_dict, t0, self.config)
        sample_dict = tensorstore_compute(sample_dict)
        sample_dict = diff_nwp_data(sample_dict, self.config)
        return self.process_and_combine_datasets(sample_dict, t0, location)

    @override
    def __getitem__(self, idx: int) -> NumpySample:

        # Get the coordinates of the sample
        if idx >= len(self):
            raise ValueError(f"Index {idx} out of range for dataset of length {len(self)}")

        if self.config.input_data.gsp:

            # t_index will be between 0 and len(self.valid_t0_times)-1
            t_index = idx % len(self.valid_t0_times)

            # For each location, there are len(self.valid_t0_times) possible samples
            loc_index = idx // len(self.valid_t0_times)

            location = self.locations[loc_index]
            t0 = self.valid_t0_times[t_index]
        else:
            # Get the coordinates of the sample
            t0, site_id = self.valid_t0_times.iloc[idx]

            # Get location from site id
            location = self.location_lookup[site_id]

        return self._get_sample(t0, location)

    def get_sample(self, t0: pd.Timestamp, id: int) -> NumpySample:
        """Generate a sample for the given coordinates.

        Useful for users to generate specific samples.

        Args:
            t0: init-time for sample
            id: ID for region/site
        """
        # Check the user has asked for a sample which we have the data for
        if t0 not in self.valid_t0_times:
            raise ValueError(f"Input init time '{t0!s}' not in valid times")
        if id not in self.location_lookup:
            raise ValueError(f"Input region/site '{id}' not known")

        location = self.location_lookup[id]

        return self._get_sample(t0, location)


class PVNetConcurrentDataset(AbstractPVNetDataset):
    """A torch Dataset for creating concurrent PVNet location samples."""

    @override
    def __len__(self) -> int:
        if self.config.input_data.gsp:
            return len(self.valid_t0_times)
        # site specific as t0s repeated for all sites
        return len(self.valid_t0_times)//len(self.locations)

    def _get_sample(self, t0: pd.Timestamp) -> NumpyBatch:
        """Generate a concurrent PVNet sample for given init-time.

        Args:
            t0: init-time for sample
        """
        # Slice by time then load to avoid loading the data multiple times from disk

        sample_dict = slice_datasets_by_time(self.datasets_dict, t0, self.config)
        sample_dict = tensorstore_compute(sample_dict)
        sample_dict = diff_nwp_data(sample_dict, self.config)

        samples = []

        # Prepare sample for each GSP
        for location in self.locations:
            sliced_sample_dict = slice_datasets_by_space(sample_dict, location, self.config)
            numpy_sample = self.process_and_combine_datasets(
                sliced_sample_dict,
                t0,
                location,
            )
            samples.append(numpy_sample)

        # Stack samples
        return stack_np_samples_into_batch(samples)

    @override
    def __getitem__(self, idx: int) -> NumpyBatch:
        if self.config.input_data.gsp:
            return self._get_sample(self.valid_t0_times[idx])
        # site specific as t0s repeated for all sites
        t_index = idx % len(self.valid_t0_times)//len(self.locations)
        return self._get_sample(self.valid_t0_times["t0"][t_index])

    def get_sample(self, t0: pd.Timestamp) -> NumpyBatch:
        """Generate a sample for the given init-time.

        Useful for users to generate specific samples.

        Args:
            t0: init-time for sample
        """
        # Check data is availablle for init-time t0
        if t0 not in self.valid_t0_times:
            raise ValueError(f"Input init time '{t0!s}' not in valid times")
        return self._get_sample(t0)
