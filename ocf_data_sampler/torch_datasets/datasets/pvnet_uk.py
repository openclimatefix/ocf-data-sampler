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
    make_sun_position_numpy_sample,
)
from ocf_data_sampler.numpy_sample.collate import stack_np_samples_into_batch
from ocf_data_sampler.numpy_sample.common_types import NumpyBatch, NumpySample
from ocf_data_sampler.numpy_sample.gsp import GSPSampleKey
from ocf_data_sampler.numpy_sample.nwp import NWPSampleKey
from ocf_data_sampler.select import Location, fill_time_periods
from ocf_data_sampler.torch_datasets.utils import (
    add_alterate_coordinate_projections,
    config_normalization_values_to_dicts,
    fill_nans_in_arrays,
    find_valid_time_periods,
    merge_dicts,
    slice_datasets_by_space,
    slice_datasets_by_time,
)
from ocf_data_sampler.utils import minutes, tensorstore_compute

xr.set_options(keep_attrs=True)


def get_gsp_locations(
    gsp_ids: list[int] | None = None,
    version: str = "20220314",
) -> list[Location]:
    """Get list of locations of all GSPs.

    Args:
        gsp_ids: List of GSP IDs to include. Defaults to all GSPs except national
        version: Version of GSP boundaries to use. Defaults to "20220314"
    """
    df_gsp_loc = get_gsp_boundaries(version)

    # Default GSP IDs is all except national (gsp_id=0)
    if gsp_ids is None:
        gsp_ids = df_gsp_loc.index.values
        gsp_ids = gsp_ids[gsp_ids != 0]

    df_gsp_loc = df_gsp_loc.loc[gsp_ids]

    locations = []

    for gsp_id in gsp_ids:
        locations.append(
            Location(
                x=df_gsp_loc.loc[gsp_id].x_osgb,
                y=df_gsp_loc.loc[gsp_id].y_osgb,
                coord_system="osgb",
                id=int(gsp_id),
            ),
        )
    return locations


class AbstractPVNetUKDataset(Dataset):
    """Abstract class for PVNet UK datasets."""

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
        config = load_yaml_configuration(config_filename)
        datasets_dict = get_dataset_dict(config.input_data, gsp_ids=gsp_ids)

        # Get t0 times where all input data is available
        valid_t0_times = self.find_valid_t0_times(datasets_dict, config)

        # Filter t0 times to given range
        if start_time is not None:
            valid_t0_times = valid_t0_times[valid_t0_times >= pd.Timestamp(start_time)]

        if end_time is not None:
            valid_t0_times = valid_t0_times[valid_t0_times <= pd.Timestamp(end_time)]

        # Construct list of locations to sample from
        locations = get_gsp_locations(gsp_ids, version=config.input_data.gsp.boundaries_version)
        self.locations = add_alterate_coordinate_projections(
            locations,
            datasets_dict,
            primary_coords="osgb",
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
        numpy_modalities = []

        if "nwp" in dataset_dict:
            nwp_numpy_modalities = {}

            for nwp_key, da_nwp in dataset_dict["nwp"].items():

                # Standardise and convert to NumpyBatch
                da_channel_means = self.means_dict["nwp"][nwp_key]
                da_channel_stds = self.stds_dict["nwp"][nwp_key]

                da_nwp = (da_nwp - da_channel_means) / da_channel_stds

                nwp_numpy_modalities[nwp_key] = convert_nwp_to_numpy_sample(da_nwp)

            # Combine the NWPs into NumpyBatch
            numpy_modalities.append({NWPSampleKey.nwp: nwp_numpy_modalities})

        if "sat" in dataset_dict:
            da_sat = dataset_dict["sat"]

            # Standardise and convert to NumpyBatch
            da_channel_means = self.means_dict["sat"]
            da_channel_stds = self.stds_dict["sat"]

            da_sat = (da_sat - da_channel_means) / da_channel_stds

            numpy_modalities.append(convert_satellite_to_numpy_sample(da_sat))

        if "gsp" in dataset_dict:
            gsp_config = self.config.input_data.gsp
            da_gsp = dataset_dict["gsp"]
            da_gsp = da_gsp / da_gsp.effective_capacity_mwp

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

        # Only add solar position if explicitly configured
        if self.config.input_data.solar_position is not None:
            solar_config = self.config.input_data.solar_position

            # Create datetime range for solar position calculation
            datetimes = pd.date_range(
                t0 + minutes(solar_config.interval_start_minutes),
                t0 + minutes(solar_config.interval_end_minutes),
                freq=minutes(solar_config.time_resolution_minutes),
            )

            # Convert OSGB coordinates to lon/lat
            lon, lat = location.in_coord_system("lon_lat")

            # Calculate solar positions and add to modalities
            numpy_modalities.append(make_sun_position_numpy_sample(datetimes, lon, lat))

        # Combine all the modalities and fill NaNs
        combined_sample = merge_dicts(numpy_modalities)
        combined_sample = fill_nans_in_arrays(combined_sample)

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



class PVNetUKRegionalDataset(AbstractPVNetUKDataset):
    """A torch Dataset for creating PVNet UK regional samples."""

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
        return len(self.locations)*len(self.valid_t0_times)

    def _get_sample(self, t0: pd.Timestamp, location: Location) -> NumpySample:
        """Generate the PVNet sample for given coordinates.

        Args:
            t0: init-time for sample
            location: location for sample
        """
        sample_dict = slice_datasets_by_space(self.datasets_dict, location, self.config)
        sample_dict = slice_datasets_by_time(sample_dict, t0, self.config)
        sample_dict = tensorstore_compute(sample_dict)

        return self.process_and_combine_datasets(sample_dict, t0, location)

    @override
    def __getitem__(self, idx: int) -> NumpySample:
        # Get the coordinates of the sample

        idx = int(idx)

        if idx >= len(self):
            raise ValueError(f"Index {idx} out of range for dataset of length {len(self)}")

        # t_index will be between 0 and len(self.valid_t0_times)-1
        t_index = idx % len(self.valid_t0_times)

        # For each location, there are len(self.valid_t0_times) possible samples
        loc_index = idx // len(self.valid_t0_times)

        location = self.locations[loc_index]
        t0 = self.valid_t0_times[t_index]

        return self._get_sample(t0, location)

    def get_sample(self, t0: pd.Timestamp, gsp_id: int) -> NumpySample:
        """Generate a sample for the given coordinates.

        Useful for users to generate specific samples.

        Args:
            t0: init-time for sample
            gsp_id: GSP ID
        """
        # Check the user has asked for a sample which we have the data for
        if t0 not in self.valid_t0_times:
            raise ValueError(f"Input init time '{t0!s}' not in valid times")
        if gsp_id not in self.location_lookup:
            raise ValueError(f"Input GSP '{gsp_id}' not known")

        location = self.location_lookup[gsp_id]

        return self._get_sample(t0, location)


class PVNetUKConcurrentDataset(AbstractPVNetUKDataset):
    """A torch Dataset for creating concurrent PVNet UK regional samples."""

    @override
    def __len__(self) -> int:
        return len(self.valid_t0_times)

    def _get_sample(self, t0: pd.Timestamp) -> NumpyBatch:
        """Generate a concurrent PVNet sample for given init-time.

        Args:
            t0: init-time for sample
        """
        # Slice by time then load to avoid loading the data multiple times from disk
        sample_dict = slice_datasets_by_time(self.datasets_dict, t0, self.config)
        sample_dict = tensorstore_compute(sample_dict)

        gsp_samples = []

        # Prepare sample for each GSP
        for location in self.locations:
            gsp_sample_dict = slice_datasets_by_space(sample_dict, location, self.config)
            gsp_numpy_sample = self.process_and_combine_datasets(
                gsp_sample_dict,
                t0,
                location,
            )
            gsp_samples.append(gsp_numpy_sample)

        # Stack GSP samples
        return stack_np_samples_into_batch(gsp_samples)

    @override
    def __getitem__(self, idx: int) -> NumpyBatch:
        return self._get_sample(self.valid_t0_times[idx])

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
