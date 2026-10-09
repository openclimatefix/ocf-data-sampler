"""Torch dataset for PVNet."""

from typing import Generic, TypeVar

import numpy as np
import xarray as xr
from torch.utils.data import Dataset, default_collate
from typing_extensions import override

from ocf_data_sampler.common.lightarray import LightDataArray
from ocf_data_sampler.config import load_yaml_configuration
from ocf_data_sampler.config.model import PVNetDataConfig
from ocf_data_sampler.datasets.cache import PickleCacheMixin
from ocf_data_sampler.datasets.pvnet.availability import (
    build_concurrent_sample_index,
    build_sample_index,
    validate_requested_periods,
)
from ocf_data_sampler.datasets.pvnet.loading import (
    add_source_coordinates,
    get_dataset_dict,
    get_locations,
)
from ocf_data_sampler.datasets.pvnet.materialise import materialise_data
from ocf_data_sampler.datasets.pvnet.preprocess import (
    build_normalisation_arrays,
    preprocess_dataset_dict,
)
from ocf_data_sampler.datasets.pvnet.sample import build_numpy_sample
from ocf_data_sampler.datasets.pvnet.sample_index import ConcurrentSampleIndex, SampleIndex
from ocf_data_sampler.datasets.pvnet.slicing import (
    reduce_spatial_extent_of_datasets,
    slice_datasets_by_space,
    slice_datasets_by_time,
)
from ocf_data_sampler.datasets.pvnet.types import Location, NumpySample, SourceDict, TensorBatch

TIndex = TypeVar("TIndex", SampleIndex, ConcurrentSampleIndex)



def xarray_to_lightarray_dict(
    dataset_dict: SourceDict[xr.DataArray],
) -> SourceDict[LightDataArray]:
    """Create a dictionary LightDataArrays from a dictionary of xarray datasets."""
    new_dataset_dict = {}
    for k, v in dataset_dict.items():
        if isinstance(v, dict):
            new_dataset_dict[k] = xarray_to_lightarray_dict(v)
        elif isinstance(v, xr.DataArray):
            new_dataset_dict[k] = LightDataArray.from_xarray(v)
        else:
            raise ValueError(f"Unexpected type ({type(v)})")
    return new_dataset_dict


class AbstractPVNetDataset(PickleCacheMixin, Dataset, Generic[TIndex]):
    """Abstract class for PVNet datasets."""

    def __init__(
        self,
        config_filename: str,
        time_periods: list[tuple[str | None, str | None]] | None = None,
        include_extra_metadata: bool = False,
        use_xarray: bool = True,
    ) -> None:
        """A generic torch Dataset for creating PVNet samples.

        Contains methods to find valid times and locations
        and process and combine these sources for various data inputs
        common to both standard and concurrent PVNet datasets.

        Args:
            config_filename: Path to the configuration file
            time_periods: Restrict sample t0 times to these (start, end) periods, with inclusive
                starts and exclusive ends. None uses all available times.
            include_extra_metadata: Whether to include non-essential metadata for each sample in the
                sample dict.
            use_xarray: Whether to use xarray.DataArray or LightDataArray as the underlying data
                structure when sampling

        """
        super().__init__()

        # Validate requested periods before loading sources to fail fast on invalid input
        if time_periods is not None:
            validate_requested_periods(time_periods)

        config = load_yaml_configuration(config_filename)

        self.locations = get_locations(
            config.sampling_grid.locations_csv_path,
            config.sampling_grid.exclude_location_ids,
        )

        location_ids = [loc.id for loc in self.locations]
        datasets_dict = get_dataset_dict(config, location_ids=location_ids)

        self.sample_index: TIndex = self._build_sample_index(
            datasets_dict, location_ids, config, time_periods,
        )
        add_source_coordinates(self.locations, datasets_dict)
        self.normalisation_arrays = build_normalisation_arrays(config)

        self.config = config
        self.include_extra_metadata = include_extra_metadata

        if use_xarray:
            self.datasets_dict = datasets_dict
        else:
            self.datasets_dict = xarray_to_lightarray_dict(datasets_dict)


    def __len__(self) -> int:
        """Return the number of samples in the dataset index."""
        return len(self.sample_index)

    @staticmethod
    def _build_sample_index(
        datasets_dict: SourceDict,
        location_ids: list[int],
        config: PVNetDataConfig,
        time_periods: list[tuple[str | None, str | None]] | None,
    ) -> TIndex:
        """Build the index appropriate to the concrete dataset."""
        raise NotImplementedError("Subclasses must implement sample index construction")


class PVNetDataset(AbstractPVNetDataset[SampleIndex]):
    """A torch Dataset for creating PVNet samples."""

    @override
    def __init__(
        self,
        config_filename: str,
        time_periods: list[tuple[str | None, str | None]] | None = None,
        include_extra_metadata: bool = False,
        use_xarray: bool = True,
    ) -> None:
        super().__init__(config_filename, time_periods, include_extra_metadata, use_xarray)
        # Construct a lookup for locations - useful for users to construct sample by location ID
        self.location_lookup = {loc.id: loc for loc in self.locations}

    @staticmethod
    @override
    def _build_sample_index(
        datasets_dict: SourceDict,
        location_ids: list[int],
        config: PVNetDataConfig,
        time_periods: list[tuple[str | None, str | None]] | None,
    ) -> SampleIndex:
        """Build an index of available t0 and location pairs."""
        return build_sample_index(datasets_dict, location_ids, config, time_periods)

    def _get_sample(self, t0: np.datetime64, location: Location) -> NumpySample:
        """Generate the PVNet sample for given coordinates.

        Args:
            t0: init-time for sample
            location: location for sample
        """
        sample_dict = slice_datasets_by_space(self.datasets_dict, location, self.config)
        sample_dict = slice_datasets_by_time(sample_dict, t0, self.config)
        sample_dict = materialise_data(sample_dict)
        sample_dict = preprocess_dataset_dict(
            sample_dict, t0, self.config, self.normalisation_arrays,
        )
        return build_numpy_sample(
            sample_dict, t0, location, self.config, self.include_extra_metadata,
        )

    @override
    def __getitem__(self, idx: int) -> NumpySample:

        # Get the coordinates of the sample
        t0, location_id = self.sample_index[idx]

        # Get location from location id
        location = self.location_lookup[location_id]

        return self._get_sample(t0, location)

    def get_sample(self, t0: np.datetime64, location_id: int) -> NumpySample:
        """Generate a sample for the given coordinates.

        Useful for users to generate specific samples.

        Args:
            t0: init-time for sample
            location_id: id for location
        """
        # Check if the requested sample is available
        if not self.sample_index.contains(t0, location_id):
            raise ValueError(
                f"Input t0 time '{t0!s}' and location id '{location_id}' "
                f"pair not in valid t0 and location pairs",
            )

        location = self.location_lookup[location_id]

        return self._get_sample(t0, location)


class PVNetConcurrentDataset(AbstractPVNetDataset[ConcurrentSampleIndex]):
    """A torch Dataset for creating concurrent PVNet location samples."""

    @override
    def __init__(
        self,
        config_filename: str,
        time_periods: list[tuple[str | None, str | None]] | None = None,
        include_extra_metadata: bool = False,
        use_xarray: bool = True,
    ) -> None:
        super().__init__(config_filename, time_periods, include_extra_metadata, use_xarray)

        self.datasets_dict = reduce_spatial_extent_of_datasets(
            self.datasets_dict,
            self.locations,
            self.config,
        )

    @staticmethod
    @override
    def _build_sample_index(
        datasets_dict: SourceDict,
        location_ids: list[int],
        config: PVNetDataConfig,
        time_periods: list[tuple[str | None, str | None]] | None,
    ) -> ConcurrentSampleIndex:
        """Build an index of t0 times available at every requested location."""
        return build_concurrent_sample_index(datasets_dict, config, time_periods)

    def _get_sample(self, t0: np.datetime64) -> TensorBatch:
        """Generate a concurrent PVNet sample for given init-time.

        Args:
            t0: init-time for sample
        """
        # Slice by time then load to avoid loading the data multiple times from disk
        sample_dict = slice_datasets_by_time(self.datasets_dict, t0, self.config)
        sample_dict = materialise_data(sample_dict)
        # Preprocessing is location-independent, so do it once before slicing per-location below
        sample_dict = preprocess_dataset_dict(
            sample_dict, t0, self.config, self.normalisation_arrays,
        )

        samples = []

        # Prepare sample for each location
        for location in self.locations:
            sliced_sample_dict = slice_datasets_by_space(sample_dict, location, self.config)
            numpy_sample = build_numpy_sample(
                sliced_sample_dict, t0, location, self.config, self.include_extra_metadata,
            )
            samples.append(numpy_sample)

        # Stack samples
        return default_collate(samples)

    @override
    def __getitem__(self, idx: int) -> TensorBatch:
        return self._get_sample(self.sample_index[idx])

    def get_sample(self, t0: np.datetime64) -> TensorBatch:
        """Generate a sample for the given init-time.

        Useful for users to generate specific samples.

        Args:
            t0: init-time for sample
        """
        # Check if the requested sample is available
        if not self.sample_index.contains(t0):
            raise ValueError(f"Input init time '{t0!s}' not in valid times")
        return self._get_sample(t0)
