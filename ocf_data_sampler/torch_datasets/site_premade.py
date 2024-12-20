from glob import glob
from torch.utils.data import Dataset

import xarray as xr

from ocf_data_sampler.torch_datasets.process_and_combine import process_and_combine_datasets
from ocf_data_sampler.config import load_yaml_configuration


def convert_from_dataset_to_dict_datasets(combined_dataset: xr.Dataset) -> dict[str, xr.DataArray]:
    """
    Convert a sample dataset to a dict of datasets

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


class SitesPreMadeSamplesDataset(Dataset):
    """Dataset to load pre-made netcdf samples"""

    def __init__(
        self,
        sample_dir,
        config_filename: str,
    ):
        """Dataset to load pre-made netcdf samples

        Args:
            sample_dir: Path to the directory of pre-saved samples.
        """
        self.sample_paths = glob(f"{sample_dir}/*.nc")
        self.config = load_yaml_configuration(config_filename)

    def __len__(self):
        return len(self.sample_paths)

    def __getitem__(self, idx):
        # open the sample
        ds = xr.open_dataset(self.sample_paths[idx])

        # convert to numpy
        sample = self.convert_netcdf_to_numpy_sample(ds)

        return sample

    def convert_netcdf_to_numpy_sample(self, ds: xr.Dataset) -> dict:
        """Convert a netcdf dataset to a numpy sample"""

        # convert the single dataset to a dict of arrays
        sample_dict = convert_from_dataset_to_dict_datasets(ds)  # this function is from ocf_datapipes

        # rename satellite to satellite actual # TODO this could be improves
        sample_dict["sat"] = sample_dict.pop("satellite")

        # process and combine the datasets
        sample = process_and_combine_datasets(
            dataset_dict=sample_dict, config=self.config, target_key="site"
        )

        # TODO think about normalization, maybe its done not in batch creation, maybe its done afterwards,
        #  to allow it to be flexible

        return sample
