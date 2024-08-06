from pathlib import Path
import xarray as xr


def open_zarr_paths(
        zarr_path: Path | str | list[Path] | list[str], 
        time_dim: str = "init_time"
    ) -> xr.Dataset:
    """Opens the NWP data

    Args:
        zarr_path: Path to the zarr(s) to open
        time_dim: Name of the time dimension

    Returns:
        The opened Xarray Dataset
    """
    if type(zarr_path) in [list, tuple] or "*" in str(zarr_path):  # Multi-file dataset
        ds = xr.open_mfdataset(
            zarr_path,
            engine="zarr",
            concat_dim=time_dim,
            combine="nested",
            chunks="auto",
        ).sortby(time_dim)
    else:
        ds = xr.open_dataset(
            zarr_path,
            engine="zarr",
            consolidated=True,
            mode="r",
            chunks="auto",
        )
    return ds
