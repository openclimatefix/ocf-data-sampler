"""Functions for randomly dropping out sequential data."""

import numpy as np

from ocf_data_sampler.common.types import TArray


def apply_dropout(
    da: TArray,
    t0: np.datetime64,
    dropout_timedeltas: list[np.timedelta64],
    dropout_frac: float | list[float],
) -> TArray:
    """Apply in-place random dropout to some sequence data.

    A timedelta relative to t0 is randomly sampled, and all data after that point in time is
    dropped out (replaced with NaNs).

    This helper requires a NumPy-backed DataArray-like object. It mutates the backing array in
    place, so it should be called after any lazy data has been materialised.

    Args:
        da: DataArray-like with 'time_utc' coordinate
        t0: The forecast init-time.
        dropout_timedeltas: List of timedeltas relative to t0 to pick from
        dropout_frac: The probabilit(ies) that each dropout timedelta will be applied. This should
            be between 0 and 1 inclusive.
    """
    if len(dropout_timedeltas)==0:
        return da

    if not isinstance(da.data, np.ndarray):
        raise ValueError(
            f"Dropout can only be applied to DataArrays with numpy data. Got: {type(da.data)}"
        )

    if isinstance(dropout_frac, float | int):

        if not (0<=dropout_frac<=1):
            raise ValueError("`dropout_frac` must be in range [0, 1]")

        # Create list with equal chance for all dropout timedeltas
        n = len(dropout_timedeltas)
        dropout_frac = [dropout_frac/n for _ in range(n)]
    else:
        if not 0<=sum(dropout_frac)<=1:
            raise ValueError("The sum of `dropout_frac` must be in range [0, 1]")
        if len(dropout_timedeltas)!=len(dropout_frac):
            raise ValueError("`dropout_timedeltas` and `dropout_frac` must have the same length")

        dropout_frac = [*dropout_frac] # Make copy of the list so we can append to it

    dropout_timedeltas = [*dropout_timedeltas] # Make copy of the list so we can append to it

    # Add chance of no dropout
    dropout_frac.append(1-sum(dropout_frac))
    dropout_timedeltas.append(None)

    timedelta_choice = np.random.choice(dropout_timedeltas, p=dropout_frac)

    if timedelta_choice is None:
        return da
    else:
        times = da["time_utc"].values
        keep = times <= t0 + timedelta_choice
        axis = da.dims.index("time_utc")
        mask = np.expand_dims(keep, tuple(i for i in range(da.data.ndim) if i != axis))
        mask = np.broadcast_to(mask, da.data.shape)

        da.data = np.where(mask, da.data, np.nan)
        return da

