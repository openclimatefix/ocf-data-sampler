"""Select a time slice from a Dataset or DataArray."""

import numpy as np
import pandas as pd
import xarray as xr


def select_time_slice(
    da: xr.DataArray,
    t0: pd.Timestamp,
    interval_start: pd.Timedelta,
    interval_end: pd.Timedelta,
    time_resolution: pd.Timedelta,
) -> xr.DataArray:
    """Select a time slice from a DataArray.

    Args:
        da: The DataArray to slice from
        t0: The init-time
        interval_start: The start of the interval with respect to t0
        interval_end: The end of the interval with respect to t0
        time_resolution: Distance between neighbouring timestamps
    """
    start_dt = t0 + interval_start
    end_dt = t0 + interval_end

    start_dt = start_dt.ceil(time_resolution)
    end_dt = end_dt.ceil(time_resolution)

    return da.sel(time_utc=slice(start_dt, end_dt))



def select_time_slice_nwp(
    da: xr.DataArray,
    t0: pd.Timestamp,
    interval_start: pd.Timedelta,
    interval_end: pd.Timedelta,
    time_resolution: pd.Timedelta,
    dropout_timedeltas: list[pd.Timedelta] | None = None,
    dropout_frac: float | None = 0,
    *,
    preserve_native: bool = False,
) -> xr.DataArray:
    """Select a time slice from an NWP DataArray.

    Notes
    -----
    This function supports two behaviours:

    - Default (preserve_native=False): behave like the original function but
      derive `available_init_times` from the data and, for a uniform
      `pd.date_range(start, end, freq=time_resolution)`, pick the most
      recent init_time <= target_time for each target time. This avoids
      misalignment when some runs have coarser steps.

    - preserve_native=True: derive `target_times` from all (init_time + step)
      combinations available in `da`, deduplicate identical target times by
      keeping the sample produced by the most recent init_time, and then
      select accordingly. This returns mixed-resolution target times.

    Args:
        da: The DataArray to slice from (must contain coords 'init_time_utc' and 'step').
        t0: The init-time
        interval_start: The start of the interval with respect to t0
        interval_end: The end of the interval with respect to t0
        time_resolution: Distance between neighbouring timestamps (used for ceil and for
                         the default uniform target generation)
        dropout_timedeltas: List of possible timedeltas before t0 where data availability may start
        dropout_frac: Probability to apply dropout
        preserve_native: If True, return mixed-resolution target times derived from the data.
    """
    # Input checking
    if dropout_timedeltas is None:
        dropout_timedeltas = []

    if len(dropout_timedeltas) > 0:
        if not all(t < pd.Timedelta(0) for t in dropout_timedeltas):
            raise ValueError("dropout timedeltas must be negative")
        if len(dropout_timedeltas) < 1:
            raise ValueError("dropout timedeltas must have at least one element")

    if not (0 <= dropout_frac <= 1):
        raise ValueError("dropout_frac must be between 0 and 1")

    consider_dropout = len(dropout_timedeltas) > 0 and dropout_frac > 0

    # compute start/end (ceil to time_resolution for consistency)
    start_dt = (t0 + interval_start).ceil(time_resolution)
    end_dt = (t0 + interval_end).ceil(time_resolution)

    # Potentially apply NWP dropout
    if consider_dropout and (np.random.uniform() < dropout_frac):
        t0_available = t0 + np.random.choice(dropout_timedeltas)
    else:
        t0_available = t0

    # get available init_times from da (ensure pandas Timestamps for comparisons)
    init_times_np = da.init_time_utc.values
    # convert to pandas DTI for easy masking (handle numpy.datetime64)
    init_times = pd.to_datetime(init_times_np)

    # compute t_min: earliest init_time that could contribute to the requested window
    # use the largest step available (last element) as in original code
    max_step = da.step.values[-1]
    # convert max_step to pandas Timedelta if necessary
    if not isinstance(max_step, pd.Timedelta):
        max_step = pd.to_timedelta(max_step)
    t_min = pd.Timestamp(start_dt) - pd.Timedelta(max_step)

    # select available init times in window [t_min, t0_available]
    available_mask = (init_times >= t_min) & (init_times <= pd.Timestamp(t0_available))
    available_init_times = init_times[available_mask]

    if len(available_init_times) == 0:
        # Nothing available - raise or return empty selection (match prior behaviour? here raise)
        raise ValueError("No available init_times found in requested window")

    # MODE A: preserve_native == False (default behaviour, uniform target grid)
    if not preserve_native:
        # Uniform target times (same behaviour as original, but using available_init_times)
        target_times = pd.date_range(start_dt, end_dt, freq=time_resolution)

        # For each target time, find the most recent available_init_time <= target_time
        # (this naturally picks the latest init_time when duplicates would occur)
        try:
            selected_init_times = np.array(
                [available_init_times[available_init_times <= pd.Timestamp(t)][-1] for t in target_times],
                dtype="datetime64[ns]",
            )
        except IndexError as exc:
            # This matches previous behaviour where an IndexError could happen;
            # surface a helpful message
            raise IndexError(
                "Unable to find an available init_time for one or more target times. "
                "This may occur when requested start/end are earlier than available init_times."
            ) from exc

        # compute steps (numpy datetime64 arithmetic produces timedelta64)
        steps = pd.to_datetime(target_times).to_numpy() - pd.to_datetime(selected_init_times).to_numpy()
        # ensure steps have same dtype as da.step (xarray likely stores as numpy.timedelta64)
        # Now perform selection (fast path if single init_time)
        if len(np.unique(selected_init_times)) == 1:
            da_sel = da.sel(init_time_utc=selected_init_times[0], step=slice(steps[0], steps[-1]))
        else:
            coords = {"step": steps}
            init_time_indexer = xr.DataArray(selected_init_times, coords=coords)
            step_indexer = xr.DataArray(steps, coords=coords)
            da_sel = da.sel(init_time_utc=init_time_indexer, step=step_indexer)

        return da_sel

    # MODE B: preserve_native == True
    # Build all (init_time + step) candidates from available_init_times and da.step
    records: list[tuple[pd.Timestamp, pd.Timestamp, pd.Timedelta]] = []
    steps_arr = da.step.values
    # convert steps to pd.Timedelta for consistent arithmetic
    steps_td = [pd.to_timedelta(s) for s in steps_arr]

    for init in available_init_times:
        for s in steps_td:
            tgt = pd.Timestamp(init) + s
            records.append((tgt, pd.Timestamp(init), s))

    if len(records) == 0:
        raise ValueError("No candidate target times could be constructed from available init times and steps")

    # sort by target time asc, then init_time desc so the most recent init for a given target is first when deduping
    records.sort(key=lambda r: (r[0], -int(r[1].timestamp())))

    # dedupe by target time keeping the first (which has the most recent init_time due to sort)
    deduped = {}
    for tgt, init, s in records:
        if tgt not in deduped:
            deduped[tgt] = (init, s)

    # get ordered lists and filter to requested [start_dt, end_dt]
    ordered_targets = sorted(deduped.keys())
    ordered_targets = [t for t in ordered_targets if (pd.Timestamp(start_dt) <= t <= pd.Timestamp(end_dt))]

    if len(ordered_targets) == 0:
        raise ValueError("No target times within requested range after deduplication")

    selected_init_times = np.array([np.datetime64(deduped[t][0]) for t in ordered_targets], dtype="datetime64[ns]")
    steps = np.array([np.timedelta64(int(deduped[t][1].total_seconds()), "s") for t in ordered_targets], dtype="timedelta64[s]")

    # selection: if single init_time then we can slice, otherwise use vectorised indexing
    if len(np.unique(selected_init_times)) == 1:
        da_sel = da.sel(init_time_utc=selected_init_times[0], step=slice(steps[0], steps[-1]))
    else:
        coords = {"step": steps}
        init_time_indexer = xr.DataArray(selected_init_times, coords=coords)
        step_indexer = xr.DataArray(steps, coords=coords)
        da_sel = da.sel(init_time_utc=init_time_indexer, step=step_indexer)

    return da_sel
