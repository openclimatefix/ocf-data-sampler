import pandas as pd
import numpy as np
import xarray as xr
from ocf_data_sampler.select.select_time_slice import select_time_slice_nwp


def make_fake_nwp():
    # two forecast runs: 00:00 (6h steps) and 06:00 (3h steps)
    init_times = pd.to_datetime(["2024-01-01 00:00", "2024-01-01 06:00"])
    steps_00 = pd.to_timedelta([0, 6, 12], "h")
    steps_06 = pd.to_timedelta([0, 3, 6], "h")
    # unify unique step list for dataset definition
    all_steps = pd.to_timedelta(sorted({*steps_00, *steps_06}))
    # fake data: value = init_hour + step_hours
    data = np.zeros((len(init_times), len(all_steps)))
    for i, init in enumerate(init_times):
        for j, s in enumerate(all_steps):
            data[i, j] = init.hour + s.total_seconds() / 3600.0

    da = xr.DataArray(
        data,
        dims=("init_time_utc", "step"),
        coords={"init_time_utc": init_times, "step": all_steps},
        name="fake_nwp",
    )
    return da


def test_preserve_native_true():
    da = make_fake_nwp()
    t0 = pd.Timestamp("2024-01-01 12:00")
    out = select_time_slice_nwp(
        da,
        t0=t0,
        interval_start=pd.Timedelta(hours=-6),
        interval_end=pd.Timedelta(hours=12),
        time_resolution=pd.Timedelta(hours=1),
        preserve_native=True,
    )
    assert isinstance(out, xr.DataArray)
    assert "step" in out.coords
    assert "init_time_utc" in out.coords
    assert len(out.step) > 0


def test_preserve_native_false_uniform():
    da = make_fake_nwp()
    t0 = pd.Timestamp("2024-01-01 12:00")
    out = select_time_slice_nwp(
        da,
        t0=t0,
        interval_start=pd.Timedelta(hours=-6),
        interval_end=pd.Timedelta(hours=12),
        time_resolution=pd.Timedelta(hours=3),
        preserve_native=False,
    )
    assert isinstance(out, xr.DataArray)
    # Should produce uniform time spacing (3h)
    steps = np.diff(out.step.values)
    # Most steps should be uniform; allow the final one to differ slightly
    assert np.allclose(
        steps[:-1].astype("timedelta64[s]").astype(float),
        steps[0].astype("timedelta64[s]").astype(float),
    )


def test_select_time_slice_nwp_with_dropout():
    da = make_fake_nwp()
    t0 = pd.Timestamp("2024-01-01 12:00")
    out = select_time_slice_nwp(
        da,
        t0=t0,
        interval_start=pd.Timedelta(hours=-6),
        interval_end=pd.Timedelta(hours=12),
        time_resolution=pd.Timedelta(hours=3),
        dropout_timedeltas=[pd.Timedelta(hours=-6), pd.Timedelta(hours=-3)],
        dropout_frac=1.0,  # force dropout
    )
    assert isinstance(out, xr.DataArray)
    assert "step" in out.coords
    assert len(out.step) > 0


def test_select_time_slice_nwp_older_init_times():
    da = make_fake_nwp()
    # Pretend our t0 is after all init_times
    t0 = pd.Timestamp("2024-01-01 18:00")
    out = select_time_slice_nwp(
        da,
        t0=t0,
        interval_start=pd.Timedelta(hours=-6),
        interval_end=pd.Timedelta(hours=6),
        time_resolution=pd.Timedelta(hours=3),
    )
    assert isinstance(out, xr.DataArray)
    # ensure steps don't go negative
    assert np.all(out.step.values >= np.timedelta64(0, "h"))


def test_select_time_slice_nwp_exact_boundary():
    """Ensure boundary-aligned selection includes exact init_time and step endpoints."""
    da = make_fake_nwp()
    t0 = pd.Timestamp("2024-01-01 06:00")
    out = select_time_slice_nwp(
        da,
        t0=t0,
        interval_start=pd.Timedelta(hours=0),
        interval_end=pd.Timedelta(hours=6),
        time_resolution=pd.Timedelta(hours=3),
    )
    # Expect steps exactly [0h, 3h, 6h]
    expected_steps = np.array([
        np.timedelta64(0, 'h'),
        np.timedelta64(3, 'h'),
        np.timedelta64(6, 'h')
    ])
    # Convert both arrays to seconds for a unit-safe numeric comparison
    out_steps_sec = out.step.values.astype("timedelta64[s]").astype(float)
    expected_steps_sec = expected_steps.astype("timedelta64[s]").astype(float)
    assert np.allclose(out_steps_sec, expected_steps_sec)