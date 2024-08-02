from ocf_data_sampler.select.select_time_slice import select_time_slice
from ocf_data_sampler.load.satellite import open_sat_data

from datetime import timedelta
import numpy as np


def test_select_time_slice(sat_zarr_path):
    sat = open_sat_data(sat_zarr_path)
    print(sat)
    t0 = sat.time_utc[3].values

    sat_sample = select_time_slice(
        ds=sat,
        t0=t0,
        sample_period_duration=timedelta(minutes=5),
        history_duration=timedelta(minutes=5),
        forecast_duration=timedelta(minutes=5),
    )

    assert len(sat_sample.time_utc) == 3
    assert sat_sample.time_utc[1] == t0


# TODO could to test with intervals, but we might want to remove this functionaility


def test_select_time_slice_out_of_bounds(sat_zarr_path):
    sat = open_sat_data(sat_zarr_path)
    t0 = sat.time_utc[-1].values

    sat_sample = select_time_slice(
        ds=sat,
        t0=t0,
        sample_period_duration=timedelta(minutes=5),
        history_duration=timedelta(minutes=5),
        forecast_duration=timedelta(minutes=5),
        fill_selection=True,
    )

    print(sat_sample.time_utc.values)

    assert len(sat_sample.time_utc) == 3
    assert sat_sample.time_utc[1] == t0
    # Correct number of time steps are all NaN
    sat_sel = sat_sample.isel(x_geostationary=0, y_geostationary=0, channel=0)
    assert np.isnan(sat_sel.values).sum() == 1


