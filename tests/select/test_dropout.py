import numpy as np
import pandas as pd
import pytest
import xarray as xr

from ocf_data_sampler.common.time_utils import minutes
from ocf_data_sampler.select.dropout import apply_dropout, choose_delay


@pytest.mark.parametrize("draw, expected", [(0.0, 30), (0.2, 60), (0.5, None)])
def test_sample_delay(monkeypatch, draw, expected):
    """Weights select their own delays, with remaining probability meaning no dropout."""
    probabilities = {30: 0.2, 60: 0.3}
    monkeypatch.setattr(np.random, "random", lambda: draw)

    delay = choose_delay(probabilities)

    assert delay == (None if expected is None else minutes(expected))


def test_apply_dropout_none(da_sample):
    original = da_sample.copy(deep=True)
    apply_dropout(da_sample, da_sample["time_utc"].values[-1], None)
    xr.testing.assert_equal(da_sample, original)


@pytest.mark.parametrize("delay_minutes", [0, 30])
def test_apply_dropout(da_sample, delay_minutes):
    da_sample = da_sample.copy(deep=True)
    t0 = np.datetime64("2024-01-01T12:30")
    cutoff = pd.Timestamp(t0 - minutes(delay_minutes))
    original = da_sample.copy(deep=True)

    apply_dropout(da_sample, t0, minutes(delay_minutes))

    # Keep the cutoff itself; mask every later observation, including any beyond t0.
    xr.testing.assert_equal(
        da_sample.sel(time_utc=slice(None, cutoff)), original.sel(time_utc=slice(None, cutoff)),
    )
    assert da_sample.sel(time_utc=slice(cutoff + minutes(5), None)).isnull().all()
    assert da_sample.dtype == original.dtype
