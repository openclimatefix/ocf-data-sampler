import numpy as np
import pytest

from ocf_data_sampler.datasets.pvnet.sample_index import (
    ConcurrentSampleIndex,
    SampleIndex,
    _sanitise_index,
)


@pytest.fixture
def t0_times():
    return np.array(["2023-01-01T00:00", "2023-01-01T00:30"], dtype="datetime64[ns]")


def test_sample_index_access_and_membership(t0_times):
    index = SampleIndex(t0=t0_times, location_id=np.array([1, 2], dtype=np.int64))
    assert len(index) == 2
    assert index[0] == (t0_times[0], 1)
    assert index[-len(index)] == (t0_times[0], 1)
    assert index[-1] == (t0_times[1], 2)
    assert index.contains(t0_times[1], 2)
    assert not index.contains(t0_times[1], 1)


def test_concurrent_index_access_and_membership(t0_times):
    index = ConcurrentSampleIndex(t0=t0_times)
    assert len(index) == 2
    assert index[0] == t0_times[0]
    assert index[-1] == t0_times[1]
    assert index.contains(t0_times[1])
    assert not index.contains(np.datetime64("2023-01-02", "ns"))


@pytest.mark.parametrize(
    ("idx", "error"), [(2, IndexError), (-3, IndexError), (True, TypeError), (0.5, TypeError)],
)
def test_invalid_index(idx, error):
    with pytest.raises(error):
        _sanitise_index(idx, n_samples=2)
