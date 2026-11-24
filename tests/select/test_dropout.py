import numpy as np
import pytest

from ocf_data_sampler.common.time_utils import minutes
from ocf_data_sampler.select.dropout import choose_delay


@pytest.mark.parametrize("draw, expected", [(0.0, 30), (0.2, 60), (0.5, None)])
def test_sample_delay(monkeypatch, draw, expected):
    """Weights select their own delays, with remaining probability meaning no dropout."""
    probabilities = {30: 0.2, 60: 0.3}
    monkeypatch.setattr(np.random, "random", lambda: draw)

    delay = choose_delay(probabilities)

    assert delay == (None if expected is None else minutes(expected))
