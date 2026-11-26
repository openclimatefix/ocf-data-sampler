from unittest.mock import Mock

import numpy as np
import pytest

from ocf_data_sampler.common.time_utils import minutes
from ocf_data_sampler.select.dropout import choose_delay


@pytest.mark.parametrize(
    "random_value, expected_minutes",
    # The cumulative sum of these probabilities rounds to just below 1, so a random value of
    # 1 - 1e-16 is not below it
    [(0.0, 0), (0.06, 30), (0.7, 60), (1 - 1e-16, 60)],
)
def test_choose_delay(random_value, expected_minutes):
    """Weights select their own delays, with rounding falling back to the last delay."""
    minute_probabilities = {0: 0.06, 30: 0.57, 60: 0.37}

    rng = Mock(spec=np.random.Generator)
    rng.random.return_value = random_value

    assert choose_delay(minute_probabilities, rng) == minutes(expected_minutes)
