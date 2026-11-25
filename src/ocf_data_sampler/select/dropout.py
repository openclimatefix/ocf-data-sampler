"""Functions for randomly delaying data availability."""

from collections.abc import Mapping

import numpy as np

from ocf_data_sampler.common.time_utils import minutes


def choose_delay(delay_minute_probabilities: Mapping[int, float]) -> np.timedelta64:
    """Draw a non-negative availability delay from validated minute/probability pairs.

    Iterating the mapping avoids allocating choice and probability arrays for every sample.

    Args:
        delay_minute_probabilities: Non-empty mapping of non-negative integer delays in
            minutes to positive probabilities summing to 1. Inputs must already be validated.

    Returns:
        The chosen delay.

    """
    draw = np.random.random()
    cumulative_probability = 0.0
    for delay_minutes, probability in delay_minute_probabilities.items():
        cumulative_probability += probability
        if draw < cumulative_probability:
            return minutes(delay_minutes)
    # If rounding leaves the cumulative probability just below the draw, the last delay is used
    return minutes(delay_minutes)
