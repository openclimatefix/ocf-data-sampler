"""Functions for randomly delaying data availability."""

from collections.abc import Mapping

import numpy as np

from ocf_data_sampler.common.time_utils import minutes


def choose_delay(delay_minute_probabilities: Mapping[int, float]) -> np.timedelta64 | None:
    """Draw a non-negative availability delay from validated minute/probability pairs.

    Unassigned probability returns None, meaning no additional availability cutoff.
    Iterating the mapping avoids allocating choice and probability arrays for every sample.

    Args:
        delay_minute_probabilities: Non-empty mapping of non-negative integer delays in
            minutes to positive probabilities summing to at most 1. Inputs must already
            be validated.

    Returns:
        The chosen delay, or None for the remaining probability of no added delay.

    """
    draw = np.random.random()
    cumulative_probability = 0.0
    for delay_minutes, probability in delay_minute_probabilities.items():
        cumulative_probability += probability
        if draw < cumulative_probability:
            return minutes(delay_minutes)
    return None
