import numpy as np
import pytest

from ocf_data_sampler.common.indexing import (
    assert_values_unique_increasing,
    get_indices_in_sorted_unique,
    get_string_index,
)


@pytest.mark.parametrize(("query", "expected"), [("site-3", 0), ("site-1", 1), ("site-2", 2)])
def test_get_string_index(query, expected):
    values = np.array(["site-3", "site-1", "site-2"])
    assert get_string_index(values, query) == expected


@pytest.mark.parametrize(
    ("values", "n_matches"),
    [(["site-3", "site-1"], 0), ([], 0), (["site-2", "site-1", "site-2"], 2)],
)
def test_get_string_index_requires_one_match(values, n_matches):
    with pytest.raises(ValueError, match=f"Expected one match for 'site-2', found {n_matches}"):
        get_string_index(np.array(values, dtype=np.str_), "site-2")


@pytest.mark.parametrize(
    ("query", "expected"),
    [(20, 1), (np.array([10, 30]), np.array([0, 2]))],
)
def test_get_indices_in_sorted_unique(query, expected):
    indices = get_indices_in_sorted_unique(np.array([10, 20, 30]), query)
    np.testing.assert_array_equal(indices, expected)


# Before the start, between values, and after the end
@pytest.mark.parametrize("query", [5, 25, np.array([20, 40])])
def test_get_indices_in_sorted_unique_rejects_missing(query):
    with pytest.raises(ValueError, match="Not all values"):
        get_indices_in_sorted_unique(np.array([10, 20, 30]), query)


@pytest.mark.parametrize("values", [[1, 1, 2], [3, 2, 1]])
def test_assert_values_unique_increasing_rejects(values):
    with pytest.raises(ValueError, match="times must be strictly increasing"):
        assert_values_unique_increasing(np.array(values), "times")
