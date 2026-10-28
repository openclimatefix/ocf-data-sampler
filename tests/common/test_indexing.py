import numpy as np
import pytest

from ocf_data_sampler.common.indexing import get_string_index


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
