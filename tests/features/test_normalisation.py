import numpy as np

from ocf_data_sampler.features.normalisation import clip_and_standardise


def test_clip_and_standardise():
    data = np.array([[-2, 6], [8, 14]], dtype=np.float32)
    original = data.copy()

    result = clip_and_standardise(
        data,
        clip_min=np.array([0, 8], dtype=np.float32),
        clip_max=np.array([4, 12], dtype=np.float32),
        mean=np.array([2, 10], dtype=np.float32),
        std=np.array([2, 4], dtype=np.float32),
    )

    # -2 is clipped up to 0, then standardised to (0 - 2) / 2 = -1
    # 6 is clipped up to 8, then standardised to (8 - 10) / 4 = -0.5
    # 8 is clipped down to 4, then standardised to (4 - 2) / 2 = 1
    # 14 is clipped down to 12, then standardised to (12 - 10) / 4 = 0.5
    np.testing.assert_allclose(result, [[-1, -0.5], [1, 0.5]])
    assert result.dtype == np.float32
    np.testing.assert_array_equal(data, original)
