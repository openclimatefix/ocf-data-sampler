import pandas as pd

from ocf_data_sampler.select.fill_time_periods import fill_time_periods


def test_fill_time_periods():
    time_periods = pd.DataFrame(
        {
            "start_dt": [
                "2021-01-01 04:10:00",
                "2021-01-01 09:00:00",
                "2021-01-01 09:15:00",
                "2021-01-01 12:00:00",
            ],
            "end_dt": [
                "2021-01-01 06:00:00",
                "2021-01-01 09:00:00",
                "2021-01-01 09:20:00",
                "2021-01-01 14:45:00",
            ],
        },
    )

    filled = fill_time_periods(time_periods, freq=pd.Timedelta("30min"))

    expected = pd.to_datetime(
        [
            "2021-01-01 04:30",
            "2021-01-01 05:00",
            "2021-01-01 05:30",
            "2021-01-01 06:00",
            "2021-01-01 09:00",
            "2021-01-01 12:00",
            "2021-01-01 12:30",
            "2021-01-01 13:00",
            "2021-01-01 13:30",
            "2021-01-01 14:00",
            "2021-01-01 14:30",
        ],
    )

    pd.testing.assert_index_equal(filled, expected)
