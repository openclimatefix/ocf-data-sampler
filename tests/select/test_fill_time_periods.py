import pandas as pd

from ocf_data_sampler.select.fill_time_periods import fill_time_periods

def test_fill_time_periods():
    time_periods = pd.DataFrame(
        {
            "start_dt": [
                "2021-01-01 04:10:00", "2021-01-01 09:00:00",
                "2021-01-01 09:15:00", "2021-01-01 12:00:00"
            ],
            "end_dt": [
                "2021-01-01 06:00:00", "2021-01-01 09:00:00", 
                "2021-01-01 09:20:00", "2021-01-01 14:45:00"
            ],
        }
    )
    freq = pd.Timedelta("30min")
    filled_time_periods = fill_time_periods(time_periods, freq)

    expected_times = [
        "04:30", "05:00", "05:30", "06:00", "09:00", "12:00", 
        "12:30", "13:00", "13:30", "14:00", "14:30"
    ]

    expected_times = pd.DatetimeIndex([f"2021-01-01 {t}" for t in expected_times])

    pd.testing.assert_index_equal(filled_time_periods, expected_times)