from datetime import timedelta
import pandas as pd

from ocf_data_sampler.select.find_contiguous_t0_time_periods import (
    find_contiguous_t0_time_periods, find_contiguous_t0_periods_nwp, 
    intersection_of_multiple_dataframes_of_periods,
)



def test_find_contiguous_t0_time_periods():

    # Create 5-minutely data timestamps
    freq = timedelta(minutes=5)
    history_duration = timedelta(minutes=60)
    forecast_duration = timedelta(minutes=15)

    datetimes = (
        pd.date_range("2023-01-01 12:00", "2023-01-01 17:00", freq=freq)
        .delete([5, 6, 30])
    )

    periods = find_contiguous_t0_time_periods(
        datetimes=datetimes,
        history_duration=history_duration,
        forecast_duration=forecast_duration,
        sample_period_duration=freq,
    )

    expected_results = pd.DataFrame(
        {
            "start_dt": pd.to_datetime(
                [
                    "2023-01-01 13:35",
                    "2023-01-01 15:35",
                ]
            ),
            "end_dt": pd.to_datetime(
                [
                    "2023-01-01 14:10",
                    "2023-01-01 16:45",
                ]
            ),
        },
    )

    assert periods.equals(expected_results)


def test_find_contiguous_t0_time_periods_nwp():

    # These are the expected results of the test
    expected_results = [
        pd.DataFrame(
            {
                "start_dt": pd.to_datetime(["2023-01-01 03:00", "2023-01-02 03:00"]),
                "end_dt": pd.to_datetime(["2023-01-01 21:00", "2023-01-03 06:00"]),
            },
        ),
        pd.DataFrame(
            {
                "start_dt": pd.to_datetime(
                    [
                        "2023-01-01 05:00",
                        "2023-01-02 05:00",
                        "2023-01-02 14:00",
                    ]
                ),
                "end_dt": pd.to_datetime(
                    [
                        "2023-01-01 21:00",
                        "2023-01-02 12:00",
                        "2023-01-03 06:00",
                    ]
                ),
            },
        ),
        pd.DataFrame(
            {
                "start_dt": pd.to_datetime(
                    [
                        "2023-01-01 05:00",
                        "2023-01-01 11:00",
                        "2023-01-02 05:00",
                        "2023-01-02 14:00",
                    ]
                ),
                "end_dt": pd.to_datetime(
                    [
                        "2023-01-01 09:00",
                        "2023-01-01 18:00",
                        "2023-01-02 09:00",
                        "2023-01-03 03:00",
                    ]
                ),
            },
        ),
        pd.DataFrame(
            {
                "start_dt": pd.to_datetime(
                    [
                        "2023-01-01 05:00",
                        "2023-01-01 11:00",
                        "2023-01-01 14:00",
                        "2023-01-02 05:00",
                        "2023-01-02 14:00",
                        "2023-01-02 17:00",
                        "2023-01-02 20:00",
                        "2023-01-02 23:00",
                    ]
                ),
                "end_dt": pd.to_datetime(
                    [
                        "2023-01-01 06:00",
                        "2023-01-01 12:00",
                        "2023-01-01 15:00",
                        "2023-01-02 06:00",
                        "2023-01-02 15:00",
                        "2023-01-02 18:00",
                        "2023-01-02 21:00",
                        "2023-01-03 00:00",
                    ]
                ),
            },
        ),
    ]

    # Create 3-hourly init times with a few time stamps missing
    freq = timedelta(minutes=180)

    datetimes = (
        pd.date_range("2023-01-01 03:00", "2023-01-02 21:00", freq=freq)
        .delete([1, 4, 5, 6, 7, 9, 10])
    )
    steps = [timedelta(hours=i) for i in range(24)]

    # Choose some history durations and max stalenesses
    history_durations_hr = [0, 2, 2, 2]
    max_stalenesses_hr = [9, 9, 6, 3]

    for i in range(len(expected_results)):
        history_duration = timedelta(hours=history_durations_hr[i])
        max_staleness = timedelta(hours=max_stalenesses_hr[i])

        time_periods = find_contiguous_t0_periods_nwp(
            datetimes=datetimes,
            history_duration=history_duration,
            max_staleness=max_staleness,
            max_dropout = timedelta(0),
        )

        # Check if results are as expected
        assert time_periods.equals(expected_results[i])


def test_intersection_of_multiple_dataframes_of_periods():
    periods_1 = pd.DataFrame(
        {
            "start_dt": pd.to_datetime(["2023-01-01 05:00", "2023-01-01 14:10"]),
            "end_dt": pd.to_datetime(["2023-01-01 13:35", "2023-01-01 18:00"]),
        },
    )

    periods_2 = pd.DataFrame(
        {
            "start_dt": pd.to_datetime(["2023-01-01 12:00"]),
            "end_dt": pd.to_datetime(["2023-01-02 00:00"]),
        },
    )

    periods_3 = pd.DataFrame(
        {
            "start_dt": pd.to_datetime(["2023-01-01 00:00", "2023-01-01 13:00"]),
            "end_dt": pd.to_datetime(["2023-01-01 12:30", "2023-01-01 23:00"]),
        },
    )

    expected_result = pd.DataFrame(
        {
            "start_dt": pd.to_datetime(
                ["2023-01-01 12:00", "2023-01-01 13:00", "2023-01-01 14:10"]
            ),
            "end_dt": pd.to_datetime([
                "2023-01-01 12:30", "2023-01-01 13:35", "2023-01-01 18:00"]
            ),
        },
    )

    overlaping_periods = intersection_of_multiple_dataframes_of_periods(
        [periods_1, periods_2, periods_3]
    )   

    # Check if results are as expected
    assert overlaping_periods.equals(expected_result)