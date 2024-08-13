import pandas as pd

from ocf_data_sampler.select.find_contiguous_time_periods import (
    find_contiguous_t0_periods, find_contiguous_t0_periods_nwp, 
    intersection_of_multiple_dataframes_of_periods,
)



def test_find_contiguous_t0_periods():

    # Create 5-minutely data timestamps
    freq = pd.Timedelta(5, "min")
    history_duration = pd.Timedelta(60, "min")
    forecast_duration = pd.Timedelta(15, "min")

    datetimes = (
        pd.date_range("2023-01-01 12:00", "2023-01-01 17:00", freq=freq)
        .delete([5, 6, 30])
    )

    periods = find_contiguous_t0_periods(
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


def test_find_contiguous_t0_periods_nwp():

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
                    ]
                ),
                "end_dt": pd.to_datetime(
                    [
                        "2023-01-01 21:00",
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
                        "2023-01-02 05:00",
                        "2023-01-02 14:00",
                    ]
                ),
                "end_dt": pd.to_datetime(
                    [
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
                        "2023-01-02 05:00",
                        "2023-01-02 14:00",
                    ]
                ),
                "end_dt": pd.to_datetime(
                    [
                        "2023-01-01 06:00",
                        "2023-01-01 15:00",
                        "2023-01-02 06:00",
                        "2023-01-03 00:00",
                    ]
                ),
            },
        ),
        pd.DataFrame(
            {
                "start_dt": pd.to_datetime(
                    [
                        "2023-01-01 06:00",
                        "2023-01-01 12:00",
                        "2023-01-02 06:00",
                        "2023-01-02 15:00",
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
    ]

    # Create 3-hourly init times with a few time stamps missing
    freq = pd.Timedelta(3, "h")

    datetimes = (
        pd.date_range("2023-01-01 03:00", "2023-01-02 21:00", freq=freq)
        .delete([1, 4, 5, 6, 7, 9, 10])
    )

    # Choose some history durations and max stalenesses
    history_durations_hr = [0, 2, 2, 2, 2]
    max_stalenesses_hr = [9, 9, 6, 3, 6]
    max_dropouts_hr = [0, 0, 0, 0, 3]

    for i in range(len(expected_results)):
        history_duration = pd.Timedelta(history_durations_hr[i], "h")
        max_staleness = pd.Timedelta(max_stalenesses_hr[i], "h")
        max_dropout = pd.Timedelta(max_dropouts_hr[i], "h")

        time_periods = find_contiguous_t0_periods_nwp(
            datetimes=datetimes,
            history_duration=history_duration,
            max_staleness=max_staleness,
            max_dropout=max_dropout,
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
