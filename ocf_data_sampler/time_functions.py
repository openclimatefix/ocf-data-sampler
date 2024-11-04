import pandas as pd


def minutes(minutes: int | list[float]) -> pd.Timedelta | pd.TimedeltaIndex:
    """Timedelta minutes

    Args:
        minutes: the number of minutes, single value or list
    """
    minutes_delta = pd.to_timedelta(minutes, unit="m")
    return minutes_delta