"""GB day-ahead auction timeline.

The hourly auctions close at 09:20 (EPEX) and 09:50 (N2EX) UK time on the day before delivery. They cover the next
"power day", which runs from 23:00 UK time the day before to 23:00 UK time on the delivery day (46, 48 or 50
half-hours, because of clock changes).
"""

from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

UK = ZoneInfo("Europe/London")
ISSUE_TIME_UK = time(9, 0)  # forecasts are made before the 09:20 EPEX gate closure
POWER_DAY_START_UK = time(23, 0)


def issue_time(delivery_day: date) -> pd.Timestamp:
    """When the forecast for `delivery_day` is made (UTC): 09:00 UK time the day before."""
    return pd.Timestamp(datetime.combine(delivery_day - timedelta(days=1), ISSUE_TIME_UK, tzinfo=UK)).tz_convert("UTC")


def delivery_periods(delivery_day: date) -> pd.DatetimeIndex:
    """Half-hour start times (UTC) of the power day: from 23:00 UK the day before to 23:00 UK on the day, excluded."""
    start = datetime.combine(delivery_day - timedelta(days=1), POWER_DAY_START_UK, tzinfo=UK)
    end_exclusive = datetime.combine(delivery_day, POWER_DAY_START_UK, tzinfo=UK)
    return pd.date_range(
        pd.Timestamp(start).tz_convert("UTC"),
        pd.Timestamp(end_exclusive).tz_convert("UTC"),
        freq="30min",
        inclusive="left",
    )
