from datetime import date, timedelta

import pandas as pd
import pytest

from njord.forecasting.day_ahead import delivery_periods, issue_time


@pytest.mark.parametrize(
    ("day", "n_periods"),
    [(date(2026, 6, 10), 48), (date(2026, 3, 29), 46), (date(2026, 10, 25), 50)],  # normal, spring, autumn
)
def test_power_day_runs_23_to_23_uk_time(day: date, n_periods: int) -> None:
    periods = delivery_periods(day)
    local = periods.tz_convert("Europe/London")
    assert len(periods) == n_periods
    assert (local[0].date(), local[0].hour) == (day - timedelta(days=1), 23)
    assert (local[-1] + pd.Timedelta("30min")).hour == 23
    assert (periods[1:] - periods[:-1] == pd.Timedelta("30min")).all()


def test_issue_time_is_before_gate_closure_the_day_before() -> None:
    assert issue_time(date(2026, 6, 10)) == pd.Timestamp("2026-06-09 08:00", tz="UTC")  # 09:00 BST
    assert issue_time(date(2026, 1, 10)) == pd.Timestamp("2026-01-09 09:00", tz="UTC")  # 09:00 GMT
    assert issue_time(date(2026, 6, 10)) < delivery_periods(date(2026, 6, 10))[0]
