from datetime import date, timedelta

import psycopg

from njord.forecast_job import run_forecasts, training_days
from njord.forecasting.data import MarketData
from njord.forecasting.day_ahead import issue_time
from njord.forecasting.models import XGBoostForecaster


def test_training_days_end_before_the_first_issue_time() -> None:
    days = training_days(date(2026, 6, 10))
    assert days[-1] == date(2026, 6, 8)  # its power day ends 23:00 on 8 June, before 09:00 on 9 June
    assert len(days) == len(set(days)) == 365


def test_run_forecasts_appends_one_row_per_half_hour(conn: psycopg.Connection, clean: MarketData) -> None:
    days = [date(2026, 5, 10), date(2026, 5, 11)]
    model = XGBoostForecaster(n_estimators=10)
    assert run_forecasts(conn, clean, days, [model]) == 96
    assert run_forecasts(conn, clean, days, [model]) == 0  # re-running never duplicates or overwrites
    rows = conn.execute(
        "SELECT delivery_day, min(issued_at), count(*) FROM price_forecast GROUP BY delivery_day ORDER BY 1"
    ).fetchall()
    assert [(d, i, n) for d, i, n in rows] == [(d, issue_time(d), 48) for d in days]
    assert days[0] - timedelta(days=2) in training_days(days[0])
