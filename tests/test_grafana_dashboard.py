import json
from datetime import date
from pathlib import Path

import pandas as pd
import psycopg

from njord import db
from njord.forecasting.day_ahead import delivery_periods, issue_time

DASHBOARD = Path(__file__).parents[1] / "grafana" / "dashboards" / "day_ahead.json"
DAY = date(2026, 6, 10)


def _sql(raw: str, frm: pd.Timestamp, to: pd.Timestamp) -> str:
    """Expand the Grafana macros and variables used by the dashboard."""
    return (
        raw.replace("$__timeFrom()", f"'{frm.isoformat()}'::timestamptz")
        .replace("$__timeTo()", f"'{to.isoformat()}'::timestamptz")
        .replace("$model", "chronos2")
    )


def test_every_dashboard_query_runs_and_covers_one_power_day(conn: psycopg.Connection) -> None:
    periods = delivery_periods(DAY)
    # Rows for the power day plus the first half-hour of the next one, which must not leak into this day.
    valid_for = periods.append(pd.DatetimeIndex([periods[-1] + pd.Timedelta("30min")]))
    fc = pd.DataFrame({"p10": 50.0, "p50": 80.0, "p90": 110.0}, index=valid_for)
    for model in ("chronos2", "xgboost"):
        db.write_forecast(conn, model, DAY, issue_time(DAY), fc)
    db.write_market_index(
        conn, pd.DataFrame({"dataProvider": "APXMIDP", "startTime": valid_for, "price": 90.0, "volume": 1.0})
    )

    panels = json.loads(DASHBOARD.read_text(encoding="utf-8"))["panels"]
    targets = [(p["title"], t) for p in panels for t in p.get("targets", [])]
    assert len(targets) == 4
    for title, t in targets:
        rows = conn.execute(_sql(t["rawSql"], periods[0], periods[-1] + pd.Timedelta("30min"))).fetchall()
        if title.startswith("Day-ahead price"):
            assert len(rows) == len(periods), (title, t["refId"])
        if title == "Power day summary":
            assert [(r[0], r[1]) for r in rows] == [("chronos2", 10.0), ("xgboost", 10.0)]
