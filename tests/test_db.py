from datetime import date

import pandas as pd
import psycopg

from njord import db
from njord.forecasting.day_ahead import delivery_periods, issue_time

DAY = date(2026, 6, 10)


def _forecast(offset: float = 0) -> pd.DataFrame:
    periods = delivery_periods(DAY)
    return pd.DataFrame({"p10": 50.0 + offset, "p50": 80.0 + offset, "p90": 110.0 + offset}, index=periods)


def test_schema_is_idempotent_and_creates_hypertables(conn: psycopg.Connection) -> None:
    db.init_schema(conn)
    names = {r[0] for r in conn.execute("SELECT hypertable_name FROM timescaledb_information.hypertables")}
    assert {"market_index_price", "price_forecast"} <= names


def test_forecasts_are_append_only(conn: psycopg.Connection) -> None:
    assert db.write_forecast(conn, "m", DAY, issue_time(DAY), _forecast()) == 48
    assert db.write_forecast(conn, "m", DAY, issue_time(DAY), _forecast(offset=999)) == 0  # same issue: kept as is
    later = issue_time(DAY) + pd.Timedelta("1h")
    assert db.write_forecast(conn, "m", DAY, later, _forecast(offset=1)) == 48  # a new issue is a new forecast
    p50s = conn.execute("SELECT issued_at, max(p50) FROM price_forecast GROUP BY issued_at ORDER BY 1").fetchall()
    assert [p for _, p in p50s] == [80.0, 81.0]


def test_market_index_rerun_does_not_duplicate(conn: psycopg.Connection) -> None:
    mid = pd.DataFrame(
        {
            "dataProvider": "APXMIDP",
            "startTime": delivery_periods(DAY)[:3],
            "price": [70.0, 71.0, 72.0],
            "volume": [100.0, 110.0, 120.0],
        }
    )
    assert db.write_market_index(conn, mid) == 3
    assert db.write_market_index(conn, mid) == 0
    assert conn.execute("SELECT count(*) FROM market_index_price").fetchone() == (3,)
