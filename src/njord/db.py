"""TimescaleDB storage. Append-only: rows are never updated, so past forecasts stay exactly as they were issued."""

import os
from datetime import date

import pandas as pd
import psycopg

from njord.forecasting.data import PRICE_AVAILABLE_AFTER_START

DEFAULT_URL = "postgresql://njord:njord@127.0.0.1:5432/njord"  # matches docker-compose.yml

SCHEMA = """
CREATE EXTENSION IF NOT EXISTS timescaledb;

CREATE TABLE IF NOT EXISTS market_index_price (
    provider text NOT NULL,
    valid_for timestamptz NOT NULL,     -- half-hour start
    issued_at timestamptz NOT NULL,     -- estimated publication time
    price double precision NOT NULL,    -- GBP/MWh
    volume double precision,            -- MWh
    PRIMARY KEY (provider, valid_for)
);
SELECT create_hypertable('market_index_price', by_range('valid_for'), if_not_exists => TRUE);

CREATE TABLE IF NOT EXISTS price_forecast (
    model text NOT NULL,
    delivery_day date NOT NULL,
    issued_at timestamptz NOT NULL,
    valid_for timestamptz NOT NULL,     -- half-hour start
    p10 double precision NOT NULL,
    p50 double precision NOT NULL,
    p90 double precision NOT NULL,
    PRIMARY KEY (model, issued_at, valid_for)
);
SELECT create_hypertable('price_forecast', by_range('valid_for'), if_not_exists => TRUE);
"""


def connect(url: str | None = None) -> psycopg.Connection:
    """Connect to `url`, else $NJORD_DB_URL, else the local docker-compose database."""
    return psycopg.connect(url or os.environ.get("NJORD_DB_URL") or DEFAULT_URL)


def init_schema(conn: psycopg.Connection) -> None:
    """Create tables and hypertables if they don't exist. Safe to re-run."""
    conn.execute(SCHEMA)
    conn.commit()


def write_market_index(conn: psycopg.Connection, mid: pd.DataFrame) -> int:
    """Insert market index rows from `ElexonClient.get_market_index`. Rows already stored are left untouched."""
    # shortcut: issued_at is estimated (see PRICE_AVAILABLE_AFTER_START), and later revisions are ignored.
    rows = [
        (r.dataProvider, r.startTime, r.startTime + PRICE_AVAILABLE_AFTER_START, r.price, r.volume)
        for r in mid.itertuples()
    ]
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO market_index_price (provider, valid_for, issued_at, price, volume)"
            " VALUES (%s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
            rows,
        )
        inserted = cur.rowcount
    conn.commit()
    return inserted


def write_forecast(
    conn: psycopg.Connection, model: str, delivery_day: date, issued_at: pd.Timestamp, forecast: pd.DataFrame
) -> int:
    """Insert one power-day forecast (index valid_for; columns p10, p50, p90). Existing rows are never replaced."""
    rows = [(model, delivery_day, issued_at, t, r.p10, r.p50, r.p90) for t, r in forecast.iterrows()]
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO price_forecast (model, delivery_day, issued_at, valid_for, p10, p50, p90)"
            " VALUES (%s, %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
            rows,
        )
        inserted = cur.rowcount
    conn.commit()
    return inserted
