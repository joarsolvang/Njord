from collections.abc import Iterator

import psycopg
import pytest

from njord import db
from njord.forecasting.data import MarketData
from tests.synthetic import raw_market_data


@pytest.fixture(scope="module")
def clean() -> MarketData:
    return MarketData.from_raw(*raw_market_data())


@pytest.fixture
def conn() -> Iterator[psycopg.Connection]:
    """A connection whose tables live in a throwaway schema. Skips if the database isn't running."""
    try:
        c = db.connect()
    except psycopg.OperationalError:
        pytest.skip("TimescaleDB not running (docker compose up -d)")
    c.execute(
        "DROP SCHEMA IF EXISTS njord_test CASCADE; CREATE SCHEMA njord_test; SET search_path TO njord_test, public"
    )
    db.init_schema(c)
    yield c
    c.rollback()
    c.execute("DROP SCHEMA njord_test CASCADE")
    c.commit()
    c.close()
