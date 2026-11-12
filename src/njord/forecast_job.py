"""Day-ahead forecast job: forecast power days with every model and append the results to TimescaleDB.

python -m njord.forecast_job run                      # tomorrow's power day (run after 09:00 UK)
python -m njord.forecast_job backfill 2026-08-01 2026-11-10
"""

import argparse
import logging
from collections.abc import Callable, Sequence
from datetime import date, datetime, timedelta

import pandas as pd
import psycopg

from njord import db
from njord.clients.elexon_client import ElexonClient
from njord.forecasting.data import MarketData
from njord.forecasting.day_ahead import UK, issue_time
from njord.forecasting.models import Chronos2Forecaster, Forecaster, XGBoostForecaster, forecast_day

log = logging.getLogger(__name__)

MODELS: dict[str, Callable[[], Forecaster]] = {"xgboost": XGBoostForecaster, "chronos2": Chronos2Forecaster}
TRAIN_DAYS = 365
CONTEXT_DAYS = 45  # Chronos-2 reads ~6 weeks of half-hours


def training_days(first_delivery_day: date) -> list[date]:
    """Power days whose prices are all known before the first issue time: up to two days before delivery."""
    last = first_delivery_day - timedelta(days=2)
    return list(pd.date_range(last - timedelta(days=TRAIN_DAYS - 1), last).date)


def run_forecasts(
    conn: psycopg.Connection, data: MarketData, delivery_days: Sequence[date], models: Sequence[Forecaster]
) -> int:
    """Fit each model once on days before the first delivery day, forecast every day, append to the database."""
    # shortcut: one fit per run; a long backfill should refit as it advances (e.g. monthly).
    train = training_days(min(delivery_days))
    written = 0
    for model in models:
        log.info("%s: fitting on %s to %s", model.name, train[0], train[-1])
        model.fit(data, train)
        for day in delivery_days:
            written += db.write_forecast(conn, model.name, day, issue_time(day), forecast_day(model, data, day))
        log.info("%s: forecast %d power day(s)", model.name, len(delivery_days))
    return written


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("run", help="forecast tomorrow's power day")
    backfill = sub.add_parser("backfill", help="forecast a range of past power days")
    backfill.add_argument("first_day_inclusive", type=date.fromisoformat)
    backfill.add_argument("last_day_inclusive", type=date.fromisoformat)
    parser.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    today = datetime.now(UK).date()
    if args.command == "run":
        days = [today + timedelta(days=1)]
    else:
        days = list(pd.date_range(args.first_day_inclusive, args.last_day_inclusive).date)

    client = ElexonClient()
    start = min(training_days(days[0])[0], days[0] - timedelta(days=CONTEXT_DAYS)) - timedelta(days=8)
    end = min(days[-1], today)  # data after each issue time is hidden from the models by forecast_day
    data = MarketData.load(client, start, end)
    with db.connect() as conn:
        db.init_schema(conn)
        mid = client.get_market_index(days[0] - timedelta(days=1), end)
        db.write_market_index(conn, mid[mid.dataProvider == "APXMIDP"])  # the N2EX index is all zeros
        written = run_forecasts(conn, data, days, [MODELS[m]() for m in args.models])
    log.info("wrote %d forecast rows", written)


if __name__ == "__main__":
    main()
