"""Synthetic market data shared by the forecasting tests."""

import numpy as np
import pandas as pd

from njord.forecasting.data import PRICE_AVAILABLE_AFTER_START, MarketData

POISON = 1e6
START, END = pd.Timestamp("2026-04-01", tz="UTC"), pd.Timestamp("2026-05-15", tz="UTC")


def raw_market_data() -> tuple[pd.Series, dict[str, pd.DataFrame]]:
    """Daily-shaped prices, and forecast runs every 3 h covering the next 48 h hourly."""
    rng = np.random.default_rng(0)
    idx = pd.date_range(START, END, freq="30min", inclusive="left")
    hours = idx.hour + idx.minute / 60
    price = pd.Series(100 + 30 * np.sin(2 * np.pi * hours / 24) + rng.normal(0, 5, len(idx)), idx)
    runs = []
    for published in pd.date_range(START, END, freq="3h"):
        targets = pd.date_range(published.ceil("1h"), periods=48, freq="1h")
        runs.append(pd.DataFrame({"publishTime": published, "startTime": targets}))
    base = pd.concat(runs, ignore_index=True)
    forecasts = {
        name: base.assign(
            value=scale * (1 + np.sin(2 * np.pi * base.startTime.dt.hour / 24)) + rng.normal(0, 1, len(base))
        )
        for name, scale in [("wind", 8000), ("demand", 25000), ("margin", 5000), ("imbalance", 1000)]
    }
    return price, forecasts


def poisoned_after(t: pd.Timestamp) -> MarketData:
    """Same data, but every value not yet public at `t` is replaced by POISON."""
    price, forecasts = raw_market_data()
    price = price.where(price.index + PRICE_AVAILABLE_AFTER_START <= t, POISON)
    forecasts = {k: v.assign(value=v.value.where(v.publishTime <= t, POISON)) for k, v in forecasts.items()}
    return MarketData.from_raw(price, forecasts)
