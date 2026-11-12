"""Day-ahead price forecasters. Each returns P10/P50/P90 for every half-hour of a power day."""

from collections.abc import Sequence
from datetime import date
from typing import Protocol

import pandas as pd
import xgboost as xgb

from njord.forecasting.data import MarketData
from njord.forecasting.day_ahead import delivery_periods, issue_time

QUANTILES = (0.1, 0.5, 0.9)
COLUMNS = ["p10", "p50", "p90"]


class Forecaster(Protocol):
    name: str

    def fit(self, data: MarketData, delivery_days: Sequence[date]) -> None:
        """Train on past power days. `data` holds actuals; features must come from `data.asof(issue_time(day))`."""

    def predict(self, data: MarketData, delivery_day: date) -> pd.DataFrame:
        """Quantiles indexed by `delivery_periods(delivery_day)`. `data` is already restricted to the issue time."""


def forecast_day(model: Forecaster, data: MarketData, delivery_day: date) -> pd.DataFrame:
    """Forecast one power day using only data published before its issue time. Use this, not `model.predict`."""
    out = model.predict(data.asof(issue_time(delivery_day)), delivery_day)
    if not out.index.equals(delivery_periods(delivery_day)):
        raise ValueError(f"{model.name} forecast does not cover exactly the {delivery_day} power day")
    return out[COLUMNS]


class XGBoostForecaster:
    """Gradient-boosted trees with quantile loss, on calendar features, price lags and NESO forecasts.

    Trees can't extrapolate, so the target is the price relative to the last known day's mean. That keeps the model
    usable when the price level moves outside the training range.
    """

    name = "xgboost"

    def __init__(self, n_estimators: int = 400, learning_rate: float = 0.05, max_depth: int = 6) -> None:
        self._model = xgb.XGBRegressor(
            objective="reg:quantileerror",
            quantile_alpha=list(QUANTILES),
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
        )

    @staticmethod
    def _features(data: MarketData, periods: pd.DatetimeIndex) -> tuple[pd.DataFrame, float]:
        prices = data.prices()
        level = float(prices.iloc[-48:].mean())
        local = periods.tz_convert("Europe/London")
        lags = {f"lag_{d}d": prices.reindex(periods - pd.Timedelta(days=d)).to_numpy() - level for d in (2, 7)}
        feats = pd.DataFrame({"hod": local.hour + local.minute / 60, "dow": local.day_of_week, **lags}, index=periods)
        return feats.join(data.covariates(periods)), level

    def fit(self, data: MarketData, delivery_days: Sequence[date]) -> None:
        xs, ys = [], []
        actual = data.prices()
        for day in delivery_days:
            periods = delivery_periods(day)
            x, level = self._features(data.asof(issue_time(day)), periods)
            xs.append(x)
            ys.append(actual.reindex(periods) - level)
        x, y = pd.concat(xs), pd.concat(ys)
        known = y.notna().to_numpy()
        self._model.fit(x[known], y[known])

    def predict(self, data: MarketData, delivery_day: date) -> pd.DataFrame:
        periods = delivery_periods(delivery_day)
        x, level = self._features(data, periods)
        return pd.DataFrame(self._model.predict(x) + level, index=periods, columns=COLUMNS)
