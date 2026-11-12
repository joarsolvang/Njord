"""Day-ahead price forecasters. Each returns P10/P50/P90 for every half-hour of a power day."""

from collections.abc import Sequence
from datetime import date
from typing import Any, Protocol

import pandas as pd
import xgboost as xgb

from njord.forecasting.data import COVARIATES, MarketData
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


class Chronos2Forecaster:
    """Amazon Chronos-2, zero-shot: reads recent prices and NESO forecasts as covariates, no training on GB data."""

    name = "chronos2"

    def __init__(
        self, covariates: Sequence[str] = COVARIATES, context: int = 2048, model_id: str = "amazon/chronos-2"
    ) -> None:
        self.covariates = list(covariates)
        self.context = context  # half-hours of history, ~6 weeks
        self.model_id = model_id
        self._pipeline: Any = None

    def fit(self, data: MarketData, delivery_days: Sequence[date]) -> None:
        """Nothing to train."""

    def predict(self, data: MarketData, delivery_day: date) -> pd.DataFrame:
        periods = delivery_periods(delivery_day)
        hist = data.prices().iloc[-self.context :].asfreq("30min").interpolate(limit_direction="both")
        # Forecast from the last known price through to the end of the power day, then keep the power day.
        future = pd.date_range(hist.index[-1] + pd.Timedelta("30min"), periods[-1], freq="30min")
        past = pd.DataFrame({"item_id": "mid", "timestamp": hist.index.tz_localize(None), "target": hist.to_numpy()})
        fut = pd.DataFrame({"item_id": "mid", "timestamp": future.tz_localize(None)})
        if self.covariates:
            cov_past = data.covariate_history(hist.index)[self.covariates].ffill().bfill()
            cov_future = data.covariates(future)[self.covariates].ffill().bfill()
            past = past.join(cov_past.reset_index(drop=True))
            fut = fut.join(cov_future.reset_index(drop=True))
        out = self._load().predict_df(
            past,
            future_df=fut if self.covariates else None,
            prediction_length=len(future),
            quantile_levels=list(QUANTILES),
            freq="30min",
        )
        out.index = pd.DatetimeIndex(out.timestamp).tz_localize("UTC")
        return pd.DataFrame(out[[str(q) for q in QUANTILES]].to_numpy(), index=out.index, columns=COLUMNS).reindex(
            periods
        )

    def _load(self) -> Any:  # noqa: ANN401 (chronos has no type stubs)
        if self._pipeline is None:
            # Imported here: torch and chronos take several seconds to import.
            import torch
            from chronos import Chronos2Pipeline

            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._pipeline = Chronos2Pipeline.from_pretrained(self.model_id, device_map=device)
        return self._pipeline
