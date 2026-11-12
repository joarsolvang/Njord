"""Point-in-time market data for forecasting: prices and NESO forecasts, as they stood at a given moment."""

from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import date

import numpy as np
import pandas as pd

from njord.clients.elexon_client import ElexonClient

# shortcut: MID for a half-hour is treated as known 30 min after the period ends; measure Elexon's real lag later.
PRICE_AVAILABLE_AFTER_START = pd.Timedelta("1h")
COVARIATES = ["wind", "demand", "margin", "imbalance", "residual"]


def _to_halfhours(s: pd.Series, index: pd.DatetimeIndex) -> np.ndarray:
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s.reindex(s.index.union(index)).interpolate(method="time").ffill().reindex(index).to_numpy()


def _nowcast(df: pd.DataFrame) -> pd.Series:
    """For each half-hour, the latest forecast for it that was published at or before that time."""
    n = df[df.publishTime <= df.startTime].sort_values("publishTime").drop_duplicates("startTime", keep="last")
    n = n.set_index("startTime").value
    idx = pd.date_range(n.index.min(), n.index.max(), freq="30min")
    return pd.Series(_to_halfhours(n, idx), idx)


@dataclass(frozen=True)
class MarketData:
    """Market index prices and NESO forecasts, optionally restricted to what was public at `as_of`.

    Models must only read through the methods, which apply `as_of`. Raw fields are private for that reason.
    """

    _price: pd.Series  # MID £/MWh, indexed by half-hour start (UTC)
    _forecasts: dict[str, pd.DataFrame]  # name -> publishTime, startTime, value; sorted by startTime
    _nowcasts: pd.DataFrame  # half-hourly, one column per covariate
    as_of: pd.Timestamp | None = None

    @classmethod
    def from_raw(cls, price: pd.Series, forecasts: dict[str, pd.DataFrame]) -> "MarketData":
        """`forecasts` must hold "wind" and "demand" (residual = demand - wind) plus any other covariates."""
        forecasts = {k: v.sort_values(["startTime", "publishTime"], ignore_index=True) for k, v in forecasts.items()}
        nowcasts = pd.DataFrame({k: _nowcast(v) for k, v in forecasts.items()})
        nowcasts["residual"] = nowcasts.demand - nowcasts.wind
        return cls(price.sort_index(), forecasts, nowcasts)

    @classmethod
    def load(cls, client: ElexonClient, start_inclusive: date, end_inclusive: date) -> "MarketData":
        mid = client.get_market_index(start_inclusive, end_inclusive)
        price = (
            mid[mid.dataProvider == "APXMIDP"]  # the N2EX index is all zeros
            .drop_duplicates("startTime")
            .set_index("startTime")
            .price.sort_index()
            .asfreq("30min")
            .interpolate(limit=4)  # shortcut: fills short gaps only; long gaps stay NaN
        )
        raw: dict[str, tuple[Callable[[date, date], pd.DataFrame], str]] = {
            "wind": (client.get_wind_forecast, "generation"),
            "demand": (client.get_demand_forecast, "demand"),
            "margin": (client.get_margin_forecast, "margin"),
            "imbalance": (client.get_imbalance_forecast, "imbalance"),
        }
        forecasts = {
            name: get(start_inclusive, end_inclusive).rename(columns={col: "value"})[
                ["publishTime", "startTime", "value"]
            ]
            for name, (get, col) in raw.items()
        }
        return cls.from_raw(price, forecasts)

    def asof(self, t: pd.Timestamp) -> "MarketData":
        """The same data, restricted to what had been published by `t`."""
        return replace(self, as_of=t)

    def prices(self) -> pd.Series:
        """Market index prices known at `as_of` (all of them if `as_of` is None)."""
        if self.as_of is None:
            return self._price
        return self._price[self._price.index + PRICE_AVAILABLE_AFTER_START <= self.as_of]

    def covariates(self, index: pd.DatetimeIndex) -> pd.DataFrame:
        """Forecasts for `index` from the latest runs published by `as_of`. Columns: COVARIATES."""
        out = {}
        for name, df in self._forecasts.items():
            lo, hi = df.startTime.searchsorted([index[0] - pd.Timedelta("1h"), index[-1] + pd.Timedelta("1h")])
            runs = df.iloc[lo : hi + 1]
            if self.as_of is not None:
                runs = runs[runs.publishTime <= self.as_of]
            latest = runs.sort_values("publishTime").drop_duplicates("startTime", keep="last")
            out[name] = _to_halfhours(latest.set_index("startTime").value, index) if len(latest) else np.nan
        cov = pd.DataFrame(out, index=index)
        return cov.assign(residual=cov.demand - cov.wind)[COVARIATES]

    def covariate_history(self, index: pd.DatetimeIndex) -> pd.DataFrame:
        """Short-lead forecasts for past times in `index`, as known at `as_of`. Later times are NaN."""
        known = self._nowcasts
        if self.as_of is not None:
            # A half-hour's nowcast interpolates the next hourly point, published up to 30 min later.
            known = known[known.index + pd.Timedelta("30min") <= self.as_of]
        return known.reindex(index)[COVARIATES]
