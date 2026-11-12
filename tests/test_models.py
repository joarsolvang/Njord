from datetime import date

import pandas as pd

from njord.forecasting.data import MarketData
from njord.forecasting.day_ahead import delivery_periods, issue_time
from njord.forecasting.models import XGBoostForecaster, forecast_day
from tests.synthetic import poisoned_after

TEST_DAY = date(2026, 5, 10)


def _assert_ignores_future(model: XGBoostForecaster, clean: MarketData) -> None:
    expected = forecast_day(model, clean, TEST_DAY)
    actual = forecast_day(model, poisoned_after(issue_time(TEST_DAY)), TEST_DAY)
    assert expected.index.equals(delivery_periods(TEST_DAY))
    assert list(expected.columns) == ["p10", "p50", "p90"]
    pd.testing.assert_frame_equal(expected, actual)


def test_xgboost_forecast_ignores_future_data(clean: MarketData) -> None:
    model = XGBoostForecaster(n_estimators=20)
    model.fit(clean, pd.date_range("2026-04-10", "2026-05-05").date)
    _assert_ignores_future(model, clean)
    assert (forecast_day(model, clean, TEST_DAY).p50.between(0, 200)).all()
