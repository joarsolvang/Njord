from datetime import date

import numpy as np
import pandas as pd

from njord.forecasting.data import COVARIATES, MarketData
from njord.forecasting.day_ahead import delivery_periods, issue_time
from tests.synthetic import POISON, poisoned_after

TEST_DAY = date(2026, 5, 10)


def test_asof_hides_everything_published_later() -> None:
    t = issue_time(TEST_DAY)
    seen = poisoned_after(t).asof(t)
    periods = delivery_periods(TEST_DAY)
    assert seen.prices().max() < POISON
    assert seen.covariates(periods).abs().max().max() < POISON
    assert seen.covariate_history(pd.date_range(t - pd.Timedelta("7D"), t, freq="30min")).abs().max().max() < POISON


def test_covariates_use_latest_run_before_as_of(clean: MarketData) -> None:
    periods = delivery_periods(TEST_DAY)
    cov = clean.asof(issue_time(TEST_DAY)).covariates(periods)
    assert list(cov.columns) == COVARIATES
    assert cov.notna().all().all()
    assert np.allclose(cov.residual, cov.demand - cov.wind)
