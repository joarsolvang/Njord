import pytest

from njord.forecasting.data import MarketData
from tests.synthetic import raw_market_data


@pytest.fixture(scope="module")
def clean() -> MarketData:
    return MarketData.from_raw(*raw_market_data())
