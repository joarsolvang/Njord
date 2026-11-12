"""Convert between UTC timestamps and GB settlement (date, period).

Period 1 starts at 00:00 UK local time. Periods are counted in elapsed half-hours, so a
spring clock-change day has 46 periods and an autumn one has 50.
"""

from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo

UK = ZoneInfo("Europe/London")
HALF_HOUR = timedelta(minutes=30)


def _midnight_utc(settlement_date: date) -> datetime:
    return datetime.combine(settlement_date, time(0), tzinfo=UK).astimezone(UTC)


def periods_in_day(settlement_date: date) -> int:
    """46, 48 or 50."""
    return (_midnight_utc(settlement_date + timedelta(days=1)) - _midnight_utc(settlement_date)) // HALF_HOUR


def to_settlement(ts: datetime) -> tuple[date, int]:
    """Settlement (date, period) containing an aware timestamp."""
    if ts.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    settlement_date = ts.astimezone(UK).date()
    return settlement_date, (ts - _midnight_utc(settlement_date)) // HALF_HOUR + 1


def to_utc(settlement_date: date, settlement_period: int) -> datetime:
    """UTC start of a settlement period."""
    if not 1 <= settlement_period <= periods_in_day(settlement_date):
        raise ValueError(f"{settlement_date} has no settlement period {settlement_period}")
    return _midnight_utc(settlement_date) + (settlement_period - 1) * HALF_HOUR
