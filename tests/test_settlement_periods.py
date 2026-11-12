from datetime import UTC, date, datetime, timedelta
from itertools import pairwise

import pytest

from njord.settlement_periods import UK, periods_in_day, to_settlement, to_utc

SPRING = date(2025, 3, 30)  # clocks go forward at 01:00 UTC
AUTUMN = date(2025, 10, 26)  # clocks go back at 01:00 UTC


@pytest.mark.parametrize(("day", "n"), [(SPRING, 46), (AUTUMN, 50), (date(2025, 6, 1), 48), (date(2025, 1, 1), 48)])
def test_periods_in_day(day: date, n: int) -> None:
    assert periods_in_day(day) == n


def test_known_values() -> None:
    # Summer: period 1 starts 23:00 UTC the previous day.
    assert to_utc(date(2025, 6, 1), 1) == datetime(2025, 5, 31, 23, 0, tzinfo=UTC)
    assert to_settlement(datetime(2025, 5, 31, 23, 15, tzinfo=UTC)) == (date(2025, 6, 1), 1)
    # Spring day: 01:00 UTC is 02:00 BST, the third period.
    assert to_settlement(datetime(2025, 3, 30, 1, 0, tzinfo=UTC)) == (SPRING, 3)
    # Autumn day: 01:00 UTC is the second 01:00 BST/GMT, period 5.
    assert to_settlement(datetime(2025, 10, 26, 1, 0, tzinfo=UTC)) == (AUTUMN, 5)
    assert to_utc(AUTUMN, 50) == datetime(2025, 10, 26, 23, 30, tzinfo=UTC)
    # Non-UTC input is fine as long as it is aware.
    assert to_settlement(datetime(2025, 6, 1, 0, 0, tzinfo=UK)) == (date(2025, 6, 1), 1)


def test_round_trip_across_clock_changes() -> None:
    ts = datetime(2025, 3, 28, tzinfo=UTC)
    seen: list[tuple[date, int]] = []
    while ts < datetime(2025, 10, 29, tzinfo=UTC):
        sd, sp = to_settlement(ts)
        assert to_utc(sd, sp) == ts
        seen.append((sd, sp))
        ts += timedelta(minutes=30)
    # Periods are contiguous: each day runs 1..N with no gaps.
    for (d0, p0), (d1, p1) in pairwise(seen):
        assert (d1, p1) == ((d0, p0 + 1) if d1 == d0 else (d0 + timedelta(days=1), 1))
        if d1 != d0:
            assert p0 == periods_in_day(d0)


def test_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="aware"):
        to_settlement(datetime(2025, 1, 1))
    with pytest.raises(ValueError, match="no settlement period"):
        to_utc(SPRING, 47)
    with pytest.raises(ValueError, match="no settlement period"):
        to_utc(SPRING, 0)
