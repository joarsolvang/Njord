import logging
import urllib.error
from datetime import date
from pathlib import Path

import pytest

from njord import elexon
from njord.elexon import ElexonClient, _chunks, _is_retryable


def test_chunks_cover_range_without_overlap() -> None:
    assert list(_chunks(date(2025, 1, 1), date(2025, 1, 16), 7)) == [
        (date(2025, 1, 1), date(2025, 1, 7)),
        (date(2025, 1, 8), date(2025, 1, 14)),
        (date(2025, 1, 15), date(2025, 1, 16)),
    ]
    assert list(_chunks(date(2025, 1, 1), date(2025, 1, 1), 1)) == [(date(2025, 1, 1), date(2025, 1, 1))]


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Replace the network with a stub that records each URL requested."""
    urls: list[str] = []

    def fake_download(url: str) -> dict:
        urls.append(url)
        return {"data": [{"publishTime": "2025-01-01T00:00Z", "generation": 1}]}

    monkeypatch.setattr(elexon, "_download", fake_download)
    return urls


def test_cache_files_are_named_by_dataset_and_date_range(tmp_path: Path, calls: list[str]) -> None:
    client = ElexonClient(cache_dir=tmp_path)
    client.wind_forecast(date(2025, 1, 1), date(2025, 1, 10))
    client.demand_forecast(date(2025, 1, 1), date(2025, 1, 1))
    names = sorted(str(p.relative_to(tmp_path).as_posix()) for p in tmp_path.rglob("*.json"))
    assert names == [
        "NDF/2025-01-01_2025-01-01_boundary-N.json",
        "WINDFOR/2025-01-01_2025-01-07.json",
        "WINDFOR/2025-01-08_2025-01-10.json",
    ]
    assert "boundary=N" in calls[-1]


def test_second_call_is_served_from_cache_and_only_downloads_are_logged(
    tmp_path: Path, calls: list[str], caplog: pytest.LogCaptureFixture
) -> None:
    client = ElexonClient(cache_dir=tmp_path)
    with caplog.at_level(logging.INFO, logger="njord.elexon"):
        client.wind_forecast(date(2025, 1, 1), date(2025, 1, 1))
        client.wind_forecast(date(2025, 1, 1), date(2025, 1, 1))
    assert len(calls) == 1
    assert sum(r.getMessage().startswith("downloaded") for r in caplog.records) == 1


def test_recent_days_are_not_cached(tmp_path: Path, calls: list[str]) -> None:
    client = ElexonClient(cache_dir=tmp_path)
    today = date.today()
    client.wind_forecast(today, today)
    client.wind_forecast(today, today)
    assert len(calls) == 2
    assert not list(tmp_path.rglob("*.json"))


def _http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("https://x", code, "msg", {}, None)  # type: ignore[arg-type]


def test_only_transient_errors_are_retried() -> None:
    assert _is_retryable(_http_error(503))
    assert _is_retryable(_http_error(429))
    assert _is_retryable(urllib.error.URLError("dns"))
    assert _is_retryable(TimeoutError())
    assert not _is_retryable(_http_error(400))
    assert not _is_retryable(ValueError())


def test_client_errors_are_raised_with_the_url(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def bad_request(url: str) -> dict:
        raise urllib.error.HTTPError(url, 400, "Bad Request", {}, None)  # type: ignore[arg-type]

    monkeypatch.setattr(elexon, "_download", bad_request)
    with pytest.raises(RuntimeError, match="400 for https://data.elexon.co.uk/bmrs/api/v1/datasets/NDF"):
        ElexonClient(cache_dir=tmp_path).demand_forecast(date(2025, 1, 1), date(2025, 1, 1))
