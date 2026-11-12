"""Minimal client for the Elexon Insights (BMRS) API. No key needed.

Docs: https://bmrs.elexon.co.uk/api-documentation
"""

import json
import logging
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import pandas as pd
from tenacity import before_sleep_log, retry, retry_if_exception, stop_after_attempt, wait_exponential

BASE_URL = "https://data.elexon.co.uk/bmrs/api/v1/"
RETRY_CODES = (429, 500, 502, 503, 504)

log = logging.getLogger(__name__)


def _is_retryable(e: BaseException) -> bool:
    if isinstance(e, urllib.error.HTTPError):
        return e.code in RETRY_CODES
    return isinstance(e, urllib.error.URLError | TimeoutError)


@retry(
    retry=retry_if_exception(_is_retryable),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, max=8),
    before_sleep=before_sleep_log(log, logging.WARNING),
    reraise=True,
)
def _download(url: str) -> Any:  # noqa: ANN401
    with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310 (fixed https base URL)
        return json.load(resp)


def _chunks(start: date, end: date, days: int) -> Iterator[tuple[date, date]]:
    """Split inclusive [start, end] into inclusive chunks of at most `days` days."""
    while start <= end:
        stop = min(start + timedelta(days=days - 1), end)
        yield start, stop
        start = stop + timedelta(days=1)


class ElexonClient:
    """Elexon Insights API with an on-disk cache: `<cache_dir>/<DATASET>/<from>_<to>.json`, one file per API call.

    Data for the last two days is never cached because it may still be filling in or be revised. Settlement data
    is also re-run for months afterwards (II/SF/R1...); delete a dataset's folder to pick those up.
    """

    def __init__(self, cache_dir: Path | None = None) -> None:
        self.cache_dir = cache_dir or Path.home() / ".njord"

    # --- Market prices -----------------------------------------------------------------------------------------

    def system_prices(self, start: date, end: date) -> pd.DataFrame:
        """System (imbalance) prices and net imbalance volume per settlement period."""
        days = pd.date_range(start, end).date
        frames = [self._get(f"balancing/settlement/system-prices/{d}", "SYSTEM_PRICES", d, d) for d in days]
        return self._concat("SYSTEM_PRICES", frames)

    def market_index(self, start: date, end: date) -> pd.DataFrame:
        """Market index prices (MID) per settlement period, one row per data provider (APXMIDP, N2EXMIDP)."""
        return self._get_range("balancing/pricing/market-index", "MID", start, end, from_param="from", to_param="to")

    # --- Outturn -------------------------------------------------------------------------------------------------

    def generation_by_fuel(self, start: date, end: date) -> pd.DataFrame:
        """Half-hourly transmission-metered generation by fuel type (FUELHH), in MW."""
        return self._get_range(
            "datasets/FUELHH", "FUELHH", start, end, from_param="settlementDateFrom", to_param="settlementDateTo"
        )

    # --- NESO forecasts, as published (publishTime = when it became available) ----------------------------------

    def wind_forecast(self, start: date, end: date) -> pd.DataFrame:
        """National wind generation forecast (WINDFOR), hourly, published by `publishTime` in [start, end]."""
        return self._get_range("datasets/WINDFOR", "WINDFOR", start, end)

    def demand_forecast(self, start: date, end: date, boundary: str = "N") -> pd.DataFrame:
        """Day-ahead national demand forecast (NDF). Transmission demand, so embedded generation is netted off."""
        return self._get_range("datasets/NDF", "NDF", start, end, chunk_days=1, boundary=boundary)

    def margin_forecast(self, start: date, end: date, boundary: str = "N") -> pd.DataFrame:
        """Indicated margin (MELNGC): available generation minus demand forecast."""
        return self._get_range("datasets/MELNGC", "MELNGC", start, end, chunk_days=1, boundary=boundary)

    def imbalance_forecast(self, start: date, end: date, boundary: str = "N") -> pd.DataFrame:
        """Indicated imbalance (IMBALNGC): generators' planned output minus demand forecast."""
        return self._get_range("datasets/IMBALNGC", "IMBALNGC", start, end, chunk_days=1, boundary=boundary)

    # --- Internals -----------------------------------------------------------------------------------------------

    def _get(
        self, path: str, dataset: str, start: date, end: date, range_params: dict[str, str] | None = None, **params: str
    ) -> pd.DataFrame:
        """One API call -> DataFrame, cached as `<dataset>/<start>_<end>[_<param>-<value>...].json`."""
        query = {**(range_params or {}), **params}
        url = BASE_URL + path + ("?" + urlencode(query) if query else "")
        file = self.cache_dir / dataset / (f"{start}_{end}" + "".join(f"_{k}-{v}" for k, v in params.items()) + ".json")
        cache = end < datetime.now(UTC).date() - timedelta(days=2)
        if cache and file.exists():
            log.debug("cache hit %s", file)
            payload = json.loads(file.read_text())
        else:
            t0 = time.perf_counter()
            try:
                payload = _download(url)
            except urllib.error.HTTPError as e:
                raise RuntimeError(f"{e.code} for {url}: {e.read()[:500]!r}") from e
            log.info("downloaded %s %s to %s in %.1f s", dataset, start, end, time.perf_counter() - t0)
            if cache:
                file.parent.mkdir(parents=True, exist_ok=True)
                file.write_text(json.dumps(payload))
        df = pd.DataFrame(payload["data"] if isinstance(payload, dict) else payload)
        for col in df.columns:
            if col.endswith("Time") or col.startswith("time"):
                df[col] = pd.to_datetime(df[col], utc=True)
        return df

    def _get_range(
        self,
        path: str,
        dataset: str,
        start: date,
        end: date,
        *,
        from_param: str = "publishDateTimeFrom",
        to_param: str = "publishDateTimeTo",
        chunk_days: int = 7,
        **params: str,
    ) -> pd.DataFrame:
        """Fetch inclusive [start, end] in chunks that respect the API's range limit (7 days, or 1 for big datasets).

        `settlementDate*` params take dates; the rest take datetimes, sent as whole UTC days.
        """
        as_dates = from_param.startswith("settlementDate")
        chunks = list(_chunks(start, end, chunk_days))
        log.info("%s: fetching %s to %s in %d chunk(s)", dataset, start, end, len(chunks))
        frames = []
        for a, b in chunks:
            lo, hi = (str(a), str(b)) if as_dates else (f"{a}T00:00Z", f"{b + timedelta(days=1)}T00:00Z")
            if not as_dates and chunk_days == 1:
                hi = f"{b}T23:59Z"  # the 1-day limit is inclusive, so a full 24 h is rejected
            frames.append(self._get(path, dataset, a, b, {from_param: lo, to_param: hi}, **params))
        return self._concat(dataset, frames)

    @staticmethod
    def _concat(dataset: str, frames: list[pd.DataFrame]) -> pd.DataFrame:
        # Datetime chunks share their boundary instant, so the same row can come back twice.
        df = pd.concat(frames, ignore_index=True).drop_duplicates(ignore_index=True)
        log.info("%s: %d rows", dataset, len(df))
        return df
