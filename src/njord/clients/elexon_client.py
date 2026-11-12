"""Client for the Elexon Insights (BMRS) API. No key needed.

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


def _chunks(start_inclusive: date, end_inclusive: date, days: int) -> Iterator[tuple[date, date]]:
    """Split days [start, end], both included, into chunks of at most `days` days, each with both ends included."""
    while start_inclusive <= end_inclusive:
        stop = min(start_inclusive + timedelta(days=days - 1), end_inclusive)
        yield start_inclusive, stop
        start_inclusive = stop + timedelta(days=1)


class ElexonClient:
    """Elexon Insights API with an on-disk cache: `<cache_dir>/<DATASET>/<from>_<to>.json`, one file per API call.

    Data for the last two days is never cached because it may still be filling in or be revised. Settlement data
    is also re-run for months afterwards (II/SF/R1...); delete a dataset's folder to pick those up.
    """

    def __init__(self, cache_dir: Path | None = None) -> None:
        self.cache_dir = cache_dir or Path.home() / ".njord"

    def get_system_prices(self, start_inclusive: date, end_inclusive: date) -> pd.DataFrame:
        """System (imbalance) prices and net imbalance volume, for settlement dates in [start, end], both included."""
        days = pd.date_range(start_inclusive, end_inclusive).date
        frames = [self._get(f"balancing/settlement/system-prices/{d}", "SYSTEM_PRICES", d, d) for d in days]
        return self._concat("SYSTEM_PRICES", frames)

    def get_market_index(self, start_inclusive: date, end_inclusive: date) -> pd.DataFrame:
        """Market index prices (MID), one row per provider (APXMIDP, N2EXMIDP), for UTC days [start, end] included."""
        return self._get_range(
            "balancing/pricing/market-index", "MID", start_inclusive, end_inclusive, from_param="from", to_param="to"
        )

    def get_generation_by_fuel(self, start_inclusive: date, end_inclusive: date) -> pd.DataFrame:
        """Generation by fuel type (FUELHH, MW), for settlement dates in [start, end], both included."""
        return self._get_range(
            "datasets/FUELHH",
            "FUELHH",
            start_inclusive,
            end_inclusive,
            from_param="settlementDateFrom",
            to_param="settlementDateTo",
        )

    def get_wind_forecast(self, start_inclusive: date, end_inclusive: date) -> pd.DataFrame:
        """National wind forecast (WINDFOR), for runs published on UTC days [start, end], both included."""
        return self._get_range("datasets/WINDFOR", "WINDFOR", start_inclusive, end_inclusive)

    def get_demand_forecast(self, start_inclusive: date, end_inclusive: date, boundary: str = "N") -> pd.DataFrame:
        """National demand forecast (NDF), for runs published on UTC days [start, end], both included.

        Transmission demand, so embedded generation is already netted off.
        """
        return self._get_range("datasets/NDF", "NDF", start_inclusive, end_inclusive, chunk_days=1, boundary=boundary)

    def get_margin_forecast(self, start_inclusive: date, end_inclusive: date, boundary: str = "N") -> pd.DataFrame:
        """Indicated margin (MELNGC), for runs published on UTC days [start, end], both included."""
        return self._get_range(
            "datasets/MELNGC", "MELNGC", start_inclusive, end_inclusive, chunk_days=1, boundary=boundary
        )

    def get_imbalance_forecast(self, start_inclusive: date, end_inclusive: date, boundary: str = "N") -> pd.DataFrame:
        """Indicated imbalance (IMBALNGC), for runs published on UTC days [start, end], both included."""
        return self._get_range(
            "datasets/IMBALNGC", "IMBALNGC", start_inclusive, end_inclusive, chunk_days=1, boundary=boundary
        )

    def _get(
        self,
        path: str,
        dataset: str,
        start_inclusive: date,
        end_inclusive: date,
        range_params: dict[str, str] | None = None,
        **params: str,
    ) -> pd.DataFrame:
        """One API call -> DataFrame, cached as `<dataset>/<start>_<end>[_<param>-<value>...].json`.

        The dates only name the cache file and decide whether to cache; the query itself is in `range_params`.
        """
        query = {**(range_params or {}), **params}
        url = BASE_URL + path + ("?" + urlencode(query) if query else "")
        file = (
            self.cache_dir
            / dataset
            / (f"{start_inclusive}_{end_inclusive}" + "".join(f"_{k}-{v}" for k, v in params.items()) + ".json")
        )
        cache = end_inclusive < datetime.now(UTC).date() - timedelta(days=2)
        if cache and file.exists():
            log.debug("cache hit %s", file)
            payload = json.loads(file.read_text())
        else:
            t0 = time.perf_counter()
            try:
                payload = _download(url)
            except urllib.error.HTTPError as e:
                raise RuntimeError(f"{e.code} for {url}: {e.read()[:500]!r}") from e
            log.info(
                "downloaded %s %s to %s in %.1f s", dataset, start_inclusive, end_inclusive, time.perf_counter() - t0
            )
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
        start_inclusive: date,
        end_inclusive: date,
        *,
        from_param: str = "publishDateTimeFrom",
        to_param: str = "publishDateTimeTo",
        chunk_days: int = 7,
        **params: str,
    ) -> pd.DataFrame:
        """Fetch days [start, end], both included, in chunks within the API's range limit (7 days, or 1 for big ones).

        `settlementDate*` params take dates; the rest take datetimes, sent as whole UTC days.
        """
        as_dates = from_param.startswith("settlementDate")
        chunks = list(_chunks(start_inclusive, end_inclusive, chunk_days))
        log.info("%s: fetching %s to %s in %d chunk(s)", dataset, start_inclusive, end_inclusive, len(chunks))
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
