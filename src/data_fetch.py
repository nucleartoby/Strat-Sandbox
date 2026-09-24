from __future__ import annotations

import concurrent.futures
import random
import time
import urllib.error
import urllib.request
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# The decoder is dukas only
from duka.core.processor import decompress_lzma, tokenize

URL = ("https://www.dukascopy.com/datafeed/{symbol}/{year}/{month:02d}/{day:02d}"
       "/{hour:02d}h_ticks.bi5")

_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

DEFAULT_CACHE = Path.home() / ".cache" / "fxtox" / "dukascopy"


def point_value(symbol: str) -> float:
    sym = symbol.upper()
    if "JPY" in sym:
        return 1e-3
    if sym.startswith("XAU") or sym.startswith("XAG"):
        return 1e-3
    return 1e-5


# Dukas status retry
_RETRY_STATUS = frozenset({301, 302, 307, 308, 408, 429, 500, 502, 503, 504})


def _fetch_hour(symbol: str, day: date, hour: int, cache_dir: Path,
                max_attempts: int = 6, timeout: float = 30.0) -> bytes:
    cache = cache_dir / symbol.upper() / f"{day:%Y-%m-%d}" / f"{hour:02d}h.bi5"
    if cache.exists():
        return cache.read_bytes()

    url = URL.format(symbol=symbol.upper(), year=day.year, month=day.month - 1,
                     day=day.day, hour=hour)  # Dukascopy months are 0 indexed
    req = urllib.request.Request(url, headers={"User-Agent": _UA})

    last_error = None
    for attempt in range(max_attempts):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                payload = resp.read()
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_bytes(payload)
            return payload
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code == 404:
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_bytes(b"")
                return b""
            if exc.code not in _RETRY_STATUS:
                raise
            wait = float(exc.headers.get("Retry-After") or 0) or (2 ** attempt)
            time.sleep(min(wait, 60.0) + random.uniform(0, 0.5))
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
            time.sleep(min(2 ** attempt, 60.0) + random.uniform(0, 0.5))

    raise RuntimeError(f"failed to fetch {url} after {max_attempts} attempts: {last_error}")


def _decode_hour(payload: bytes, day: date, hour: int, scale: float) -> list:
    if not payload:
        return []
    records = tokenize(decompress_lzma(payload))
    base = datetime(day.year, day.month, day.day) + timedelta(hours=hour)
    rows = []
    for ms, ask, bid, ask_vol, bid_vol in records:
        rows.append((
            base + timedelta(milliseconds=ms),
            bid * scale,
            ask * scale,
            # Volumes are in millions of base currency
            float(bid_vol) * 1e6,
            float(ask_vol) * 1e6,))
    return rows


def fetch_day(symbol: str, day: date, cache_dir=DEFAULT_CACHE,
              max_workers: int = 2) -> pd.DataFrame:
    cache_dir = Path(cache_dir)
    scale = point_value(symbol)

    payloads: dict = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_fetch_hour, symbol, day, h, cache_dir): h
                   for h in range(24)}
        for fut in concurrent.futures.as_completed(futures):
            hour = futures[fut]
            payloads[hour] = fut.result()

    rows: list = []
    for hour in range(24):  # rebuild in order the pool completes out of order
        rows.extend(_decode_hour(payloads.get(hour, b""), day, hour, scale))

    return _frame(rows)


def _frame(rows: list) -> pd.DataFrame:
    cols = ["timestamp", "bid", "ask", "bid_size", "ask_size"]
    if not rows:
        empty = pd.DataFrame({c: pd.Series(dtype="float64") for c in cols})
        empty["timestamp"] = pd.Series(dtype="datetime64[ns, UTC]")
        empty["mid"] = empty["spread"] = empty["volume"] = pd.Series(dtype="float64")
        return empty

    df = pd.DataFrame(rows, columns=cols)
    # Dukascopy timestamps are GMT
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    df["mid"] = (df["bid"] + df["ask"]) / 2.0
    df["spread"] = df["ask"] - df["bid"]
    df["volume"] = df["bid_size"] + df["ask_size"]
    df.attrs["volume_is_proxy"] = False
    return df


def fetch_ticks(symbol: str, start, end, cache_dir=DEFAULT_CACHE,
                max_workers: int = 2, verbose: bool = True) -> pd.DataFrame:
    start = _as_date(start)
    end = _as_date(end)
    if start > end:
        raise ValueError(f"start {start} is after end {end}")

    frames = []
    day = start
    total = (end - start).days + 1
    done = 0
    while day <= end:
        done += 1
        if day.weekday() == 5:  # Saturday no data published
            day += timedelta(days=1)
            continue
        frame = fetch_day(symbol, day, cache_dir=cache_dir, max_workers=max_workers)
        if verbose:
            print(f"  [{done}/{total}] {symbol} {day}: {len(frame):,} ticks")
        if len(frame):
            frames.append(frame)
        day += timedelta(days=1)

    if not frames:
        raise ValueError(
            f"no ticks for {symbol} between {start} and {end}. Check the symbol "
            "spelling (EURUSD, not EUR/USD) and that the range is not all weekend.")
    
    out = pd.concat(frames, ignore_index=True)
    out.attrs["volume_is_proxy"] = False
    out.attrs["source"] = f"dukascopy:{symbol}"
    return out


def _as_date(value) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    return pd.Timestamp(value).date()


def to_pipeline_csv(df: pd.DataFrame, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["timestamp", "bid", "ask", "volume", "bid_size", "ask_size"]
    df[[c for c in cols if c in df]].to_csv(path, index=False)
    return path


def to_engine_csv(df: pd.DataFrame, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({
        "ts_ns": pd.DatetimeIndex(df["timestamp"]).view("int64"),
        "bid": df["bid"], "ask": df["ask"], "volume": df["volume"],
        "bid_size": df.get("bid_size", 0.0), "ask_size": df.get("ask_size", 0.0),})
    
    out.to_csv(path, index=False)
    return path
