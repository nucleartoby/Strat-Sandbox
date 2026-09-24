from __future__ import annotations

import os
import warnings
from typing import Iterable

import numpy as np
import pandas as pd

TICK_COLUMNS = ["timestamp", "bid", "ask", "mid", "spread", "volume"]
BLOTTER_COLUMNS = ["timestamp", "side", "price", "size", "counterparty_id"]


def _to_utc(values, fallback_format: str | None = None) -> pd.Series:
    if fallback_format is not None:
        parsed = pd.to_datetime(values, format=fallback_format, errors="coerce", utc=True)
        if not parsed.isna().all():
            return parsed
    try:
        return pd.to_datetime(values, format="ISO8601", utc=True)
    except (ValueError, TypeError):
        return pd.to_datetime(values, format="mixed", utc=True, errors="coerce")


def _finalize(df: pd.DataFrame, volume_is_proxy: bool = False) -> pd.DataFrame:
    df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    df["mid"] = (df["bid"] + df["ask"]) / 2.0
    df["spread"] = df["ask"] - df["bid"]
    if "volume" not in df.columns:
        df["volume"] = 1.0
        volume_is_proxy = True
    df.attrs["volume_is_proxy"] = volume_is_proxy
    return df


def load_dukascopy_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = {c.lower().strip(): c for c in df.columns}
    ts_col = cols.get("gmt time") or cols.get("timestamp") or cols.get("time")
    if ts_col is None:
        raise ValueError(f"{path}: no Dukascopy timestamp column found in {list(df.columns)}")

    out = pd.DataFrame()
    out["timestamp"] = _to_utc(df[ts_col], "%d.%m.%Y %H:%M:%S.%f")
    out["bid"] = df[cols["bid"]].astype(float)
    out["ask"] = df[cols["ask"]].astype(float)

    have_vol = "askvolume" in cols and "bidvolume" in cols
    if have_vol:
        out["volume"] = (df[cols["askvolume"]].astype(float)
                         + df[cols["bidvolume"]].astype(float))
    out = out.dropna(subset=["timestamp", "bid", "ask"])
    return _finalize(out, volume_is_proxy=not have_vol)


def load_duka_csv(path: str) -> pd.DataFrame:
    names = ["time", "ask", "bid", "ask_volume", "bid_volume"]
    with open(path, "r", errors="replace") as fh:
        first = fh.readline()
    has_header = "time" in first.lower() and "ask" in first.lower()

    df = pd.read_csv(path, header=0 if has_header else None,
                     names=None if has_header else names)
    df.columns = [str(c).strip().lower() for c in df.columns]

    out = pd.DataFrame()
    out["timestamp"] = _to_utc(df["time"])
    out["bid"] = df["bid"].astype(float)
    out["ask"] = df["ask"].astype(float)
    have_vol = {"ask_volume", "bid_volume"} <= set(df.columns)
    if have_vol:
        out["bid_size"] = df["bid_volume"].astype(float)
        out["ask_size"] = df["ask_volume"].astype(float)
        out["volume"] = out["bid_size"] + out["ask_size"]
    out = out.dropna(subset=["timestamp", "bid", "ask"])

    if len(out) and (out["ask"] < out["bid"]).mean() > 0.5:
        raise ValueError(
            f"{path}: most quotes are crossed, which means bid and ask are "
            "swapped. duka writes ask before bid; check the column order.")
    if len(out) and out["bid"].median() < 0.01:
        warnings.warn(
            f"{path}: median price {out['bid'].median():.6f} looks 100x too small. "
            "duka scales every instrument by 1e-5, which is wrong for JPY crosses "
            "(they quote to 3 decimals). Use data_fetch.fetch_ticks instead.",
            RuntimeWarning, stacklevel=2)
    return _finalize(out, volume_is_proxy=not have_vol)


def load_truefx_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, names=["symbol", "timestamp", "bid", "ask"])
    out = pd.DataFrame()
    out["timestamp"] = _to_utc(df["timestamp"], "%Y%m%d %H:%M:%S.%f")
    out["bid"] = df["bid"].astype(float)
    out["ask"] = df["ask"].astype(float)
    out = out.dropna(subset=["timestamp", "bid", "ask"])
    return _finalize(out, volume_is_proxy=True)


def load_histdata_csv(path: str, spread_pips: float = 0.8, pip: float = 1e-4) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", header=None,
                     names=["timestamp", "open", "high", "low", "close", "volume"])
    out = pd.DataFrame()
    out["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y%m%d %H%M%S",
                                      errors="coerce", utc=True)
    half = 0.5 * spread_pips * pip
    close = df["close"].astype(float)
    out["bid"] = close - half
    out["ask"] = close + half
    out = out.dropna(subset=["timestamp", "bid", "ask"])
    out["volume"] = 1.0
    res = _finalize(out, volume_is_proxy=True)
    res.attrs["source"] = "histdata_m1"
    return res


def load_tick_data(path: str, provider: str = "auto") -> pd.DataFrame:
    loaders = {
        "dukascopy": load_dukascopy_csv,
        "duka": load_duka_csv,
        "truefx": load_truefx_csv,
        "histdata": load_histdata_csv,
        "generic": _load_generic_csv,}
    
    if provider != "auto":
        if provider not in loaders:
            raise ValueError(f"provider must be one of {sorted(loaders)} or 'auto'")
        return loaders[provider](path)

    with open(path, "r", errors="replace") as fh:
        head = fh.readline()
    lowered = head.lower()
    if "gmt time" in lowered:
        return load_dukascopy_csv(path)
    if "ask_volume" in lowered or "bid_volume" in lowered:
        return load_duka_csv(path)
    if ";" in head and head.count(";") >= 4:
        return load_histdata_csv(path)
    if "," in head and not any(c.isalpha() for c in head.split(",")[1][:4]):
        # No header and the second field starts with digits TrueFX.
        return load_truefx_csv(path)
    return _load_generic_csv(path)


def _load_generic_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = {"timestamp", "bid", "ask"} - set(df.columns)
    if missing:
        raise ValueError(f"{path}: tick data missing required columns: {sorted(missing)}")
    df["timestamp"] = _to_utc(df["timestamp"])
    return _finalize(df, volume_is_proxy="volume" not in df.columns)


def load_trade_blotter(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = {"timestamp", "side", "price", "size"} - set(df.columns)
    if missing:
        raise ValueError(f"{path}: trade blotter missing required columns: {sorted(missing)}")
    df["timestamp"] = _to_utc(df["timestamp"])
    df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    df["side"] = df["side"].astype(str).str.lower().str.strip()

    bad = set(df["side"].unique()) - {"buy", "sell", "b", "s"}
    if bad:
        raise ValueError(f"{path}: unrecognised trade sides: {sorted(bad)}")
    df["side"] = df["side"].map({"buy": "buy", "b": "buy", "sell": "sell", "s": "sell"})

    if "counterparty_id" not in df.columns:
        df["counterparty_id"] = "unknown"
    return df


def generate_synthetic_ticks(n: int = 200_000, seed: int = 42,
                             start: str = "2026-01-05 00:00:00",
                             base_price: float = 1.1000,
                             informed_rate: float = 0.0004) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # Regime process mostly zero drift, punctuated by informed episodes
    drift = np.zeros(n)
    in_burst = np.zeros(n, dtype=bool)
    cur, remaining = 0.0, 0
    for i in range(n):
        if remaining > 0:
            remaining -= 1
        elif rng.random() < informed_rate:
            cur = rng.choice([-1.0, 1.0]) * rng.uniform(2e-6, 6e-6)
            remaining = rng.integers(200, 1200)
        else:
            cur = 0.0
        drift[i] = cur
        in_burst[i] = remaining > 0

    step_ns = rng.integers(200_000, 40_000_000, n).astype(np.int64)
    ts = pd.Timestamp(start, tz="UTC").value + np.cumsum(step_ns)
    timestamp = pd.to_datetime(ts, utc=True)
    hour = timestamp.hour.to_numpy()
    liquidity = np.where((hour >= 7) & (hour < 17), 1.0, 2.2)  # London/NY vs Asia

    vol_scale = 3e-5 * np.sqrt(liquidity)
    mid = base_price + np.cumsum(drift + rng.normal(0, 1, n) * vol_scale)
    spread = (6e-5 + 3e-5 * rng.random(n)) * liquidity

    df = pd.DataFrame({
        "timestamp": timestamp,
        "bid": mid - spread / 2,
        "ask": mid + spread / 2,
        "bid_size": 1e6 * (0.3 + rng.random(n)),
        "ask_size": 1e6 * (0.3 + rng.random(n)),
        "volume": 1000.0 * rng.integers(1, 60, n).astype(float) / liquidity,})
    
    df["informed_regime"] = in_burst
    return _finalize(df)


def generate_synthetic_trades(ticks: pd.DataFrame, n_trades: int = 5_000,
                              seed: int = 7, n_counterparties: int = 40,
                              informed_fraction: float = 0.25) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(ticks)
    if n < 10:
        raise ValueError("need at least 10 ticks to sample fills from")

    informed_cps = set(range(int(n_counterparties * informed_fraction)))
    in_burst = (ticks["informed_regime"].to_numpy() if "informed_regime" in ticks
                else np.zeros(n, dtype=bool))
    burst_idx = np.flatnonzero(in_burst[: n - 1])
    any_idx = np.arange(n - 1)

    cps = rng.integers(0, n_counterparties, n_trades)
    idx = np.empty(n_trades, dtype=np.int64)
    informed = np.array([c in informed_cps for c in cps])

    n_inf = int(informed.sum())
    if n_inf and len(burst_idx):
        idx[informed] = rng.choice(burst_idx, size=n_inf, replace=True)
    elif n_inf:
        idx[informed] = rng.choice(any_idx, size=n_inf, replace=True)
    idx[~informed] = rng.choice(any_idx, size=int((~informed).sum()), replace=True)

    order = np.argsort(idx, kind="mergesort")
    idx, cps, informed = idx[order], cps[order], informed[order]

    lookahead = np.minimum(idx + 300, n - 1)
    future_move = ticks["mid"].to_numpy()[lookahead] - ticks["mid"].to_numpy()[idx]
    side = np.where(
        informed,
        np.where(future_move > 0, 1, -1),
        rng.choice([1, -1], size=n_trades),)
    
    flip = informed & (rng.random(n_trades) < 0.15)
    side = np.where(flip, -side, side)

    rows = ticks.iloc[idx]
    price = np.where(side == 1, rows["ask"].to_numpy(), rows["bid"].to_numpy())
    size = 1000.0 * rng.integers(1, 20, n_trades) * np.where(informed, 2.0, 1.0)

    return pd.DataFrame({
        "timestamp": rows["timestamp"].to_numpy(),
        "side": np.where(side == 1, "buy", "sell"),
        "price": price,
        "size": size,
        "counterparty_id": [f"CP_{c:03d}" for c in cps],
        "is_informed": informed,}).reset_index(drop=True)
