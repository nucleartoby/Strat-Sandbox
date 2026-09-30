import numpy as np
import pandas as pd
import native


def compute_vpin(ticks: pd.DataFrame, bucket_volume: float, window: int = 50,
                 dist: str = "normal", df: float = 0.25,
                 granularity: str = "bucket", sigma_mode: str = "expanding",
                 sigma_fixed: float = 0.0, sigma_warmup: int = 20,
                 pctile_warmup: int = 30, use_native: bool = True) -> pd.DataFrame:
    if "volume" not in ticks.columns:
        raise ValueError("ticks need a 'volume' column; VPIN is defined in volume time")
    if not bucket_volume > 0:
        raise ValueError("bucket_volume must be > 0")

    out = native.compute_vpin(
        ticks["timestamp"], ticks["bid"].to_numpy(),
        ticks["ask"].to_numpy(), ticks["volume"].to_numpy(),
        bucket_volume=bucket_volume, window=window, dist=dist, df=df,
        granularity=granularity, sigma_mode=sigma_mode, sigma_fixed=sigma_fixed,
        sigma_warmup=sigma_warmup, pctile_warmup=pctile_warmup,
        use_native=use_native,)
    res = pd.DataFrame({
        "timestamp_end": pd.to_datetime(np.asarray(out["ts_end"]), utc=True),
        "vpin": out["vpin"],
        "vpin_pctile": out["percentile"],
        "order_imbalance": out["imbalance"],
        "v_buy": out["v_buy"],
        "v_sell": out["v_sell"],
        "price_end": out["price_end"],
        "warmup": np.asarray(out["warmup"], dtype=bool),})
    res.attrs["bucket_volume"] = bucket_volume
    res.attrs["window"] = window
    res.attrs["backend"] = native.backend()
    return res


def suggest_bucket_volume(ticks: pd.DataFrame, buckets_per_day: int = 50,
                          vpin_window: int = 50, min_buckets_per_window: int = 8) -> float:
    total = float(ticks["volume"].sum())
    span_days = max(
        (ticks["timestamp"].iloc[-1] - ticks["timestamp"].iloc[0]).total_seconds() / 86400.0,
        1e-9,)
    
    by_rate = total / (span_days * buckets_per_day)
    target_buckets = max(buckets_per_day, min_buckets_per_window * max(vpin_window, 1))
    floor = total / target_buckets
    return float(min(by_rate, floor))


def full_sample_sigma(ticks: pd.DataFrame, bucket_volume: float) -> float:
    vol = ticks["volume"].to_numpy()
    mid = ticks["mid"].to_numpy() if "mid" in ticks else (
        (ticks["bid"].to_numpy() + ticks["ask"].to_numpy()) / 2.0)
    cum = np.cumsum(vol)
    n_buckets = int(cum[-1] // bucket_volume)
    if n_buckets < 2:
        return 0.0
    idx = np.clip(np.searchsorted(cum, bucket_volume * np.arange(1, n_buckets + 1),
                                  side="left"), 0, len(mid) - 1)
    prices = mid[idx]
    dp = np.diff(np.concatenate(([mid[0]], prices)))
    return float(dp.std(ddof=0))


def vpin_regime(vpin_pctile: float, high: float = 0.90, low: float = 0.35) -> str:
    """Discrete regime label from a VPIN percentile."""
    if np.isnan(vpin_pctile):
        return "unknown"
    if vpin_pctile >= high:
        return "toxic"
    if vpin_pctile <= low:
        return "benign"
    return "neutral"


def attach_vpin_to_fills(fills: pd.DataFrame, buckets: pd.DataFrame,
                         columns: tuple = ("vpin", "vpin_pctile")) -> pd.DataFrame:
    left = fills.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    right = (buckets[["timestamp_end", *columns]]
             .dropna(subset=["timestamp_end"])
             .sort_values("timestamp_end", kind="mergesort"))
    merged = pd.merge_asof(left, right, left_on="timestamp", right_on="timestamp_end",
                           direction="backward", allow_exact_matches=True)
    for c in columns:
        left[c] = merged[c].to_numpy()
    return left
