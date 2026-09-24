from __future__ import annotations

import numpy as np
import pandas as pd


def clean_ticks(df: pd.DataFrame, max_log_spread_z: float = 6.0,
                drop_zero_spread: bool = True, verbose: bool = False) -> pd.DataFrame:
    n0 = len(df)
    df = df.sort_values("timestamp", kind="mergesort")
    df = df.drop_duplicates(subset=["timestamp", "bid", "ask"], keep="last")
    n_dupes = n0 - len(df)

    crossed = df["ask"] < df["bid"]
    locked = df["ask"] == df["bid"]
    mask = crossed | (locked if drop_zero_spread else False)
    n_crossed = int(mask.sum())
    df = df[~mask]

    spread = (df["ask"] - df["bid"]).to_numpy()
    n_outliers = 0
    if len(spread) > 2 and np.all(spread > 0):
        ls = np.log(spread)
        sd = ls.std(ddof=0)
        if sd > 0:
            keep = np.abs((ls - ls.mean()) / sd) < max_log_spread_z
            n_outliers = int((~keep).sum())
            df = df[keep]

    df = df.reset_index(drop=True)
    df["mid"] = (df["bid"] + df["ask"]) / 2.0
    df["spread"] = df["ask"] - df["bid"]

    report = {
        "rows_in": n0, "rows_out": len(df), "duplicates": n_dupes,
        "crossed_or_locked": n_crossed, "spread_outliers": n_outliers,}

    df.attrs["cleaning_report"] = report
    if verbose:
        print(f"[clean_ticks] {n0:,} -> {len(df):,} rows "
              f"(dupes {n_dupes:,}, crossed/locked {n_crossed:,}, "
              f"spread outliers {n_outliers:,})")
    return df


def flag_session_gaps(df: pd.DataFrame, max_gap_seconds: float = 300.0) -> pd.DataFrame:
    df = df.copy()
    gap = df["timestamp"].diff().dt.total_seconds()
    df["gap_seconds"] = gap.fillna(0.0)
    df["gap_flag"] = df["gap_seconds"] > max_gap_seconds
    df["session_id"] = df["gap_flag"].cumsum()
    return df


def session_label(ts: pd.Series) -> pd.Series:
    hour = ts.dt.hour
    return pd.Series(
        np.select(
            [(hour >= 12) & (hour < 17), (hour >= 7) & (hour < 12),
             (hour >= 17) & (hour < 21)],
            ["london_ny_overlap", "london", "ny_afternoon"],
            default="asia",),index=ts.index, name="session",)


def resample_ohlc(df: pd.DataFrame, freq: str = "1s") -> pd.DataFrame:
    g = df.set_index("timestamp")
    out = g["mid"].resample(freq).ohlc()
    out["spread"] = g["spread"].resample(freq).mean()
    if "volume" in g.columns:
        out["volume"] = g["volume"].resample(freq).sum()
    return out.dropna(subset=["close"]).reset_index()


def coverage_report(df: pd.DataFrame, max_gap_seconds: float = 300.0) -> pd.DataFrame:
    d = flag_session_gaps(df, max_gap_seconds)
    d["date"] = d["timestamp"].dt.date
    rep = d.groupby("date").agg(
        ticks=("timestamp", "size"),
        mean_spread=("spread", "mean"),
        max_gap_seconds=("gap_seconds", "max"),
        n_gaps=("gap_flag", "sum"),).reset_index()
    if "volume" in d.columns:
        rep["volume"] = d.groupby("date")["volume"].sum().to_numpy()
    return rep
