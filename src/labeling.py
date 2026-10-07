from __future__ import annotations

import numpy as np
import pandas as pd

import native

LABEL_TOXIC = native.LABEL_TOXIC
LABEL_BENIGN = native.LABEL_BENIGN
LABEL_UNKNOWN = native.LABEL_UNKNOWN


def label_fills(fills: pd.DataFrame, ticks: pd.DataFrame, rule: str = "triple_barrier",
                horizon_seconds: float = 60.0, theta: float = 0.0,
                min_adverse: float = 0.0, markout_column: str | None = None,
                center: str = "mid_at_fill", use_native: bool = True) -> pd.Series:
    mid = ticks["mid"].to_numpy() if "mid" in ticks else (
        (ticks["bid"].to_numpy() + ticks["ask"].to_numpy()) / 2.0)

    mk = None
    if rule == "markout_threshold":
        col = markout_column or f"markout_{int(horizon_seconds)}s"
        if col not in fills:
            raise ValueError(f"{col} missing; run compute_markouts first")
        mk = fills[col].to_numpy()

    labels = native.label_fills(
        fills["timestamp"], fills["price"].to_numpy(), fills["side"].to_numpy(),
        ticks["timestamp"], mid, rule=rule, horizon_sec=horizon_seconds,
        theta=theta, min_adverse=min_adverse, markouts=mk, center=center,
        use_native=use_native,)
    
    return pd.Series(np.asarray(labels), index=fills.index, name="toxic_label")


def suggest_theta(fills: pd.DataFrame, ticks: pd.DataFrame,
                  horizon_seconds: float = 60.0, vol_multiple: float = 1.0,
                  sample: int = 5000) -> dict:
    mid = ticks["mid"].to_numpy()
    ts = ticks["timestamp"]

    ts_ns = native._i64(ts)
    horizon_ns = np.int64(round(horizon_seconds * 1e9))
    step = max(len(mid) // max(sample, 1), 1)
    start_idx = np.arange(0, len(mid), step)
    end_idx = np.searchsorted(ts_ns, ts_ns[start_idx] + horizon_ns, side="right") - 1
    ok = (end_idx > start_idx) & (end_idx < len(mid))
    if not ok.any():
        raise ValueError(
            f"no {horizon_seconds}s windows fit inside the tick data; "
            "use a shorter label horizon or more data")
    
    moves = np.abs(mid[end_idx[ok]] - mid[start_idx[ok]])
    horizon_vol = float(np.median(moves))
    theta = vol_multiple * horizon_vol

    y = label_fills(fills, ticks, rule="triple_barrier",
                    horizon_seconds=horizon_seconds, theta=theta)
    resolved = y[y >= 0]
    rate = float(resolved.mean()) if len(resolved) else np.nan
    return {
        "theta": theta,
        "horizon_vol": horizon_vol,
        "vol_multiple": vol_multiple,
        "toxic_rate": rate,
        "n_resolved": int(len(resolved)),}


def barrier_rate_curve(fills: pd.DataFrame, ticks: pd.DataFrame,
                       thetas=None, rule: str = "triple_barrier",
                       horizon_seconds: float = 60.0) -> pd.DataFrame:
    thetas = thetas if thetas is not None else np.geomspace(1e-6, 1e-2, 25)
    rows = []
    for th in thetas:
        kw = ({"theta": float(th)} if rule == "triple_barrier"
              else {"min_adverse": float(th)})
        y = label_fills(fills, ticks, rule=rule, horizon_seconds=horizon_seconds, **kw)
        resolved = y[y >= 0]
        rows.append({"theta": float(th),
                     "toxic_rate": float(resolved.mean()) if len(resolved) else np.nan,
                     "n_resolved": int(len(resolved))})
    return pd.DataFrame(rows)


def calibrate_crossback(fills: pd.DataFrame, ticks: pd.DataFrame,
                        target_rate: float = 0.30, horizon_seconds: float = 60.0,
                        lo: float = 1e-7, hi: float = 1e-1, iters: int = 40) -> dict:
    def rate_for(bar: float) -> float:
        y = label_fills(fills, ticks, rule="crossback",
                        horizon_seconds=horizon_seconds, min_adverse=bar)
        resolved = y[y >= 0]
        return float(resolved.mean()) if len(resolved) else np.nan

    lo_rate, hi_rate = rate_for(lo), rate_for(hi)
    if not (hi_rate <= target_rate <= lo_rate):
        return {"min_adverse": np.nan, "achieved_rate": np.nan,
                "target_rate": target_rate, "bracket": (lo_rate, hi_rate),
                "error": f"target {target_rate:.2f} is outside the achievable range "
                         f"[{hi_rate:.3f}, {lo_rate:.3f}] on this data"}

    for _ in range(iters):
        bar = float(np.sqrt(lo * hi))
        r = rate_for(bar)
        if np.isnan(r):
            break
        if r > target_rate:
            lo = bar
        else:
            hi = bar
    bar = float(np.sqrt(lo * hi))
    return {"min_adverse": bar, "achieved_rate": rate_for(bar),
            "target_rate": target_rate, "bracket": (lo_rate, hi_rate)}


def label_report(labels: pd.Series, fills: pd.DataFrame | None = None,
                 by: str | None = None) -> pd.DataFrame:
    def summarize(y: pd.Series) -> dict:
        n = len(y)
        resolved = y[y >= 0]
        return {
            "n": n,
            "toxic": int((y == LABEL_TOXIC).sum()),
            "benign": int((y == LABEL_BENIGN).sum()),
            "undecidable": int((y == LABEL_UNKNOWN).sum()),
            "toxic_rate": float(resolved.mean()) if len(resolved) else np.nan,}

    if by is None or fills is None:
        return pd.DataFrame([summarize(labels)])
    rows = []
    for key, idx in fills.groupby(by).groups.items():
        row = {by: key}
        row.update(summarize(labels.loc[idx]))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("toxic_rate", ascending=False).reset_index(drop=True)


def drop_undecidable(fills: pd.DataFrame, labels: pd.Series) -> tuple:
    keep = labels >= 0
    return fills.loc[keep].reset_index(drop=True), labels.loc[keep].reset_index(drop=True)
