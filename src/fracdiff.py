from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller


def ffd_weights(d: float, thresh: float = 1e-4, max_width: int = 10_000) -> np.ndarray:
    w = [1.0]
    for k in range(1, max_width):
        w_k = -w[-1] * (d - k + 1) / k
        if abs(w_k) < thresh:
            break
        w.append(w_k)
    return np.array(w[::-1])  # oldest observation first


def frac_diff_ffd(series: pd.Series, d: float, thresh: float = 1e-4) -> pd.Series:
    s = pd.Series(series).astype(float)
    if d == 0:
        return s.copy()

    w = ffd_weights(d, thresh)
    width = len(w)
    values = s.to_numpy()
    out = np.full(len(values), np.nan)
    if len(values) < width:
        return pd.Series(out, index=s.index, name=s.name)
    windows = np.lib.stride_tricks.sliding_window_view(values, width)
    out[width - 1:] = windows @ w
    return pd.Series(out, index=s.index, name=s.name)


def stationarity_report(series: pd.Series, max_lag: int | None = None) -> dict:
    s = pd.Series(series).astype(float).dropna()
    result = {"n": len(s), "adf_stat": np.nan, "adf_pvalue": np.nan,
              "is_stationary_5pct": False}
    if len(s) < 20 or s.nunique() < 3:
        return result
    stat, pvalue, *_ = adfuller(s.to_numpy(), maxlag=max_lag, autolag="AIC")
    result.update(adf_stat=float(stat), adf_pvalue=float(pvalue),
                  is_stationary_5pct=bool(pvalue < 0.05))
    return result


def min_ffd_order(series: pd.Series, candidates=None, thresh: float = 1e-4,
                  pvalue: float = 0.05) -> pd.DataFrame:
    candidates = candidates if candidates is not None else np.arange(0.0, 1.01, 0.05)
    base = pd.Series(series).astype(float)
    rows = []
    for d in candidates:
        diffed = frac_diff_ffd(base, d=float(d), thresh=thresh)
        joined = pd.concat([base, diffed], axis=1).dropna()
        corr = (float(joined.corr().iloc[0, 1]) if len(joined) > 2
                and joined.iloc[:, 1].nunique() > 1 else np.nan)
        rep = stationarity_report(diffed)
        rows.append({"d": float(d), "adf_stat": rep["adf_stat"],
                     "adf_pvalue": rep["adf_pvalue"],
                     "stationary": rep["is_stationary_5pct"],
                     "corr_with_level": corr, "n_obs": rep["n"]})
    return pd.DataFrame(rows)
