import numpy as np
import pandas as pd

import native
from fracdiff import frac_diff_ffd, stationarity_report

NATIVE_FEATURES = list(native.FEATURE_NAMES)

FEATURE_COLUMNS = NATIVE_FEATURES + [
    "vpin",
    "vpin_pctile",
    "size",
    "cp_toxic_rate_hist",
    "cp_fill_count",
    "is_client_buy",]


def build_feature_matrix(fills: pd.DataFrame, ticks: pd.DataFrame,
                         buckets: pd.DataFrame | None = None,
                         labels: pd.Series | None = None,
                         short_window_sec: float = 60.0,
                         long_window_sec: float = 900.0,
                         cp_prior_rate: float = 0.5, cp_prior_weight: float = 5.0,
                         use_native: bool = True) -> pd.DataFrame:
    fills = fills.sort_values("timestamp", kind="mergesort").reset_index(drop=True).copy()
    if labels is not None:
        labels = labels.loc[fills.index] if labels.index.equals(fills.index) else labels
        labels = pd.Series(np.asarray(labels), index=fills.index)

    mat = native.build_features(
        ticks["timestamp"], ticks["bid"].to_numpy(), ticks["ask"].to_numpy(),
        fills["timestamp"], fills["size"].to_numpy(),
        bid_size=ticks["bid_size"].to_numpy() if "bid_size" in ticks else None,
        ask_size=ticks["ask_size"].to_numpy() if "ask_size" in ticks else None,
        volume=ticks["volume"].to_numpy() if "volume" in ticks else None,
        short_window_sec=short_window_sec, long_window_sec=long_window_sec,
        use_native=use_native,)
    
    for j, name in enumerate(NATIVE_FEATURES):
        fills[name] = mat[:, j]

    if buckets is not None and len(buckets):
        from vpin import attach_vpin_to_fills
        fills = attach_vpin_to_fills(fills, buckets, columns=("vpin", "vpin_pctile"))
    else:
        fills["vpin"] = np.nan
        fills["vpin_pctile"] = np.nan

    cp_codes = _encode_counterparty(fills)
    y = (np.asarray(labels, dtype=np.int8) if labels is not None
         else np.full(len(fills), native.LABEL_UNKNOWN, dtype=np.int8))
    rate, count = native.counterparty_history(cp_codes, y, prior_rate=cp_prior_rate,
                                              prior_weight=cp_prior_weight,
                                              use_native=use_native)
    fills["cp_toxic_rate_hist"] = rate
    fills["cp_fill_count"] = count

    fills["is_client_buy"] = (
        fills["side"].astype(str).str.lower().isin(["buy", "b"]).astype(float))
    fills["session"] = _session_from_seconds(fills["sec_of_day"])
    return fills


def _encode_counterparty(fills: pd.DataFrame) -> np.ndarray:
    if "counterparty_id" not in fills:
        return np.full(len(fills), -1, dtype=np.int32)
    codes = pd.Categorical(fills["counterparty_id"].astype(str)).codes
    return np.ascontiguousarray(codes, dtype=np.int32)


def _session_from_seconds(sec_of_day: pd.Series) -> pd.Series:
    hour = (sec_of_day / 3600.0).fillna(-1)
    return pd.Series(
        np.select(
            [(hour >= 12) & (hour < 17), (hour >= 7) & (hour < 12),
             (hour >= 17) & (hour < 21), hour >= 0],
            ["london_ny_overlap", "london", "ny_afternoon", "asia"],
            default="unknown",),index=sec_of_day.index, name="session",)


def add_fractional_differencing(features: pd.DataFrame, columns=("vpin", "spread_mean"),
                                d: float | None = None, thresh: float = 1e-4,
                                suffix: str = "_ffd") -> pd.DataFrame:
    out = features.copy()
    for col in columns:
        if col not in out:
            continue
        series = out[col]
        use_d = d if d is not None else find_min_d(series, thresh=thresh)
        out[f"{col}{suffix}"] = frac_diff_ffd(series, d=use_d, thresh=thresh)
        out.attrs.setdefault("ffd_orders", {})[col] = use_d
    return out


def find_min_d(series: pd.Series, candidates=None, thresh: float = 1e-4,
               pvalue: float = 0.05) -> float:
    candidates = candidates if candidates is not None else np.arange(0.0, 1.01, 0.1)
    for d in candidates:
        s = frac_diff_ffd(series, d=float(d), thresh=thresh).dropna()
        if len(s) < 30:
            continue
        rep = stationarity_report(s)
        if rep["adf_pvalue"] < pvalue:
            return float(d)
    return 1.0


def feature_health(features: pd.DataFrame, columns=None) -> pd.DataFrame:
    columns = columns or [c for c in FEATURE_COLUMNS if c in features]
    rows = []
    for c in columns:
        s = pd.to_numeric(features[c], errors="coerce")
        clean = s.dropna()
        row = {
            "feature": c,
            "nan_rate": float(s.isna().mean()),
            "std": float(clean.std()) if len(clean) > 1 else np.nan,
            "n_unique": int(clean.nunique()),}
        if len(clean) >= 30 and clean.nunique() > 2:
            row["adf_pvalue"] = stationarity_report(clean)["adf_pvalue"]
        else:
            row["adf_pvalue"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def select_matrix(features: pd.DataFrame, labels: pd.Series | None = None,
                  columns=None, return_rows: bool = False):
    columns = [c for c in (columns or FEATURE_COLUMNS) if c in features]
    X = features[columns].astype(float)

    mask = X.notna().all(axis=1)
    y = None
    if labels is not None:
        y = pd.Series(np.asarray(labels), index=features.index)
        mask &= y >= 0

    X_out = X.loc[mask].reset_index(drop=True)
    y_out = y.loc[mask].reset_index(drop=True) if y is not None else None
    if return_rows:
        return X_out, y_out, features.loc[mask].reset_index(drop=True)
    return X_out, y_out
