import os
import sys
import warnings
import numpy as np
import pandas as pd
import _reference as ref
import fxtox_native as _nat

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


FEATURE_NAMES = ref.FEATURE_NAMES
FEATURE_COUNT = ref.FEATURE_COUNT
LABEL_TOXIC = ref.LABEL_TOXIC
LABEL_BENIGN = ref.LABEL_BENIGN
LABEL_UNKNOWN = ref.LABEL_UNKNOWN
NS_PER_SEC = ref.NS_PER_SEC

if HAVE_NATIVE:
    _nat_names = list(_nat.feature_names())
    if _nat_names != FEATURE_NAMES:
        raise ImportError(
            "fxtox_native feature layout does not match the Python side.\n"
            f"  native:    {_nat_names}\n"
            f"  reference: {FEATURE_NAMES}\n"
            "Rebuild the extension: cmake --build build")


def warn_if_slow(n: int, threshold: int = 200_000) -> None:
    if not HAVE_NATIVE and n >= threshold:
        warnings.warn(
            f"fxtox_native is not built; falling back to the NumPy reference for "
            f"{n:,} rows. Build it with `cmake -S . -B build && cmake --build build -j` "
            f"for a large speedup.",
            RuntimeWarning, stacklevel=2,)


def _i64(a) -> np.ndarray:
    dtype = getattr(a, "dtype", None)
    if isinstance(dtype, pd.DatetimeTZDtype):
        return np.ascontiguousarray(
            pd.DatetimeIndex(a).tz_convert("UTC").view("int64"))

    arr = np.asarray(a)
    if arr.dtype.kind == "M":
        return np.ascontiguousarray(arr.astype("datetime64[ns]").view("int64"))
    if arr.dtype.kind in "OSU":
        idx = pd.DatetimeIndex(pd.to_datetime(arr, utc=True))
        return np.ascontiguousarray(idx.view("int64"))
    return np.ascontiguousarray(arr, dtype=np.int64)


def _f64(a) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(a, dtype=np.float64))


def _sides(side) -> np.ndarray:
    a = np.asarray(side)
    if a.dtype.kind in "USO":
        lowered = np.array([str(x).strip().lower() for x in a.ravel()])
        known = {"buy", "b", "sell", "s"}
        bad = sorted(set(lowered.tolist()) - known)
        if bad:
            raise ValueError(f"unrecognised trade sides: {bad[:5]}")
        return np.where(np.isin(lowered, ("buy", "b")), 1, -1).astype(np.int8)
    return np.where(np.asarray(a, dtype=np.float64) >= 0, 1, -1).astype(np.int8)


def compute_vpin(ts, bid, ask, volume, bucket_volume, window=50, dist="normal",
                 df=0.25, granularity="bucket", sigma_mode="expanding",
                 sigma_fixed=0.0, sigma_warmup=20, pctile_warmup=30, use_native=True):
    ts, bid, ask, volume = _i64(ts), _f64(bid), _f64(ask), _f64(volume)
    warn_if_slow(len(ts))
    if use_native and HAVE_NATIVE:
        return _nat.compute_vpin(
            ts, bid, ask, volume, bucket_volume=float(bucket_volume),
            window=int(window),
            dist=_nat.BvcDist.NORMAL if dist == "normal" else _nat.BvcDist.STUDENT_T,
            df=float(df),
            granularity=(_nat.BvcGranularity.BUCKET if granularity == "bucket"
                         else _nat.BvcGranularity.TICK),
            sigma_mode=(_nat.SigmaMode.EXPANDING if sigma_mode == "expanding"
                        else _nat.SigmaMode.FIXED),
            sigma_fixed=float(sigma_fixed), sigma_warmup=int(sigma_warmup),
            pctile_warmup=int(pctile_warmup),)
    if granularity != "bucket":
        raise NotImplementedError(
            "the NumPy reference implements bucket-granularity BVC only; "
            "build fxtox_native for tick granularity")
    return ref.reference_vpin(ts, bid, ask, volume, bucket_volume, window, dist,
                              df, sigma_mode, sigma_fixed, sigma_warmup,
                              pctile_warmup)


def compute_markouts(fill_ts, fill_price, fill_side, mid_ts, mid, horizons_sec,
                     use_native=True):
    fill_ts, fill_price = _i64(fill_ts), _f64(fill_price)
    side = _sides(fill_side)
    mid_ts, mid = _i64(mid_ts), _f64(mid)
    horizons = _f64(horizons_sec)
    warn_if_slow(len(mid_ts))
    if use_native and HAVE_NATIVE:
        return _nat.compute_markouts(fill_ts, fill_price, side, mid_ts, mid, horizons)
    return ref.reference_markouts(fill_ts, fill_price, side, mid_ts, mid, horizons)


def mid_at_fill(fill_ts, mid_ts, mid, use_native=True):
    fill_ts, mid_ts, mid = _i64(fill_ts), _i64(mid_ts), _f64(mid)
    if use_native and HAVE_NATIVE:
        return _nat.mid_at_fill(fill_ts, mid_ts, mid)
    return ref.reference_mid_at_fill(fill_ts, mid_ts, mid)


_RULES = ("markout_threshold", "crossback", "triple_barrier")


def label_fills(fill_ts, fill_price, fill_side, mid_ts, mid, rule="crossback",
                horizon_sec=30.0, theta=0.0, min_adverse=0.0, markouts=None,
                center="mid_at_fill", use_native=True):
    if rule not in _RULES:
        raise ValueError(f"rule must be one of {_RULES}, got {rule!r}")
    if center not in ("exec_price", "mid_at_fill"):
        raise ValueError(f"center must be 'exec_price' or 'mid_at_fill', got {center!r}")
    fill_ts, fill_price = _i64(fill_ts), _f64(fill_price)
    side = _sides(fill_side)
    mid_ts, mid = _i64(mid_ts), _f64(mid)
    mk = None if markouts is None else _f64(markouts)
    warn_if_slow(len(mid_ts))
    if use_native and HAVE_NATIVE:
        enum = {"markout_threshold": _nat.LabelRule.MARKOUT_THRESHOLD,
                "crossback": _nat.LabelRule.CROSSBACK,
                "triple_barrier": _nat.LabelRule.TRIPLE_BARRIER}[rule]
        ctr = {"exec_price": _nat.BarrierCenter.EXEC_PRICE,
               "mid_at_fill": _nat.BarrierCenter.MID_AT_FILL}[center]
        return _nat.label_fills(fill_ts, fill_price, side, mid_ts, mid, enum,
                                float(horizon_sec), float(theta),
                                float(min_adverse), ctr, mk)
    return ref.reference_labels(fill_ts, fill_price, side, mid_ts, mid, rule,
                                horizon_sec, theta, min_adverse, mk, center)


def build_features(tick_ts, bid, ask, fill_ts, fill_size, bid_size=None,
                   ask_size=None, volume=None, short_window_sec=60.0,
                   long_window_sec=900.0, use_native=True):
    tick_ts, bid, ask = _i64(tick_ts), _f64(bid), _f64(ask)
    fill_ts, fill_size = _i64(fill_ts), _f64(fill_size)
    bs = None if bid_size is None else _f64(bid_size)
    as_ = None if ask_size is None else _f64(ask_size)
    vol = None if volume is None else _f64(volume)
    warn_if_slow(len(tick_ts))
    if use_native and HAVE_NATIVE:
        return _nat.build_features(tick_ts, bid, ask, fill_ts, fill_size, bs, as_,
                                   vol, float(short_window_sec), float(long_window_sec))
    return ref.reference_features(tick_ts, bid, ask, fill_ts, fill_size, bs, as_,
                                  vol, short_window_sec, long_window_sec)


def counterparty_history(counterparty, labels, prior_rate=0.5, prior_weight=5.0,
                         use_native=True):
    cp = np.ascontiguousarray(np.asarray(counterparty), dtype=np.int32)
    y = np.ascontiguousarray(np.asarray(labels), dtype=np.int8)
    if use_native and HAVE_NATIVE:
        return _nat.counterparty_history(cp, y, float(prior_rate), float(prior_weight))
    return ref.reference_counterparty_history(cp, y, prior_rate, prior_weight)


def load_model(path):
    if not HAVE_NATIVE:
        raise RuntimeError(
            "fxtox_native is required to load .fxm models; build it with cmake. "
            "For Python-side inference use the scikit-learn/LightGBM model directly.")
    return _nat.Model.load(str(path))


def make_gate(**kwargs):
    if not HAVE_NATIVE:
        raise RuntimeError("fxtox_native is required for RiskGate; build it with cmake.")
    cfg = _nat.GateConfig()
    for k, v in kwargs.items():
        if not hasattr(cfg, k):
            raise TypeError(f"unknown GateConfig field: {k}")
        setattr(cfg, k, v)
    return _nat.RiskGate(cfg)


def backend() -> str:
    return "fxtox_native (C++)" if HAVE_NATIVE else "numpy reference"
