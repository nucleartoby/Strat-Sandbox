from __future__ import annotations

import numpy as np
import pandas as pd

from cv import _as_ns
from modeling import evaluate


def holdout_split(event_times: pd.Series, horizon_seconds: float,
                  test_frac: float = 0.25, embargo_frac: float = 0.01) -> tuple:
    ts = _as_ns(event_times)
    n = len(ts)
    test_lo = int(n * (1.0 - test_frac))
    if test_lo < 1 or test_lo >= n:
        raise ValueError(f"test_frac={test_frac} leaves an empty development or test set")
    horizon_ns = np.int64(round(horizon_seconds * 1e9))
    embargo_ns = np.int64(round(embargo_frac * max(ts[-1] - ts[0], 1)))
    cutoff = ts[test_lo] - embargo_ns
    dev = np.flatnonzero(ts[:test_lo] + horizon_ns < cutoff)
    test = np.arange(test_lo, n)
    return dev, test


def hour_blocks(event_times: pd.Series) -> np.ndarray:
    return _as_ns(event_times) // np.int64(3_600 * 1_000_000_000)


def block_bootstrap(stat, blocks: np.ndarray, n_boot: int = 300, ci: float = 0.95,
                    seed: int = 0) -> tuple:
    uniq, inverse = np.unique(blocks, return_inverse=True)
    members = [np.flatnonzero(inverse == k) for k in range(len(uniq))]
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(members), len(members))
        v = stat(np.concatenate([members[k] for k in pick]))
        if np.isfinite(v):
            draws.append(v)
    if not draws:
        return np.nan, np.nan
    lo, hi = np.quantile(draws, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return float(lo), float(hi)


def choose_threshold(sweep: pd.DataFrame) -> float | None:
    if sweep is None or sweep.empty:
        return None
    row = sweep.loc[sweep["pnl_improvement"].idxmax()]
    return float(row["threshold"]) if row["pnl_improvement"] > 0 else None


def gate_uplift(pnl: np.ndarray, proba: np.ndarray, threshold: float) -> float:
    total = pnl.sum()
    if total == 0:
        return np.nan
    return float(-pnl[proba >= threshold].sum() / abs(total))


def holdout_scores(y: np.ndarray, proba: np.ndarray, blocks: np.ndarray,
                   n_boot: int = 300) -> dict:
    y = np.asarray(y)
    proba = np.asarray(proba, dtype=float)
    out = evaluate(y, proba)

    def metric(name):
        def f(idx):
            if len(np.unique(y[idx])) < 2:
                return np.nan
            return evaluate(y[idx], proba[idx])[name]
        return f

    out["auc_ci"] = block_bootstrap(metric("auc"), blocks, n_boot)
    out["brier_skill_ci"] = block_bootstrap(metric("brier_skill"), blocks, n_boot)
    return out


def gate_scores(pnl: np.ndarray, proba: np.ndarray, threshold: float,
                blocks: np.ndarray, n_boot: int = 300) -> dict:
    pnl = np.asarray(pnl, dtype=float)
    proba = np.asarray(proba, dtype=float)
    ok = ~(np.isnan(pnl) | np.isnan(proba))
    pnl, proba, blocks = pnl[ok], proba[ok], blocks[ok]
    blocked = proba >= threshold
    return {
        "threshold": threshold,
        "blocked_share": float(blocked.mean()),
        "pnl_ungated": float(pnl.sum()),
        "pnl_gated": float(pnl[~blocked].sum()),
        "uplift": gate_uplift(pnl, proba, threshold),
        "uplift_ci": block_bootstrap(lambda idx: gate_uplift(pnl[idx], proba[idx], threshold),
                                     blocks, n_boot),
        "mean_pnl_blocked": float(pnl[blocked].mean()) if blocked.any() else np.nan,
        "mean_pnl_kept": float(pnl[~blocked].mean()) if (~blocked).any() else np.nan,}
