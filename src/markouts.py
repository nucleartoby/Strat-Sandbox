import numpy as np
import pandas as pd
from scipy.optimize import least_squares

import native

DEFAULT_HORIZONS = [1.0, 5.0, 30.0, 300.0, 1800.0]


def compute_markouts(fills: pd.DataFrame, ticks: pd.DataFrame,
                     horizons_seconds=DEFAULT_HORIZONS,
                     use_native: bool = True) -> pd.DataFrame:
    fills = fills.sort_values("timestamp", kind="mergesort").reset_index(drop=True).copy()
    mid = ticks["mid"].to_numpy() if "mid" in ticks else (
        (ticks["bid"].to_numpy() + ticks["ask"].to_numpy()) / 2.0)
    tick_ts = ticks["timestamp"]

    m = native.compute_markouts(fills["timestamp"], fills["price"].to_numpy(),
                                fills["side"].to_numpy(), tick_ts, mid,
                                horizons_seconds, use_native=use_native)
    for j, h in enumerate(horizons_seconds):
        fills[_col(h)] = m[:, j]

    fills["mid_at_fill"] = native.mid_at_fill(fills["timestamp"], tick_ts,
                                              mid, use_native=use_native)
    
    sign = np.where(fills["side"].astype(str).str.lower().isin(["buy", "b"]), 1.0, -1.0)
    fills["effective_half_spread"] = sign * (fills["price"] - fills["mid_at_fill"])
    fills.attrs["horizons"] = list(horizons_seconds)
    return fills


def _col(h) -> str:
    return f"markout_{int(h)}s" if float(h).is_integer() else f"markout_{h}s"


def markout_curve(tau, half_spread, alpha_mu, lam):
    return half_spread - alpha_mu * (1.0 - np.exp(-lam * np.asarray(tau, float)))


def _markout_residuals(params, taus, means, sigma):
    resid = markout_curve(taus, *params) - means
    return resid if sigma is None else resid / sigma


def _covariance_stderr(result, n_obs, n_params, sigma):
    _, s, VT = np.linalg.svd(result.jac, full_matrices=False)
    threshold = np.finfo(float).eps * max(result.jac.shape) * s[0]
    s = s[s > threshold]
    VT = VT[:s.size]
    pcov = np.dot(VT.T / s ** 2, VT)
    if sigma is None:
        if n_obs > n_params:
            pcov = pcov * (2.0 * result.cost) / (n_obs - n_params)
        else:
            pcov = np.full_like(pcov, np.inf)
    if np.isnan(pcov).any():
        pcov = np.full_like(pcov, np.inf)
    return np.sqrt(np.diag(pcov))


def fit_markout_curve(fills: pd.DataFrame, horizons_seconds=DEFAULT_HORIZONS,
                      weight_by_count: bool = True) -> dict:
    taus, means, counts, sems = [], [], [], []
    for h in horizons_seconds:
        col = _col(h)
        if col not in fills:
            raise ValueError(f"{col} missing; run compute_markouts with horizon {h} first")
        v = fills[col].to_numpy()
        v = v[~np.isnan(v)]
        if len(v) < 2:
            continue
        taus.append(float(h))
        means.append(v.mean())
        counts.append(len(v))
        sems.append(v.std(ddof=1) / np.sqrt(len(v)))

    taus, means = np.array(taus), np.array(means)
    counts, sems = np.array(counts), np.array(sems)
    if len(taus) < 3:
        raise ValueError("need at least 3 horizons with observed markouts to fit the curve")

    span = max(taus.max() - taus.min(), 1.0)
    p0 = [means[0], max(means[0] - means[-1], 1e-6), 1.0 / span]
    bounds = ([-np.inf, 0.0, 1e-6], [np.inf, np.inf, 10.0])
    sigma = sems if weight_by_count and np.all(sems > 0) else None

    result = least_squares(_markout_residuals, p0, bounds=bounds, max_nfev=20000,
                           args=(taus, means, sigma))
    converged = result.success
    if converged:
        popt = result.x
        perr = _covariance_stderr(result, len(taus), len(p0), sigma)
    else:
        popt, perr = np.array(p0), np.full(3, np.nan)

    half_spread, alpha_mu, lam = popt

    lam_at_bound = lam <= bounds[0][2] * 1.01 or lam >= bounds[1][2] * 0.99
    alpha_unresolved = (not np.isfinite(perr[1])) or perr[1] > abs(alpha_mu)
    identified = bool(converged and not lam_at_bound and not alpha_unresolved)

    empirical_decay = float(means[0] - means[-1])
    return {
        "half_spread": float(half_spread),
        "alpha_mu": float(alpha_mu),
        "lambda": float(lam),
        "breakeven_spread": float(2.0 * alpha_mu),
        "half_life_seconds": float(np.log(2.0) / lam) if lam > 0 else np.inf,
        "stderr": {"half_spread": perr[0], "alpha_mu": perr[1], "lambda": perr[2]},
        "converged": converged,
        "identified": identified,
        "empirical_decay": empirical_decay,
        "unidentified_reason": (
            None if identified else
            ("profile is too flat to separate alpha_mu from lambda -- "
             "lambda hit its bound" if lam_at_bound else
             "alpha_mu standard error exceeds the estimate"
             if alpha_unresolved else "fit did not converge")),
        "taus": taus,
        "empirical_means": means,
        "counts": counts,
        "fitted": markout_curve(taus, *popt),
        "curve": lambda t: markout_curve(t, *popt),}


def markout_summary(fills: pd.DataFrame, horizons_seconds=DEFAULT_HORIZONS,
                    by: str | None = None) -> pd.DataFrame:
    cols = [_col(h) for h in horizons_seconds if _col(h) in fills]
    if by is None:
        rows = [{"horizon_s": h, "mean_markout": fills[_col(h)].mean(),
                 "n_observed": int(fills[_col(h)].notna().sum())}
                for h in horizons_seconds if _col(h) in fills]
        return pd.DataFrame(rows)
    out = fills.groupby(by)[cols].mean()
    out["n_fills"] = fills.groupby(by).size()
    return out.reset_index()


def spread_vs_adverse_selection(fit: dict, fills: pd.DataFrame) -> dict:
    realized = float(fills["effective_half_spread"].mean()) if "effective_half_spread" in fills else np.nan

    if fit.get("identified", True):
        alpha_mu = fit["alpha_mu"]
        source = "fitted"
    else:
        alpha_mu = max(fit.get("empirical_decay", 0.0), 0.0)
        source = "empirical decay (curve fit unidentified)"

    return {
        "realized_half_spread": realized,
        "fitted_half_spread": fit["half_spread"],
        "alpha_mu": alpha_mu,
        "alpha_mu_source": source,
        "breakeven_spread": 2.0 * alpha_mu,
        "net_edge_per_fill": realized - alpha_mu,
        "profitable": bool(realized > alpha_mu),}
