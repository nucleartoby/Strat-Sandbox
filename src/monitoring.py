import numpy as np
import pandas as pd

from modeling import evaluate, reliability_curve


def pnl_by_toxicity_decile(fills: pd.DataFrame, toxicity_proba,
                           pnl_column: str = "markout_30s",
                           n_bins: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({
        "proba": np.asarray(toxicity_proba, dtype=float),
        "pnl": fills[pnl_column].to_numpy(dtype=float),}).dropna()
    if df.empty:
        return pd.DataFrame()

    df["decile"] = pd.qcut(df["proba"], n_bins, labels=False, duplicates="drop")
    out = df.groupby("decile").agg(
        n=("pnl", "size"),
        mean_proba=("proba", "mean"),
        mean_pnl=("pnl", "mean"),
        total_pnl=("pnl", "sum"),
        pnl_std=("pnl", "std"),).reset_index()
    out["cumulative_pnl"] = out["total_pnl"].cumsum()
    return out


def monotonicity_score(decile_table: pd.DataFrame) -> float:
    if len(decile_table) < 3:
        return np.nan
    return float(decile_table["mean_proba"].corr(decile_table["mean_pnl"], method="spearman"))


def gate_value_analysis(fills: pd.DataFrame, toxicity_proba, threshold: float,
                        pnl_column: str = "markout_30s",
                        spread_cost_per_fill: float = 0.0) -> dict:
    p = np.asarray(toxicity_proba, dtype=float)
    pnl = fills[pnl_column].to_numpy(dtype=float)
    ok = ~(np.isnan(p) | np.isnan(pnl))
    p, pnl = p[ok], pnl[ok]
    if len(pnl) == 0:
        return {}

    blocked = p >= threshold
    avoided = pnl[blocked]
    kept = pnl[~blocked]
    forgone = spread_cost_per_fill * int((avoided > 0).sum())

    return {
        "threshold": threshold,
        "n_fills": len(pnl),
        "n_blocked": int(blocked.sum()),
        "blocked_share": float(blocked.mean()),
        "pnl_ungated": float(pnl.sum()),
        "pnl_gated": float(kept.sum()) - forgone,
        "pnl_improvement": float(kept.sum()) - forgone - float(pnl.sum()),
        "mean_pnl_blocked": float(avoided.mean()) if len(avoided) else np.nan,
        "mean_pnl_kept": float(kept.mean()) if len(kept) else np.nan,
        "opportunity_cost": forgone,}


def sweep_gate_thresholds(fills: pd.DataFrame, toxicity_proba,
                          thresholds=None, pnl_column: str = "markout_30s",
                          spread_cost_per_fill: float = 0.0) -> pd.DataFrame:
    thresholds = thresholds if thresholds is not None else np.arange(0.5, 0.96, 0.05)
    rows = [gate_value_analysis(fills, toxicity_proba, float(t), pnl_column,
                                spread_cost_per_fill) for t in thresholds]
    return pd.DataFrame([r for r in rows if r])


def drift_report(y_true, proba, reference_metrics: dict | None = None,
                 n_bins: int = 10) -> dict:
    current = evaluate(y_true, proba)
    curve = reliability_curve(y_true, proba, n_bins=n_bins)
    out = {"current": current, "reliability": curve,
           "max_calibration_error": float(curve["calibration_error"].max())
           if len(curve) else np.nan}
    if reference_metrics:
        out["delta"] = {k: current.get(k, np.nan) - reference_metrics.get(k, np.nan)
                        for k in ("auc", "brier", "brier_skill")
                        if k in current and k in reference_metrics}
    return out


def should_retrain(fold_results: pd.DataFrame, auc_floor: float = 0.55,
                   brier_skill_floor: float = 0.02, lookback: int = 3) -> dict:
    if fold_results is None or fold_results.empty:
        return {"retrain": True, "reason": "no fold results available"}

    recent = fold_results.tail(lookback)
    reasons = []
    if "auc" in recent and recent["auc"].mean() < auc_floor:
        reasons.append(f"AUC {recent['auc'].mean():.3f} < {auc_floor}")
    if "brier_skill" in recent and recent["brier_skill"].mean() < brier_skill_floor:
        reasons.append(f"Brier skill {recent['brier_skill'].mean():.3f} < {brier_skill_floor}")

    return {"retrain": bool(reasons), "reason": "; ".join(reasons) or "within tolerance",
            "recent_auc": float(recent["auc"].mean()) if "auc" in recent else np.nan,
            "recent_brier_skill": float(recent["brier_skill"].mean())
            if "brier_skill" in recent else np.nan}


def alpha_mu_drift(alpha_mu_history: pd.Series, window: int = 20,
                   z_threshold: float = 2.0) -> pd.DataFrame:
    s = pd.Series(alpha_mu_history, dtype=float)
    roll_mean = s.rolling(window, min_periods=5).mean()
    roll_std = s.rolling(window, min_periods=5).std().replace(0, np.nan)
    z = (s - roll_mean) / roll_std
    return pd.DataFrame({"alpha_mu": s, "rolling_mean": roll_mean,
                         "zscore": z, "drifted": z.abs() > z_threshold})
