"""End-to-end: the pipeline runs, and recovers the signal planted in the data."""
import numpy as np
import pytest

import native
from cv import walk_forward_splits
from data_cleaning import clean_ticks, coverage_report
from data_ingestion import generate_synthetic_ticks, generate_synthetic_trades
from features import build_feature_matrix, feature_health, select_matrix
from integration import gate_summary, meta_label_gate, required_spread, simulate_gate
from labeling import drop_undecidable, label_fills, suggest_theta
from markouts import compute_markouts, fit_markout_curve, markout_summary
from modeling import cross_validate, forward_select
from monitoring import monotonicity_score, pnl_by_toxicity_decile, sweep_gate_thresholds
from vpin import compute_vpin, suggest_bucket_volume

HORIZONS = [1.0, 5.0, 30.0, 300.0]
LABEL_HORIZON = 60.0


@pytest.fixture(scope="module")
def built():
    ticks = clean_ticks(generate_synthetic_ticks(n=120_000, seed=101))
    fills = compute_markouts(
        generate_synthetic_trades(ticks, n_trades=5_000, seed=101), ticks, HORIZONS)
    buckets = compute_vpin(ticks, suggest_bucket_volume(ticks, 50, vpin_window=30),
                           window=30, dist="student_t")
    theta = suggest_theta(fills, ticks, horizon_seconds=LABEL_HORIZON)["theta"]
    labels = label_fills(fills, ticks, rule="triple_barrier",
                         horizon_seconds=LABEL_HORIZON, theta=theta)
    fills, labels = drop_undecidable(fills, labels)
    feats = build_feature_matrix(fills, ticks, buckets, labels=labels)
    X, y, rows = select_matrix(feats, labels, return_rows=True)
    return dict(ticks=ticks, fills=fills, buckets=buckets, labels=labels,
                feats=feats, X=X, y=y, rows=rows)


def test_cleaning_reports_what_it_removed(built):
    report = built["ticks"].attrs["cleaning_report"]
    assert report["rows_out"] <= report["rows_in"]
    assert set(report) >= {"duplicates", "crossed_or_locked", "spread_outliers"}
    assert len(coverage_report(built["ticks"])) >= 1


def test_markouts_decay_and_the_curve_fits(built):
    fills = built["fills"]
    summary = markout_summary(fills, HORIZONS)
    means = summary["mean_markout"].to_numpy()
    # Adverse selection accumulates: the LP does worse the longer it holds.
    assert means[0] > means[-1], f"markout profile does not decay: {means}"

    fit = fit_markout_curve(fills, HORIZONS)
    assert fit["converged"]
    assert fit["alpha_mu"] > 0
    assert fit["half_life_seconds"] > 0
    assert np.isclose(fit["breakeven_spread"], 2 * fit["alpha_mu"])


def test_informed_flow_costs_the_lp_more(built):
    """The planted cohort must show up in the markouts."""
    by_cohort = markout_summary(built["fills"], HORIZONS, by="is_informed")
    informed = by_cohort.loc[by_cohort["is_informed"], "markout_300s"].iloc[0]
    other = by_cohort.loc[~by_cohort["is_informed"], "markout_300s"].iloc[0]
    assert informed < other, "informed flow should mark out worse for the LP"


def test_feature_matrix_is_usable(built):
    X, feats = built["X"], built["feats"]
    assert len(X) > 1000, "too many rows dropped"
    assert X.notna().all().all()
    health = feature_health(feats)
    # No feature should be entirely missing -- that silently empties the matrix.
    assert (health["nan_rate"] < 1.0).all(), health[health["nan_rate"] >= 1.0]


def test_model_recovers_the_signal(built):
    """Beat the base rate out of sample, on both ranking and calibration."""
    X, y, rows = built["X"], built["y"], built["rows"]
    splits = list(walk_forward_splits(rows["timestamp"], LABEL_HORIZON, n_splits=4))

    selected = forward_select(X, y, splits, max_features=4)
    assert selected["selected"], "no feature cleared the improvement threshold"
    # Counterparty history is the feature that identifies informed flow here.
    assert "cp_toxic_rate_hist" in selected["selected"]

    res = cross_validate(X[selected["selected"]], y, splits, kind="logreg")
    folds = res["fold_results"]
    assert folds["auc"].mean() > 0.53, f"no ranking signal: {folds['auc'].mean():.3f}"
    assert folds["brier_skill"].mean() > 0.0, (
        f"worse calibrated than the base rate: {folds['brier_skill'].mean():+.4f}")


def test_gate_improves_realized_pnl(built):
    """The economic test: does acting on the signal actually help?"""
    X, y, fills = built["X"], built["y"], built["rows"]
    splits = list(walk_forward_splits(fills["timestamp"], LABEL_HORIZON, n_splits=4))

    res = cross_validate(X[["cp_toxic_rate_hist"]], y, splits, kind="logreg")
    proba = res["oof_proba"].fillna(res["oof_proba"].mean())

    deciles = pnl_by_toxicity_decile(fills, proba, pnl_column="markout_30s")
    assert len(deciles) >= 3
    assert monotonicity_score(deciles) < 0, "PnL should fall as predicted toxicity rises"

    sweep = sweep_gate_thresholds(fills, proba, pnl_column="markout_30s")
    assert sweep["pnl_improvement"].max() > 0, "gating never helps at any threshold"


@pytest.mark.skipif(not native.HAVE_NATIVE, reason="RiskGate lives in the C++ core")
def test_live_gate_does_not_flap(built):
    """Hysteresis and dwell must keep the quoting state from churning."""
    fills = built["rows"]
    rng = np.random.default_rng(0)
    # A probability hovering right on the widen threshold: the worst case.
    proba = np.clip(0.65 + rng.normal(0, 0.02, len(fills)), 0, 1)

    sim = simulate_gate(fills["timestamp"], proba, fills["spread"].to_numpy(),
                        alpha_mu=1e-4)
    summary = gate_summary(sim)
    assert summary.attrs["changes_per_1k"] < 400, (
        f"gate flaps: {summary.attrs['changes_per_1k']:.0f} changes per 1k updates")

    no_hyst = simulate_gate(fills["timestamp"], proba, fills["spread"].to_numpy(),
                            alpha_mu=1e-4, hysteresis=0.0, min_dwell_seconds=0.0)
    assert (gate_summary(no_hyst).attrs["changes_per_1k"]
            > summary.attrs["changes_per_1k"]), "hysteresis had no effect"


def test_meta_label_gate_only_shrinks():
    """A risk gate must never increase a position."""
    import pandas as pd
    signal = pd.Series([1, 1, -1, -1])
    out = meta_label_gate(signal, [0.0, 0.9, 0.1, 0.99], act_threshold=0.5)
    assert (out["size_multiplier"] <= 1.0).all()
    assert (out["size_multiplier"] >= 0.0).all()
    assert (out.loc[~out["act"], "gated_signal"] == 0).all()


def test_required_spread_tracks_alpha_mu():
    assert np.isclose(required_spread(0.5, 1e-4), 1e-4)
    assert np.isclose(required_spread(0.0, 1e-4), 0.0)
    assert required_spread(0.9, 2e-4) > required_spread(0.9, 1e-4)
