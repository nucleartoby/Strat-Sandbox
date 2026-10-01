"""The verdict must follow the evidence, and the fit must admit when it failed."""
import numpy as np
import pandas as pd
import pytest

from markouts import compute_markouts, fit_markout_curve, spread_vs_adverse_selection
from report import build_summary, charts


def _fit(means, taus=(1.0, 5.0, 30.0, 300.0, 1800.0)):
    """Fit the curve to a synthetic markout profile."""
    fills = pd.DataFrame({f"markout_{int(t)}s": np.full(500, m)
                          + np.random.default_rng(0).normal(0, 1e-8, 500)
                          for t, m in zip(taus, means)})
    return fit_markout_curve(fills, list(taus))


def test_flat_profile_is_reported_as_unidentified():
    """A flat markout curve cannot separate alpha_mu from lambda.

    curve_fit returns happily anyway, with an alpha_mu whose standard error
    dwarfs it and lambda pinned to its bound. Left unflagged that number sets
    the breakeven spread and then the live gate's quoted spread -- a book
    earning a fifth of a pip gets told to quote 26.
    """
    fit = _fit([4e-6, 4e-6, 4e-6, 4e-6, 4e-6])
    assert not fit["identified"]
    assert fit["unidentified_reason"]


def test_genuine_decay_is_identified():
    fit = _fit([1e-4, 8e-5, 4e-5, 5e-6, 1e-6])
    assert fit["identified"], fit["unidentified_reason"]
    assert fit["alpha_mu"] > 0
    assert 0 < fit["half_life_seconds"] < 1e5


def test_unidentified_fit_falls_back_to_empirical_decay():
    """The economics must not inherit a meaningless alpha_mu."""
    fit = _fit([4e-6, 4e-6, 4e-6, 4e-6, 4e-6])
    fills = pd.DataFrame({"effective_half_spread": np.full(100, 5e-6)})
    econ = spread_vs_adverse_selection(fit, fills)
    assert "empirical" in econ["alpha_mu_source"]
    # A flat profile has no erosion, so the cost is ~0, not the fitted 1e-3.
    assert econ["alpha_mu"] < 1e-5
    assert econ["breakeven_spread"] < 1e-4


def _summary(*, skill, mono, uplift, net_edge_pips):
    folds = pd.DataFrame({"auc": [0.6], "brier_skill": [skill]})
    sweep = pd.DataFrame({"threshold": [0.5], "blocked_share": [0.15],
                          "pnl_ungated": [1.0], "pnl_gated": [1.0 + uplift / 100],
                          "pnl_improvement": [uplift / 100]})
    ts = pd.date_range("2024-01-01", periods=200, freq="min", tz="UTC")
    fills = pd.DataFrame({
        "timestamp": ts,
        "markout_1s": 1e-4 * 0.05,
        "markout_30s": 1e-4 * net_edge_pips,
    })
    return build_summary(
        symbol="TEST", ticks=fills, fills=fills,
        labels=pd.Series(np.zeros(200, dtype=np.int8)),
        fit={"taus": [1.0, 30.0], "empirical_means": [5e-6, 1e-6]},
        econ={"alpha_mu": 1e-5, "alpha_mu_source": "fitted"},
        cv_result={"fold_results": folds, "feature_names": []},
        deciles=None, sweep=sweep, monotonicity=mono,
        horizons=[1.0, 30.0], label_horizon=30.0)


@pytest.mark.parametrize("skill,mono,uplift,edge,expected", [
    (0.03, -0.8, 70.0, 0.05, "PROFITABLE, TOXIC SUBSET"),
    (0.03, -0.8, 70.0, -0.05, "LOSING TO TOXIC FLOW"),
    (-0.01, -0.1, 0.0, -0.05, "LOSING MONEY (no toxic subset found)"),
    (-0.01, -0.1, 0.0, 0.05, "BENIGN"),
])
def test_verdict_follows_the_evidence(skill, mono, uplift, edge, expected):
    """A book can earn overall while a minority of its flow bleeds.

    Collapsing that to one word loses the distinction that decides what to do,
    so the verdict names both axes.
    """
    assert _summary(skill=skill, mono=mono, uplift=uplift,
                    net_edge_pips=edge)["verdict"] == expected


def test_benign_flow_reports_no_gate_value():
    s = _summary(skill=-0.01, mono=-0.1, uplift=0.0, net_edge_pips=0.05)
    assert s["gate"] is None
    assert s["status"] == "good"


def test_charts_render_without_counterparties(tmp_path):
    """A blotter with no counterparty ids must still produce a chart."""
    s = _summary(skill=0.03, mono=-0.8, uplift=70.0, net_edge_pips=0.05)
    assert s["worst_counterparties"] is None
    out = charts(path=tmp_path / "r.png", summary=s,
                 fills=pd.DataFrame({"markout_30s": [1e-5]}), buckets=None,
                 deciles=None,
                 sweep=pd.DataFrame({"threshold": [0.5, 0.6],
                                     "blocked_share": [0.1, 0.05],
                                     "pnl_ungated": [1.0, 1.0],
                                     "pnl_gated": [1.7, 1.2],
                                     "pnl_improvement": [0.7, 0.2]}),
                 fit={"taus": [1.0, 30.0], "empirical_means": [5e-6, 1e-6],
                      "identified": False, "converged": True},
                 horizons=[1.0, 30.0])
    assert (tmp_path / "r.png").exists()
    assert (tmp_path / "r.png").stat().st_size > 5000
