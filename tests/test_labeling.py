"""Labeling rules: direction, barrier centring, and the undecidable class."""
import numpy as np
import pandas as pd
import pytest

from data_cleaning import clean_ticks
from data_ingestion import generate_synthetic_ticks, generate_synthetic_trades
from labeling import (LABEL_BENIGN, LABEL_TOXIC, LABEL_UNKNOWN, barrier_rate_curve,
                      calibrate_crossback, drop_undecidable, label_fills,
                      label_report, suggest_theta)
from markouts import compute_markouts


@pytest.fixture(scope="module")
def data():
    ticks = clean_ticks(generate_synthetic_ticks(n=60_000, seed=5))
    fills = compute_markouts(generate_synthetic_trades(ticks, n_trades=3_000, seed=5), ticks)
    return ticks, fills


def _flat_market(price=1.1000, n=120, spread=1e-4):
    ts = pd.date_range("2026-03-02", periods=n, freq="1s", tz="UTC")
    return pd.DataFrame({
        "timestamp": ts, "bid": price - spread / 2, "ask": price + spread / 2,
        "mid": price, "spread": spread, "volume": 1000.0,
    })


def test_crossback_is_not_trivially_satisfied_on_a_flat_market():
    """The direction trap: a client buy fills at the ask, above the mid.

    Reading "has the mid reached the entry price" from the wrong side makes it
    true at the first tick for every buy, and labels nothing toxic.
    """
    ticks = _flat_market()
    fills = pd.DataFrame({
        "timestamp": [ticks["timestamp"].iloc[0]] * 2,
        "price": [ticks["ask"].iloc[0], ticks["bid"].iloc[0]],
        "size": [1e6, 1e6], "side": ["buy", "sell"],
    })
    y = label_fills(fills, ticks, rule="crossback", horizon_seconds=30.0)
    # A flat market never reaches either entry: the LP keeps the spread.
    assert list(y) == [LABEL_BENIGN, LABEL_BENIGN]


def test_crossback_fires_when_the_market_runs_through_the_entry():
    ticks = _flat_market(n=120).copy()
    ticks.loc[30:, "mid"] += 5e-4  # market gaps up through the client's entry
    ticks["bid"] = ticks["mid"] - 5e-5
    ticks["ask"] = ticks["mid"] + 5e-5

    fills = pd.DataFrame({
        "timestamp": [ticks["timestamp"].iloc[0]] * 2,
        "price": [ticks["ask"].iloc[0], ticks["bid"].iloc[0]],
        "size": [1e6, 1e6], "side": ["buy", "sell"],
    })
    y = label_fills(fills, ticks, rule="crossback", horizon_seconds=60.0)
    assert y.iloc[0] == LABEL_TOXIC    # client bought, market ran up: LP underwater
    assert y.iloc[1] == LABEL_BENIGN   # client sold into a rally: LP made money


def test_crossback_rate_is_monotone_in_min_adverse(data):
    """Raising the bar can only reduce the toxic rate -- which is what makes
    the crossback rule safe to bisect on."""
    ticks, fills = data
    rates = [
        label_fills(fills, ticks, rule="crossback", horizon_seconds=60.0,
                    min_adverse=b).pipe(lambda y: y[y >= 0].mean())
        for b in (0.0, 1e-4, 5e-4, 2e-3, 1e-2)
    ]
    assert rates == sorted(rates, reverse=True), rates
    assert rates[0] > 0.8 and rates[-1] < 0.2


def test_triple_barrier_centres_on_the_mid_not_the_execution_price(data):
    """Centring on the execution price biases every label toward benign.

    A fill sits half a spread from the mid, so the adverse barrier is always
    that much closer. With a small theta the asymmetry decides nearly every
    label, and the toxic rate collapses instead of sitting at the coin-flip
    baseline a symmetric barrier should produce.
    """
    ticks, fills = data
    theta = 1e-5  # small relative to the spread
    mid_centred = label_fills(fills, ticks, rule="triple_barrier",
                              horizon_seconds=60.0, theta=theta, center="mid_at_fill")
    exec_centred = label_fills(fills, ticks, rule="triple_barrier",
                               horizon_seconds=60.0, theta=theta, center="exec_price")

    mid_rate = mid_centred[mid_centred >= 0].mean()
    exec_rate = exec_centred[exec_centred >= 0].mean()
    assert 0.40 < mid_rate < 0.60, f"symmetric barrier should be a coin flip, got {mid_rate:.3f}"
    assert exec_rate < 0.15, f"exec-centred should collapse, got {exec_rate:.3f}"


def test_undecidable_labels_are_at_the_end_and_are_dropped(data):
    """Fills whose horizon runs past the data must not be coerced to benign."""
    ticks, fills = data
    y = label_fills(fills, ticks, rule="triple_barrier", horizon_seconds=600.0,
                    theta=1e-3)
    unknown = np.flatnonzero(y == LABEL_UNKNOWN)
    assert len(unknown) > 0
    # They are the most recent fills, which is exactly why zeroing them would
    # teach the model that recent flow is safe.
    assert unknown.min() > len(y) * 0.5

    kept_fills, kept_labels = drop_undecidable(fills, y)
    assert (kept_labels >= 0).all()
    assert len(kept_fills) == len(kept_labels) == int((y >= 0).sum())


def test_labels_separate_the_planted_informed_cohort(data):
    """End-to-end check that the labels find flow we know is toxic."""
    ticks, fills = data
    theta = suggest_theta(fills, ticks, horizon_seconds=60.0)["theta"]
    y = label_fills(fills, ticks, rule="triple_barrier", horizon_seconds=60.0, theta=theta)
    report = label_report(y, fills, by="is_informed")
    rates = dict(zip(report["is_informed"], report["toxic_rate"]))
    assert rates[True] > rates[False] + 0.1, (
        f"informed cohort not separated: {rates}")


def test_suggest_theta_scales_with_the_horizon(data):
    ticks, fills = data
    short = suggest_theta(fills, ticks, horizon_seconds=10.0)
    long = suggest_theta(fills, ticks, horizon_seconds=300.0)
    assert long["horizon_vol"] > short["horizon_vol"]
    # A wider barrier relative to the same horizon must catch fewer fills.
    tight = suggest_theta(fills, ticks, horizon_seconds=60.0, vol_multiple=0.5)
    wide = suggest_theta(fills, ticks, horizon_seconds=60.0, vol_multiple=3.0)
    assert tight["toxic_rate"] > wide["toxic_rate"]


def test_calibrate_crossback_hits_its_target(data):
    ticks, fills = data
    res = calibrate_crossback(fills, ticks, target_rate=0.25, horizon_seconds=60.0)
    assert "error" not in res, res
    assert abs(res["achieved_rate"] - 0.25) < 0.03


def test_calibrate_crossback_reports_an_unreachable_target(data):
    ticks, fills = data
    res = calibrate_crossback(fills, ticks, target_rate=0.999, horizon_seconds=60.0)
    assert "error" in res, "should refuse rather than return a meaningless bar"


def test_markout_threshold_rule(data):
    ticks, fills = data
    y = label_fills(fills, ticks, rule="markout_threshold", horizon_seconds=30.0,
                    theta=1e-4, markout_column="markout_30s")
    assert set(np.unique(y)).issubset({-1, 0, 1})
    toxic = fills.loc[y == LABEL_TOXIC, "markout_30s"]
    assert (toxic < -1e-4).all()


def test_rejects_unknown_rule(data):
    ticks, fills = data
    with pytest.raises(ValueError, match="rule must be one of"):
        label_fills(fills, ticks, rule="nonsense")
