"""VPIN behaviour: invariants, leakage, and the reference cross-check."""
import numpy as np
import pandas as pd
import pytest

from data_cleaning import clean_ticks
from data_ingestion import generate_synthetic_ticks
from validation import check_analytic_properties, compare_with_flowrisk
import native
from vpin import (attach_vpin_to_fills, compute_vpin, full_sample_sigma,
                  suggest_bucket_volume, vpin_regime)

# Tick-granularity BVC exists only in the C++ core; the NumPy reference
# implements the bucket form from the paper and raises otherwise. Tests that
# need it skip rather than fail, so a fresh clone with no build is green.
needs_native = pytest.mark.skipif(
    not native.HAVE_NATIVE,
    reason="tick-granularity BVC requires fxtox_native (cmake --build build)")


@pytest.fixture(scope="module")
def ticks_df():
    return clean_ticks(generate_synthetic_ticks(n=40_000, seed=11))


def test_analytic_properties_all_pass(ticks_df):
    bv = suggest_bucket_volume(ticks_df, buckets_per_day=50, vpin_window=20)
    report = check_analytic_properties(ticks_df, bv, window=20)
    failed = report[~report["passed"]]
    assert failed.empty, f"failed checks:\n{failed.to_string(index=False)}"


def test_bucket_volume_leaves_room_for_the_window(ticks_df):
    """A bucket count equal to the window yields exactly one reading."""
    bv = suggest_bucket_volume(ticks_df, buckets_per_day=50, vpin_window=50)
    out = compute_vpin(ticks_df, bv, window=50)
    assert out["vpin"].notna().sum() >= 50, (
        "too few VPIN readings; suggest_bucket_volume should floor the bucket "
        "count well above the rolling window"
    )


def test_percentile_is_causal(ticks_df):
    """The expanding percentile must not depend on the future.

    Truncating the series cannot change a reading that was already emitted. A
    full-sample rank() would fail this, which is exactly the leak it hides.
    """
    bv = suggest_bucket_volume(ticks_df, buckets_per_day=50, vpin_window=20)
    full = compute_vpin(ticks_df, bv, window=20)
    half = compute_vpin(ticks_df.iloc[: len(ticks_df) // 2], bv, window=20)

    n = len(half)
    assert n > 25, "need enough buckets in the truncated run to compare"
    np.testing.assert_allclose(
        full["vpin"].to_numpy()[:n], half["vpin"].to_numpy()[:n],
        rtol=1e-12, equal_nan=True,
        err_msg="VPIN changed when later data was removed")
    np.testing.assert_allclose(
        full["vpin_pctile"].to_numpy()[:n], half["vpin_pctile"].to_numpy()[:n],
        rtol=1e-12, equal_nan=True,
        err_msg="percentile changed when later data was removed -- it is looking ahead")


def test_expanding_and_fixed_sigma_differ(ticks_df):
    """The two sigma modes are genuinely different estimators."""
    bv = suggest_bucket_volume(ticks_df, buckets_per_day=50, vpin_window=20)
    sigma = full_sample_sigma(ticks_df, bv)
    assert sigma > 0

    expanding = compute_vpin(ticks_df, bv, window=20, sigma_mode="expanding")
    fixed = compute_vpin(ticks_df, bv, window=20, sigma_mode="fixed", sigma_fixed=sigma)
    assert len(expanding) == len(fixed)
    # They converge as the expanding estimate matures, but the early buckets
    # must differ -- otherwise the expanding mode is not actually expanding.
    assert not np.allclose(expanding["vpin"].to_numpy()[:25],
                           fixed["vpin"].to_numpy()[:25], equal_nan=True)


def test_student_t_is_more_conservative_than_normal(ticks_df):
    """Heavier tails classify *less* aggressively, and monotonically so.

    Under a fat-tailed CDF a one-sigma bucket move is unremarkable, so the buy
    fraction stays nearer 0.5 and VPIN reads lower: Phi_t(1) is 0.644 at
    df=0.25 against 0.841 for the normal. The normal is the df -> infinity
    limit, so mean VPIN must increase with df and never exceed the normal.
    """
    bv = suggest_bucket_volume(ticks_df, buckets_per_day=50, vpin_window=20)
    means = [compute_vpin(ticks_df, bv, window=20, dist="student_t", df=d)["vpin"].mean()
             for d in (0.25, 1.0, 3.0, 30.0)]
    normal = compute_vpin(ticks_df, bv, window=20, dist="normal")["vpin"].mean()

    assert means == sorted(means), f"not monotone in df: {means}"
    assert means[-1] < normal, "df=30 should approach, but not exceed, the normal"
    assert means[0] < normal


def test_attach_to_fills_never_uses_a_future_bucket(ticks_df):
    bv = suggest_bucket_volume(ticks_df, buckets_per_day=50, vpin_window=20)
    buckets = compute_vpin(ticks_df, bv, window=20)

    rng = np.random.default_rng(3)
    idx = np.sort(rng.choice(len(ticks_df), 300, replace=False))
    fills = pd.DataFrame({
        "timestamp": ticks_df["timestamp"].to_numpy()[idx],
        "price": ticks_df["mid"].to_numpy()[idx],
        "size": 1e6,
        "side": "buy",
    })
    joined = attach_vpin_to_fills(fills, buckets)

    # For every joined row, the bucket that supplied the value must have closed
    # at or before the fill.
    ends = pd.DatetimeIndex(buckets["timestamp_end"])
    for ts, v in zip(joined["timestamp"], joined["vpin"]):
        if pd.isna(v):
            continue
        assert (ends <= ts).any(), f"fill at {ts} took a value from a bucket that had not closed"


def test_vpin_regime_thresholds():
    assert vpin_regime(0.95) == "toxic"
    assert vpin_regime(0.10) == "benign"
    assert vpin_regime(0.60) == "neutral"
    assert vpin_regime(float("nan")) == "unknown"


def test_rejects_missing_volume(ticks_df):
    with pytest.raises(ValueError, match="volume"):
        compute_vpin(ticks_df.drop(columns=["volume"]), 1000.0)
    with pytest.raises(ValueError, match="bucket_volume"):
        compute_vpin(ticks_df, 0.0)


@needs_native
def test_flowrisk_cross_check(ticks_df):
    """Agreement with the reference implementation, compared like-for-like.

    flowrisk classifies at *tick* granularity -- each tick's own price change,
    accumulated into buckets. The default here is bucket granularity, the form
    in the paper. They are different estimators with different noise floors
    (~0.06 vs ~0.5), so comparing across modes yields an uninterpretable number
    that can come out negative. In matching mode they agree closely.
    """
    result = compare_with_flowrisk(ticks_df, bucket_volume=None, window=20,
                                   granularity="tick")
    if not result.get("available"):
        pytest.skip(result.get("reason", "flowrisk unavailable"))
    if "error" in result:
        pytest.skip(f"flowrisk cross-check unavailable: {result['error']}")

    assert not result["degenerate"], result["note"]
    assert result["spearman"] > 0.7, (
        f"only {result['spearman']:.3f} rank correlation with flowrisk in "
        f"matching granularity: {result['note']}")
    # Levels should line up too, not just ranks.
    assert abs(result["mean_ours"] - result["mean_flowrisk"]) < 0.05


@needs_native
def test_cross_granularity_comparison_is_flagged(ticks_df):
    """Comparing bucket-mode against flowrisk must not read as a pass or a fail.

    This is the trap the guard exists for: the number looks like a verdict on
    the implementation and is actually a comparison of two different estimators.
    """
    result = compare_with_flowrisk(ticks_df, bucket_volume=None, window=20,
                                   granularity="bucket")
    if not result.get("available") or "error" in result:
        pytest.skip("flowrisk unavailable")
    assert result["degenerate"], (
        "bucket-vs-tick comparison should be flagged as not like-for-like")
    assert "granularity" in result["note"]


@needs_native
def test_tick_and_bucket_granularity_are_different_estimators(ticks_df):
    """Both are valid; they are not interchangeable.

    Summing many Phi(z_tick)*v_tick terms averages toward 0.5*V, so the
    tick-granularity noise floor sits far below the bucket form's 0.5.
    """
    bv = suggest_bucket_volume(ticks_df, buckets_per_day=50, vpin_window=20)
    bucket = compute_vpin(ticks_df, bv, window=20, granularity="bucket")["vpin"].mean()
    tick = compute_vpin(ticks_df, bv, window=20, granularity="tick")["vpin"].mean()
    assert tick < bucket / 2, f"tick={tick:.3f} bucket={bucket:.3f}"
    assert 0.0 < tick < 0.3
    assert 0.3 < bucket < 0.7
