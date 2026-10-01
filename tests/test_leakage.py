"""
Leakage tests.

These are the tests that matter most. Every leak in this pipeline produces a
model that scores well and is worth nothing live, and none of them show up in
AUC, Brier score, or any other metric you would normally look at. The only way
to catch them is to assert causality directly.
"""
import numpy as np
import pandas as pd
import pytest

from cv import assert_no_leakage, purged_kfold_splits, walk_forward_splits
from data_cleaning import clean_ticks
from data_ingestion import generate_synthetic_ticks, generate_synthetic_trades
from features import build_feature_matrix, select_matrix
from labeling import label_fills, suggest_theta
from markouts import compute_markouts
from vpin import compute_vpin, suggest_bucket_volume
import native


@pytest.fixture(scope="module")
def pipeline():
    ticks = clean_ticks(generate_synthetic_ticks(n=60_000, seed=13))
    fills = compute_markouts(generate_synthetic_trades(ticks, n_trades=2_500, seed=13), ticks)
    buckets = compute_vpin(ticks, suggest_bucket_volume(ticks, 50, vpin_window=20), window=20)
    theta = suggest_theta(fills, ticks, horizon_seconds=60.0)["theta"]
    labels = label_fills(fills, ticks, rule="triple_barrier",
                         horizon_seconds=60.0, theta=theta)
    return ticks, fills, buckets, labels


def test_features_ignore_everything_after_the_fill(pipeline):
    """Truncating the tick stream at the last fill must not change a feature.

    If any feature reads a quote after its fill, dropping those quotes changes
    the value. This is the single most important assertion in the suite.
    """
    ticks, fills, buckets, labels = pipeline
    subset = fills.head(500).reset_index(drop=True)
    cutoff = subset["timestamp"].max()

    full = build_feature_matrix(subset, ticks, buckets, labels=labels.head(500))
    truncated_ticks = ticks[ticks["timestamp"] <= cutoff].reset_index(drop=True)
    truncated = build_feature_matrix(subset, truncated_ticks, buckets,
                                     labels=labels.head(500))

    for col in native.FEATURE_NAMES:
        np.testing.assert_allclose(
            full[col].to_numpy(), truncated[col].to_numpy(), rtol=1e-12,
            equal_nan=True,
            err_msg=f"feature {col!r} changed when post-fill quotes were removed")


def test_counterparty_history_excludes_the_fill_itself(pipeline):
    """The feature must not contain the label it is used to predict.

    An expanding groupby mean that includes the current row makes this a
    perfect predictor in-sample -- and useless out of it.
    """
    ticks, fills, buckets, labels = pipeline
    feats = build_feature_matrix(fills, ticks, buckets, labels=labels)

    # A counterparty's first fill can only ever see the prior.
    first = feats.groupby("counterparty_id").head(1)
    assert np.allclose(first["cp_toxic_rate_hist"], 0.5), (
        "first fill for a counterparty should read the prior, not its own label")
    assert (first["cp_fill_count"] == 0).all()

    # Flipping a fill's own label must not change that fill's feature.
    flipped = labels.copy()
    resolved = np.flatnonzero(np.asarray(flipped) >= 0)
    flipped.iloc[resolved] = 1 - flipped.iloc[resolved]
    feats_flipped = build_feature_matrix(fills, ticks, buckets, labels=flipped)

    changed = ~np.isclose(feats["cp_toxic_rate_hist"], feats_flipped["cp_toxic_rate_hist"])
    # Later fills legitimately change (their history changed); the *first*
    # fill of each counterparty must not.
    first_positions = feats.reset_index().groupby("counterparty_id")["index"].first()
    assert not changed[first_positions].any(), (
        "a fill's own label leaked into its own counterparty-history feature")


def test_vpin_join_never_uses_an_unclosed_bucket(pipeline):
    ticks, fills, buckets, labels = pipeline
    feats = build_feature_matrix(fills, ticks, buckets, labels=labels)
    ends = pd.DatetimeIndex(buckets["timestamp_end"])

    joined = feats[feats["vpin"].notna()]
    for ts, v in zip(joined["timestamp"], joined["vpin"]):
        earlier = buckets.loc[ends <= ts, "vpin"].dropna()
        assert len(earlier) and np.isclose(earlier.iloc[-1], v), (
            f"fill at {ts} did not take the most recent *closed* bucket's VPIN")


def test_markouts_use_the_prevailing_not_the_nearest_quote():
    """Snapping to the nearest quote can pick one *after* the horizon.

    Constructed so the difference is unambiguous: the quote after t+tau is far
    closer in time than the one before it.
    """
    ts = pd.to_datetime(["2026-04-01T00:00:00.0", "2026-04-01T00:00:01.0",
                         "2026-04-01T00:00:29.9"], format="ISO8601", utc=True)
    ticks = pd.DataFrame({
        "timestamp": ts, "bid": [1.0999, 1.0999, 1.1999], "ask": [1.1001, 1.1001, 1.2001],
        "mid": [1.1000, 1.1000, 1.2000], "spread": 2e-4, "volume": 1000.0,
    })
    fills = pd.DataFrame({
        "timestamp": [ts[0]], "price": [1.1001], "size": [1e6], "side": ["buy"]})

    out = compute_markouts(fills, ticks, [5.0])
    # At t+5s the prevailing quote is the one at t+1s (mid 1.1000), not the
    # much nearer one at t+29.9s.
    assert np.isclose(out["markout_5s"].iloc[0], 1.1001 - 1.1000), out["markout_5s"].iloc[0]


def test_purged_kfold_removes_overlapping_labels(pipeline):
    ticks, fills, buckets, labels = pipeline
    feats = build_feature_matrix(fills, ticks, buckets, labels=labels)
    X, y = select_matrix(feats, labels)
    kept = feats.index[feats.index.isin(X.index)]
    event_times = fills["timestamp"].iloc[: len(X)].reset_index(drop=True)

    splits = list(purged_kfold_splits(event_times, horizon_seconds=60.0, n_splits=5))
    assert_no_leakage(splits, event_times, 60.0)

    # And confirm the purge actually removed something.
    total = sum(len(tr) + len(te) for tr, te in splits)
    assert total < len(event_times) * len(splits), "nothing was purged"


def test_unpurged_kfold_would_leak(pipeline):
    """Sanity check on the leakage detector itself."""
    ticks, fills, buckets, labels = pipeline
    event_times = fills["timestamp"].reset_index(drop=True)
    n = len(event_times)
    naive = [(np.arange(0, n // 2), np.arange(n // 2, n // 2 + 100))]
    with pytest.raises(AssertionError, match="overlap"):
        assert_no_leakage(naive, event_times, horizon_seconds=60.0)


def test_walk_forward_never_trains_on_the_future(pipeline):
    ticks, fills, buckets, labels = pipeline
    event_times = fills["timestamp"].reset_index(drop=True)
    for train_idx, test_idx in walk_forward_splits(event_times, 60.0, n_splits=4):
        assert train_idx.max() < test_idx.min(), "training data sits after the test block"
