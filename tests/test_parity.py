"""
Agreement between the C++ core and the NumPy reference.

The two implementations share no code: the C++ walks monotone cursors, skips
blocks and streams accumulators, while the reference uses searchsorted and
cumsum over whole arrays. Where they agree, the maths is almost certainly
right; where they diverge, one of them has a bug worth finding. These tests
are the reason to trust the fast path.
"""
import numpy as np
import pytest

import native
import _reference as ref

pytestmark = pytest.mark.skipif(
    not native.HAVE_NATIVE,
    reason="fxtox_native is not built; nothing to compare the reference against",
)

HORIZONS = [1.0, 5.0, 30.0, 300.0]


def test_feature_names_match():
    import fxtox_native
    assert list(fxtox_native.feature_names()) == ref.FEATURE_NAMES
    assert fxtox_native.FEATURE_COUNT == ref.FEATURE_COUNT


def test_norm_cdf_matches_scipy():
    from scipy.stats import norm
    z = np.linspace(-8, 8, 4001)
    import fxtox_native
    np.testing.assert_allclose(fxtox_native.norm_cdf(z), norm.cdf(z), rtol=0, atol=1e-15)


def test_student_t_cdf_matches_scipy():
    from scipy.stats import t as student_t
    z = np.linspace(-6, 6, 601)
    import fxtox_native
    for df in (0.25, 1.0, 3.0, 30.0):
        np.testing.assert_allclose(
            fxtox_native.student_t_cdf(z, df), student_t.cdf(z, df), rtol=1e-10, atol=1e-12
        )


@pytest.mark.parametrize("dist", ["normal", "student_t"])
def test_vpin_matches_reference(ticks, dist):
    V = ticks["volume"].sum() / 400.0
    kw = dict(bucket_volume=V, window=25, dist=dist, df=0.25, sigma_warmup=10)
    got = native.compute_vpin(ticks["ts"], ticks["bid"], ticks["ask"],
                              ticks["volume"], use_native=True, **kw)
    want = native.compute_vpin(ticks["ts"], ticks["bid"], ticks["ask"],
                               ticks["volume"], use_native=False, **kw)

    assert len(got["vpin"]) == len(want["vpin"]), "bucket counts differ"
    np.testing.assert_array_equal(got["ts_end"], want["ts_end"])
    for col in ("v_buy", "v_sell", "imbalance", "price_end", "vpin"):
        np.testing.assert_allclose(got[col], want[col], rtol=1e-9, atol=1e-12,
                                   equal_nan=True, err_msg=f"column {col}")
    # The C++ percentile is a quantised rank, so it can disagree with an exact
    # rank when two readings land in the same bin. Assert the bound that
    # actually matters: never more than one rank position apart.
    d = np.abs(got["percentile"] - want["percentile"])
    finite = ~np.isnan(d)
    ranks = np.arange(1, finite.sum() + 1)
    assert np.all(d[finite] * ranks <= 1.0 + 1e-9), (
        f"percentile ranks differ by more than one position "
        f"(max {np.max(d[finite] * ranks):.3f})")


def test_vpin_volume_is_conserved(ticks):
    V = ticks["volume"].sum() / 300.0
    out = native.compute_vpin(ticks["ts"], ticks["bid"], ticks["ask"],
                              ticks["volume"], bucket_volume=V, window=20)
    np.testing.assert_allclose(out["v_buy"] + out["v_sell"], V, rtol=1e-12)
    assert np.all(out["imbalance"] >= 0) and np.all(out["imbalance"] <= 1 + 1e-12)
    finite = out["vpin"][~np.isnan(out["vpin"])]
    assert np.all((finite >= 0) & (finite <= 1 + 1e-12))


def test_markouts_match_reference(ticks, fills):
    got = native.compute_markouts(fills["ts"], fills["price"], fills["side"],
                                  ticks["ts"], ticks["mid"], HORIZONS, use_native=True)
    want = native.compute_markouts(fills["ts"], fills["price"], fills["side"],
                                   ticks["ts"], ticks["mid"], HORIZONS, use_native=False)
    np.testing.assert_allclose(got, want, rtol=0, atol=0, equal_nan=True)


def test_mid_at_fill_matches_reference(ticks, fills):
    got = native.mid_at_fill(fills["ts"], ticks["ts"], ticks["mid"], use_native=True)
    want = native.mid_at_fill(fills["ts"], ticks["ts"], ticks["mid"], use_native=False)
    np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize("rule,kw", [
    ("crossback", dict(horizon_sec=30.0, min_adverse=2e-4)),
    ("crossback", dict(horizon_sec=120.0, min_adverse=5e-4)),
    ("triple_barrier", dict(horizon_sec=60.0, theta=1e-4)),
    ("triple_barrier", dict(horizon_sec=300.0, theta=3e-4)),
])
def test_labels_match_reference(ticks, fills, rule, kw):
    got = native.label_fills(fills["ts"], fills["price"], fills["side"],
                             ticks["ts"], ticks["mid"], rule=rule,
                             use_native=True, **kw)
    want = native.label_fills(fills["ts"], fills["price"], fills["side"],
                              ticks["ts"], ticks["mid"], rule=rule,
                              use_native=False, **kw)
    np.testing.assert_array_equal(got, want)
    # A degenerate labelling would make everything downstream meaningless.
    assert set(np.unique(got)).issubset({-1, 0, 1})
    resolved = got[got >= 0]
    assert len(resolved) > 100
    # A rule that labels (almost) everything one way carries no information and
    # would make every downstream metric meaningless, so pin it here.
    assert 0.02 < resolved.mean() < 0.98, (
        f"labels are nearly constant: toxic rate {resolved.mean():.4f}")


def test_markout_threshold_labels_match(ticks, fills):
    mk = native.compute_markouts(fills["ts"], fills["price"], fills["side"],
                                 ticks["ts"], ticks["mid"], [30.0])[:, 0]
    kw = dict(rule="markout_threshold", horizon_sec=30.0, theta=5e-5, markouts=mk)
    got = native.label_fills(fills["ts"], fills["price"], fills["side"],
                             ticks["ts"], ticks["mid"], use_native=True, **kw)
    want = native.label_fills(fills["ts"], fills["price"], fills["side"],
                              ticks["ts"], ticks["mid"], use_native=False, **kw)
    np.testing.assert_array_equal(got, want)


def test_features_match_reference(ticks, fills):
    kw = dict(short_window_sec=60.0, long_window_sec=900.0)
    got = native.build_features(ticks["ts"], ticks["bid"], ticks["ask"],
                                fills["ts"], fills["size"], ticks["bid_size"],
                                ticks["ask_size"], ticks["volume"],
                                use_native=True, **kw)
    want = native.build_features(ticks["ts"], ticks["bid"], ticks["ask"],
                                 fills["ts"], fills["size"], ticks["bid_size"],
                                 ticks["ask_size"], ticks["volume"],
                                 use_native=False, **kw)
    assert got.shape == want.shape == (len(fills["ts"]), ref.FEATURE_COUNT)
    for j, name in enumerate(ref.FEATURE_NAMES):
        # Running accumulators vs. cumsum: same maths, different summation
        # order, so compare to float tolerance rather than exactly. micro_dev
        # subtracts two prices around 1.1 and divides by a spread around 1e-4,
        # which discards roughly nine digits, so it gets its own bound.
        rtol = 1e-6 if name == "micro_dev" else 1e-9
        np.testing.assert_allclose(got[:, j], want[:, j], rtol=rtol, atol=1e-15,
                                   equal_nan=True, err_msg=f"feature {name}")


def test_counterparty_history_matches_reference(ticks, fills):
    labels = native.label_fills(fills["ts"], fills["price"], fills["side"],
                                ticks["ts"], ticks["mid"], rule="crossback",
                                horizon_sec=30.0)
    got_rate, got_count = native.counterparty_history(
        fills["counterparty"], labels, use_native=True)
    want_rate, want_count = native.counterparty_history(
        fills["counterparty"], labels, use_native=False)
    np.testing.assert_allclose(got_rate, want_rate, rtol=1e-12)
    np.testing.assert_array_equal(got_count, want_count)


def test_live_features_match_batch(ticks, fills):
    """The streaming path must reproduce the batch path exactly."""
    import fxtox_native
    batch = native.build_features(ticks["ts"], ticks["bid"], ticks["ask"],
                                  fills["ts"], fills["size"], ticks["bid_size"],
                                  ticks["ask_size"], ticks["volume"],
                                  short_window_sec=30.0, long_window_sec=300.0)
    live = fxtox_native.LiveFeatures(short_window_sec=30.0, long_window_sec=300.0)
    fill_ts = fills["ts"]
    pos = 0
    checked = 0
    for i in range(len(ticks["ts"])):
        live.on_tick(int(ticks["ts"][i]), float(ticks["bid"][i]), float(ticks["ask"][i]),
                     float(ticks["bid_size"][i]), float(ticks["ask_size"][i]),
                     float(ticks["volume"][i]))
        while pos < len(fill_ts) and fill_ts[pos] == ticks["ts"][i]:
            row = live.snapshot(float(fills["size"][pos]), int(fill_ts[pos]))
            np.testing.assert_array_equal(row, batch[pos], err_msg=f"fill {pos}")
            pos += 1
            checked += 1
    assert checked == len(fill_ts)


def test_crossback_direction_is_not_trivially_satisfied(ticks, fills):
    """A client buy fills at the ask, so the mid starts *below* the execution
    price. Reading the rule in the wrong direction makes it fire on essentially
    every fill (or none), which is the failure mode this pins down."""
    labels = native.label_fills(fills["ts"], fills["price"], fills["side"],
                                ticks["ts"], ticks["mid"], rule="crossback",
                                horizon_sec=60.0, min_adverse=2e-4)
    resolved = labels[labels >= 0]
    assert 0.05 < resolved.mean() < 0.95

    # Raising the bar must make fewer fills toxic, monotonically.
    rates = []
    for bar in (0.0, 1e-4, 3e-4, 1e-3, 5e-3):
        lab = native.label_fills(fills["ts"], fills["price"], fills["side"],
                                 ticks["ts"], ticks["mid"], rule="crossback",
                                 horizon_sec=60.0, min_adverse=bar)
        rates.append(lab[lab >= 0].mean())
    assert rates == sorted(rates, reverse=True), f"not monotone in min_adverse: {rates}"
    assert rates[0] > 0.9 and rates[-1] < 0.1
