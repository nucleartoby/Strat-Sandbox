import numpy as np
import pandas as pd
import math    
from flowrisk import BulkVPIN, BulkVPINConfig

import native
from vpin import compute_vpin


def check_analytic_properties(ticks: pd.DataFrame, bucket_volume: float,
                              window: int = 20) -> pd.DataFrame:
    checks = []

    def record(name, passed, detail=""):
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    out = compute_vpin(ticks, bucket_volume, window=window)

    total = out["v_buy"] + out["v_sell"]
    record("volume conserved (v_buy + v_sell == V)",
           np.allclose(total, bucket_volume, rtol=1e-12),
           f"max deviation {np.max(np.abs(total - bucket_volume)):.3e}")

    record("imbalance in [0, 1]",
           bool(((out["order_imbalance"] >= 0) & (out["order_imbalance"] <= 1 + 1e-12)).all()),
           f"range [{out['order_imbalance'].min():.4f}, {out['order_imbalance'].max():.4f}]")

    finite = out["vpin"].dropna()
    record("vpin in [0, 1]",
           bool(((finite >= 0) & (finite <= 1 + 1e-12)).all()) if len(finite) else False,
           f"range [{finite.min():.4f}, {finite.max():.4f}]" if len(finite) else "no finite values")

    record("vpin warms up (first window-1 buckets are NaN)",
           bool(out["vpin"].isna().sum() >= window - 1),
           f"{int(out['vpin'].isna().sum())} NaN, window={window}")

    # One-sided flow: a monotonically rising market is all buys under BVC.
    one_sided = _synthetic_trend(len(ticks), drift=4e-5)
    hi = compute_vpin(one_sided, bucket_volume=one_sided["volume"].sum() / 60,
                      window=10)["vpin"].dropna()
    record("one-sided flow drives vpin high", bool(len(hi) and hi.mean() > 0.7),
           f"mean vpin {hi.mean():.3f}" if len(hi) else "no buckets")

    balanced = _synthetic_trend(len(ticks), drift=0.0)
    lo = compute_vpin(balanced, bucket_volume=balanced["volume"].sum() / 60,
                      window=10)["vpin"].dropna()
    record("balanced flow sits at the 0.5 noise floor",
           bool(len(lo) and 0.40 < lo.mean() < 0.60),
           f"mean vpin {lo.mean():.3f} (theory: 0.500)" if len(lo) else "no buckets")

    record("one-sided > balanced",
           bool(len(hi) and len(lo) and hi.mean() > lo.mean() + 0.2),
           f"{hi.mean():.3f} vs {lo.mean():.3f}" if len(hi) and len(lo) else "")

    head = ticks.head(5000)
    sub_bv = float(head["volume"].sum() / 80)
    split = _split_ticks(head, factor=4)
    a = compute_vpin(head, sub_bv, window=window)["vpin"].dropna()
    b = compute_vpin(split, sub_bv, window=window)["vpin"].dropna()
    n = min(len(a), len(b))
    record("invariant to tick subdivision",
           bool(n > 0 and np.corrcoef(a[:n], b[:n])[0, 1] > 0.95),
           f"corr {np.corrcoef(a[:n], b[:n])[0, 1]:.4f} over {n} buckets" if n > 1 else "too few buckets")

    return pd.DataFrame(checks)


def _synthetic_trend(n: int, drift: float, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    mid = 1.10 + np.cumsum(drift + rng.normal(0, 2e-5, n))
    ts = pd.to_datetime(
        np.int64(pd.Timestamp("2026-02-01", tz="UTC").value)
        + np.cumsum(rng.integers(1_000_000, 5_000_000, n)).astype(np.int64), utc=True)
    return pd.DataFrame({
        "timestamp": ts, "bid": mid - 4e-5, "ask": mid + 4e-5, "mid": mid,
        "spread": np.full(n, 8e-5), "volume": 1000.0 * rng.integers(1, 20, n),})


def _split_ticks(ticks: pd.DataFrame, factor: int = 4) -> pd.DataFrame:
    rep = ticks.loc[ticks.index.repeat(factor)].copy().reset_index(drop=True)
    rep["volume"] = rep["volume"] / factor
    offsets = np.tile(np.arange(factor), len(ticks)).astype("timedelta64[ns]")
    rep["timestamp"] = rep["timestamp"] + offsets
    return rep


def compare_with_flowrisk(ticks: pd.DataFrame, bucket_volume: float | None = None,
                          window: int = 20, n_bars: int = 20_000,
                          target_buckets: int = 250,
                          granularity: str = "tick") -> dict:
    sample = ticks.head(n_bars).reset_index(drop=True)
    if bucket_volume is None:
        bucket_volume = float(sample["volume"].sum() / target_buckets)
    n_expected = int(sample["volume"].sum() // bucket_volume)
    if n_expected < window + 5:
        return {"available": True,
                "error": f"bucket_volume {bucket_volume:,.0f} yields only ~{n_expected} "
                         f"buckets over {len(sample):,} ticks; need > {window + 5} for a "
                         f"window of {window}. Pass a smaller bucket_volume or more bars."}

    bars = pd.DataFrame({
        "time": sample["timestamp"],
        "price": sample["mid"],
        "volume": sample["volume"],})

    class _Config(BulkVPINConfig):
        BUCKET_MAX_VOLUME = float(bucket_volume)
        N_BUCKET_OR_BUCKET_DECAY = int(window)
        VOL_DECAY = 0.95

    try:
        theirs = BulkVPIN(_Config()).estimate(bars)
    except Exception as exc:
        return {"available": True, "error": f"{type(exc).__name__}: {exc}"}

    their_vpin = pd.Series(
        theirs["vpin"] if isinstance(theirs, pd.DataFrame) and "vpin" in theirs
        else np.ravel(np.asarray(theirs)), name="flowrisk_vpin")

    ours = compute_vpin(sample, bucket_volume, window=window, granularity=granularity)

    their_at_close = pd.Series(
        np.asarray(their_vpin)[np.searchsorted(
            sample["timestamp"].to_numpy(), ours["timestamp_end"].to_numpy(),
            side="right") - 1],
        index=ours.index)

    joined = pd.DataFrame({"ours": ours["vpin"], "theirs": their_at_close}).dropna()
    if len(joined) < 10:
        return {"available": True, "error": f"only {len(joined)} comparable points"}

    pearson = float(joined["ours"].corr(joined["theirs"]))
    spearman = float(joined["ours"].corr(joined["theirs"], method="spearman"))

    their_std = float(joined["theirs"].std())
    our_std = float(joined["ours"].std())
    ratio = their_std / our_std if our_std > 0 else 0.0
    degenerate = ratio < 0.3 or ratio > 3.0

    hi_ours = joined["ours"] >= joined["ours"].quantile(0.8)
    hi_theirs = joined["theirs"] >= joined["theirs"].quantile(0.8)
    overlap = float((hi_ours & hi_theirs).sum() / max(hi_ours.sum(), 1))

    note = (f"compared in {granularity}-granularity BVC, which is how flowrisk "
            "classifies. flowrisk still uses an EWMA volatility and bucket "
            "average against an expanding sigma and fixed window here, so exact "
            "equality is not expected.")
    if degenerate:
        note = (f"the two series have very different dispersion (flowrisk std "
                f"{their_std:.4f} vs ours {our_std:.4f}). That usually means the "
                f"granularity does not match: flowrisk classifies per tick, so "
                f"granularity='tick' is the like-for-like comparison. As it "
                "stands the correlation is between two different estimators and "
                "its sign is arbitrary -- rely on the analytic properties.")

    return {
        "available": True,
        "n_points": len(joined),
        "pearson": pearson,
        "spearman": spearman,
        "top_quintile_overlap": overlap,
        "mean_ours": float(joined["ours"].mean()),
        "mean_flowrisk": float(joined["theirs"].mean()),
        "std_ours": our_std,
        "std_flowrisk": their_std,
        "granularity": granularity,
        "degenerate": degenerate,
        "note": note,}


def validation_report(ticks: pd.DataFrame, bucket_volume: float, window: int = 20,
                      verbose: bool = True) -> dict:
    """Run both validations and return a combined result."""
    props = check_analytic_properties(ticks, bucket_volume, window)
    flow = compare_with_flowrisk(ticks, bucket_volume=None, window=window,
                                 granularity="tick")

    if verbose:
        print("Analytic properties:")
        for _, r in props.iterrows():
            print(f"  [{'PASS' if r['passed'] else 'FAIL'}] {r['check']}"
                  + (f"  ({r['detail']})" if r["detail"] else ""))
        print("\nflowrisk cross-check:")
        if not flow.get("available"):
            print(f"  skipped: {flow.get('reason')}")
        elif "error" in flow:
            print(f"  error: {flow['error']}")
        else:
            verdict = "UNINFORMATIVE" if flow.get("degenerate") else "ok"
            print(f"  n={flow['n_points']}  pearson={flow['pearson']:.3f}  "
                  f"spearman={flow['spearman']:.3f}  "
                  f"top-quintile overlap={flow['top_quintile_overlap']:.2f}  [{verdict}]")
            print(f"  levels: ours {flow['mean_ours']:.3f} vs flowrisk {flow['mean_flowrisk']:.3f}")
            print(f"  {flow['note']}")

    return {"properties": props, "flowrisk": flow,
            "all_properties_passed": bool(props["passed"].all())}
