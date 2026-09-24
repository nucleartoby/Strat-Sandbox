from __future__ import annotations

from typing import Iterator, Sequence

import numpy as np
import pandas as pd


def _as_ns(ts) -> np.ndarray:
    import native
    return native._i64(ts)


def purged_kfold_splits(event_times: pd.Series, horizon_seconds: float,
                        n_splits: int = 5, embargo_frac: float = 0.01
                        ) -> Iterator[tuple]:
    ts = _as_ns(event_times)
    n = len(ts)
    if n < n_splits * 2:
        raise ValueError(f"need at least {n_splits * 2} events for {n_splits} splits")
    if not np.all(np.diff(ts) >= 0):
        raise ValueError("event_times must be sorted ascending")

    horizon_ns = np.int64(round(horizon_seconds * 1e9))
    # Each events label spans [t, t + horizon]
    label_end = ts + horizon_ns

    embargo_ns = np.int64(round(embargo_frac * max(ts[-1] - ts[0], 1)))
    folds = np.array_split(np.arange(n), n_splits)

    for test_idx in folds:
        if len(test_idx) == 0:
            continue
        test_start = ts[test_idx[0]]
        test_end = label_end[test_idx[-1]]

        overlaps = (label_end >= test_start) & (ts <= test_end)
        embargoed = (ts > test_end) & (ts <= test_end + embargo_ns)

        train_mask = ~overlaps & ~embargoed
        train_mask[test_idx] = False
        train_idx = np.flatnonzero(train_mask)
        if len(train_idx) == 0:
            continue
        yield train_idx, test_idx


def walk_forward_splits(event_times: pd.Series, horizon_seconds: float,
                        n_splits: int = 5, min_train_frac: float = 0.4,
                        embargo_frac: float = 0.01, expanding: bool = True
                        ) -> Iterator[tuple]:
    ts = _as_ns(event_times)
    n = len(ts)
    horizon_ns = np.int64(round(horizon_seconds * 1e9))
    embargo_ns = np.int64(round(embargo_frac * max(ts[-1] - ts[0], 1)))

    min_train = int(n * min_train_frac)
    if min_train < 1 or n - min_train < n_splits:
        raise ValueError("not enough events for the requested walk-forward splits")
    fold = (n - min_train) // n_splits

    for k in range(n_splits):
        train_end = min_train + k * fold
        test_lo = train_end
        test_hi = min(train_end + fold, n)
        if test_hi <= test_lo:
            break

        test_idx = np.arange(test_lo, test_hi)
        cutoff = ts[test_lo] - embargo_ns
        train_start = 0 if expanding else max(0, train_end - fold * 2)
        candidate = np.arange(train_start, train_end)
        train_idx = candidate[ts[candidate] + horizon_ns <= cutoff]
        if len(train_idx) == 0:
            continue
        yield train_idx, test_idx


def split_report(splits: Sequence[tuple], event_times: pd.Series,
                 labels: pd.Series | None = None) -> pd.DataFrame:
    ts = pd.to_datetime(pd.Series(_as_ns(event_times)), utc=True)
    rows = []
    for i, (tr, te) in enumerate(splits):
        row = {
            "fold": i, "n_train": len(tr), "n_test": len(te),
            "train_start": ts.iloc[tr[0]] if len(tr) else pd.NaT,
            "train_end": ts.iloc[tr[-1]] if len(tr) else pd.NaT,
            "test_start": ts.iloc[te[0]] if len(te) else pd.NaT,
            "test_end": ts.iloc[te[-1]] if len(te) else pd.NaT,}
        if labels is not None:
            y = np.asarray(labels)
            row["train_toxic_rate"] = float(y[tr].mean()) if len(tr) else np.nan
            row["test_toxic_rate"] = float(y[te].mean()) if len(te) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def assert_no_leakage(splits: Sequence[tuple], event_times: pd.Series,
                      horizon_seconds: float) -> None:
    ts = _as_ns(event_times)
    horizon_ns = np.int64(round(horizon_seconds * 1e9))
    for i, (tr, te) in enumerate(splits):
        if len(tr) == 0 or len(te) == 0:
            continue
        test_start, test_end = ts[te[0]], ts[te[-1]] + horizon_ns
        train_end = ts[tr] + horizon_ns
        bad = np.flatnonzero((train_end >= test_start) & (ts[tr] <= test_end))
        if len(bad):
            raise AssertionError(
                f"fold {i}: {len(bad)} training events overlap the test window "
                f"(first at index {tr[bad[0]]})")
