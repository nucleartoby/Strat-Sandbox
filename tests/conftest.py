import os
import sys

import numpy as np
import pytest
import native

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)


@pytest.fixture(scope="session")
def ticks():
    rng = np.random.default_rng(20240917)
    n = 40_000
    drift = np.zeros(n)
    burst = rng.random(n) < 0.0005
    cur = 0.0
    for i in range(n):
        if burst[i]:
            cur = rng.choice([-1.0, 1.0]) * 3e-6
        elif rng.random() < 0.01:
            cur = 0.0
        drift[i] = cur
    mid = 1.1000 + np.cumsum(drift + rng.normal(0, 3e-5, n))
    spread = 8e-5 + 2e-5 * np.abs(rng.normal(size=n))
    ts = np.int64(1_700_000_000 * native.NS_PER_SEC) + np.cumsum(
        rng.integers(1_000_000, 20_000_000, n)).astype(np.int64)
    return {
        "ts": ts,
        "bid": mid - spread / 2,
        "ask": mid + spread / 2,
        "mid": mid,
        "bid_size": 1e6 * (0.5 + rng.random(n)),
        "ask_size": 1e6 * (0.5 + rng.random(n)),
        "volume": 1000.0 * rng.integers(1, 50, n).astype(float),}


@pytest.fixture(scope="session")
def fills(ticks):
    rng = np.random.default_rng(99)
    n = len(ticks["ts"])
    idx = np.sort(rng.choice(n - 1, size=2_000, replace=False))
    side = rng.choice([1, -1], size=len(idx)).astype(np.int8)
    price = np.where(side == 1, ticks["ask"][idx], ticks["bid"][idx])
    return {
        "ts": ticks["ts"][idx],
        "price": price,
        "size": 1000.0 * rng.integers(1, 20, len(idx)).astype(float),
        "side": side,
        "counterparty": rng.integers(0, 40, len(idx)).astype(np.int32),}
