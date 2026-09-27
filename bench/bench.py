import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import native
from data_cleaning import clean_ticks
from data_ingestion import generate_synthetic_ticks, generate_synthetic_trades


def timed(fn, repeat=3):
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def main():
    n_ticks = int(sys.argv[1]) if len(sys.argv) > 1 else 500_000
    n_fills = int(sys.argv[2]) if len(sys.argv) > 2 else 20_000

    ticks = clean_ticks(generate_synthetic_ticks(n=n_ticks))
    fills = generate_synthetic_trades(ticks, n_trades=n_fills)
    print(f"{len(ticks):,} ticks, {len(fills):,} fills, backend: {native.backend()}\n")

    ts = native._i64(ticks["timestamp"])
    bid, ask = ticks["bid"].to_numpy(), ticks["ask"].to_numpy()
    mid, vol = ticks["mid"].to_numpy(), ticks["volume"].to_numpy()
    bsz, asz = ticks["bid_size"].to_numpy(), ticks["ask_size"].to_numpy()
    f_ts = native._i64(fills["timestamp"])
    f_px, f_sz = fills["price"].to_numpy(), fills["size"].to_numpy()
    f_side = fills["side"].to_numpy()
    bv = vol.sum() / 2000
    horizons = [1.0, 5.0, 30.0, 300.0]

    stages = {
        "VPIN (bucket + BVC + roll)":
            lambda nat: native.compute_vpin(ts, bid, ask, vol, bucket_volume=bv,
                                            window=50, use_native=nat),
        "markouts (4 horizons)":
            lambda nat: native.compute_markouts(f_ts, f_px, f_side, ts, mid,
                                                horizons, use_native=nat),
        "labels (triple barrier)":
            lambda nat: native.label_fills(f_ts, f_px, f_side, ts, mid,
                                           rule="triple_barrier", horizon_sec=60.0,
                                           theta=2e-3, use_native=nat),
        "features (13 columns)":
            lambda nat: native.build_features(ts, bid, ask, f_ts, f_sz, bsz, asz,
                                              vol, use_native=nat),}

    print(f"{'stage':<30} {'C++':>10} {'NumPy ref':>12} {'speedup':>9}")
    print("-" * 64)
    total_c, total_p = 0.0, 0.0
    for name, fn in stages.items():
        tc = timed(lambda: fn(True))
        tp = timed(lambda: fn(False), repeat=1)
        total_c += tc
        total_p += tp
        print(f"{name:<30} {tc * 1e3:>8.1f}ms {tp * 1e3:>10.1f}ms {tp / tc:>8.1f}x")
    print("-" * 64)
    print(f"{'total':<30} {total_c * 1e3:>8.1f}ms {total_p * 1e3:>10.1f}ms "
          f"{total_p / total_c:>8.1f}x")
    print(f"\nper-tick cost of the C++ path: {total_c / len(ticks) * 1e9:.0f} ns")


if __name__ == "__main__":
    main()
