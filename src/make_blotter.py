import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def lp_blotter(ticks: pd.DataFrame, refresh_ticks: int = 40,
               quote_multiple: float = 1.0, n_counterparties: int = 40,
               fast_fraction: float = 0.2, benign_per_pickoff: float = 3.0,
               pickoff_edge: float = 1.0, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    mid = ticks["mid"].to_numpy()
    spread = ticks["spread"].to_numpy()
    ts = ticks["timestamp"].to_numpy()
    n = len(mid)

    latency = rng.lognormal(mean=np.log(150), sigma=0.8, size=n_counterparties)
    n_fast = max(int(n_counterparties * fast_fraction), 1)
    latency[:n_fast] = rng.uniform(1.0, 8.0, n_fast)
    speed = 1.0 / latency

    starts = np.arange(0, n - refresh_ticks, refresh_ticks)
    rows = []
    for i in starts:
        quote_mid = mid[i]
        half = 0.5 * spread[i] * quote_multiple
        bid, ask = quote_mid - half, quote_mid + half

        edge = pickoff_edge * half
        window = mid[i + 1: i + 1 + refresh_ticks]
        up = np.flatnonzero(window >= ask + edge)
        down = np.flatnonzero(window <= bid - edge)
        if len(up) == 0 and len(down) == 0:
            continue

        first_up = up[0] if len(up) else np.iinfo(np.int64).max
        first_down = down[0] if len(down) else np.iinfo(np.int64).max
        lifted = first_up < first_down
        k = i + 1 + (first_up if lifted else first_down)

        cp = rng.choice(n_counterparties, p=speed / speed.sum())
        rows.append((
            ts[k],
            "buy" if lifted else "sell",
            ask if lifted else bid,
            float(rng.integers(1, 20) * 100_000),
            f"CP_{cp:03d}",))

    if not rows:
        raise ValueError(
            "no fills generated. The quoted spread is never crossed -- lower "
            "--quote-multiple or raise --refresh-ticks.")
    
    pickoffs = pd.DataFrame(rows, columns=["timestamp", "side", "price", "size",
                                           "counterparty_id"])
    pickoffs["is_informed"] = True

    n_benign = int(len(pickoffs) * benign_per_pickoff)
    if n_benign > 0:
        idx = np.sort(rng.choice(n - 1, size=min(n_benign, n - 1), replace=False))
        side = rng.choice(["buy", "sell"], size=len(idx))
        slow_weight = latency / latency.sum()
        cps = rng.choice(n_counterparties, size=len(idx), p=slow_weight)
        benign = pd.DataFrame({
            "timestamp": ts[idx],
            "side": side,
            "price": np.where(side == "buy", ticks["ask"].to_numpy()[idx],
                              ticks["bid"].to_numpy()[idx]),
            "size": rng.integers(1, 20, len(idx)) * 100_000.0,
            "counterparty_id": [f"CP_{c:03d}" for c in cps],
            "is_informed": False,})
        return pd.concat([pickoffs, benign], ignore_index=True)
    return pickoffs


def ma_blotter(ticks: pd.DataFrame, fast: int = 200, slow: int = 800,
               size: float = 1_000_000.0, seed: int = 0) -> pd.DataFrame:
    mid = ticks["mid"]
    fast_ma = mid.ewm(span=fast).mean()
    slow_ma = mid.ewm(span=slow).mean()
    signal = np.where(fast_ma > slow_ma, 1, -1)
    signal = pd.Series(signal, index=ticks.index).shift(1)  # no lookahead

    flips = np.flatnonzero(signal.diff().fillna(0).to_numpy() != 0)
    flips = flips[flips > slow]  # skip the EMA warmup
    if len(flips) == 0:
        raise ValueError(
            f"no crossovers with fast={fast} slow={slow} over {len(ticks):,} ticks. "
            "Use shorter spans, or more data.")

    rng = np.random.default_rng(seed)
    side = np.where(signal.to_numpy()[flips] > 0, "buy", "sell")
    price = np.where(side == "buy", ticks["ask"].to_numpy()[flips],
                     ticks["bid"].to_numpy()[flips])
    return pd.DataFrame({
        "timestamp": ticks["timestamp"].to_numpy()[flips],
        "side": side,
        "price": price,
        "size": size,
        "counterparty_id": "SELF",})


def random_blotter(ticks: pd.DataFrame, n_trades: int = 10_000,
                   n_counterparties: int = 40, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(ticks) - 1, size=n_trades, replace=False))
    side = rng.choice(["buy", "sell"], size=n_trades)
    price = np.where(side == "buy", ticks["ask"].to_numpy()[idx],
                     ticks["bid"].to_numpy()[idx])
    return pd.DataFrame({
        "timestamp": ticks["timestamp"].to_numpy()[idx],
        "side": side,
        "price": price,
        "size": rng.integers(1, 20, n_trades) * 100_000.0,
        "counterparty_id": [f"CP_{c:03d}" for c in
                            rng.integers(0, n_counterparties, n_trades)],})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_argument_group("tick source (one required)")
    src.add_argument("--symbol", help="fetch from Dukascopy, e.g. EURUSD")
    src.add_argument("--start", help="fetch start, YYYY-MM-DD")
    src.add_argument("--end", help="fetch end, YYYY-MM-DD (inclusive)")
    src.add_argument("--ticks", help="or read a tick CSV")
    ap.add_argument("--provider", default="auto",
                    choices=["auto", "dukascopy", "duka", "truefx", "histdata", "generic"])

    ap.add_argument("--mode", default="lp", choices=["lp", "ma", "random"])
    ap.add_argument("-o", "--out", default="my_fills.csv")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--refresh-ticks", type=int, default=40,
                    help="lp: how long a quote stands before refresh [40]")
    ap.add_argument("--quote-multiple", type=float, default=1.0,
                    help="lp: quoted spread as a multiple of the market's [1.0]")
    ap.add_argument("--counterparties", type=int, default=40)
    ap.add_argument("--fast-fraction", type=float, default=0.2,
                    help="lp: share of counterparties that are latency-fast [0.2]")
    ap.add_argument("--benign-per-pickoff", type=float, default=3.0,
                    help="lp: benign spread-crossing fills per pickoff. Lower "
                         "means a more toxic book [3.0]")
    ap.add_argument("--pickoff-edge", type=float, default=1.0,
                    help="lp: how far past the quote the mid must move, in "
                         "half-spreads, before a pickoff fires [1.0]")
    ap.add_argument("--fast", type=int, default=200, help="ma: fast EMA span")
    ap.add_argument("--slow", type=int, default=800, help="ma: slow EMA span")
    ap.add_argument("--n-trades", type=int, default=10_000, help="random: fill count")
    args = ap.parse_args(argv)

    from data_cleaning import clean_ticks
    from data_ingestion import load_tick_data

    if args.symbol:
        if not (args.start and args.end):
            ap.error("--symbol needs --start and --end")
        from data_fetch import fetch_ticks
        raw = fetch_ticks(args.symbol, args.start, args.end)
    elif args.ticks:
        if not os.path.exists(args.ticks):
            ap.error(f"--ticks: no such file: {args.ticks}")
        raw = load_tick_data(args.ticks, provider=args.provider)
    else:
        ap.error("need --symbol with --start/--end, or --ticks")

    ticks = clean_ticks(raw, verbose=True).reset_index(drop=True)

    if args.mode == "lp":
        fills = lp_blotter(ticks, refresh_ticks=args.refresh_ticks,
                           quote_multiple=args.quote_multiple,
                           n_counterparties=args.counterparties,
                           fast_fraction=args.fast_fraction,
                           benign_per_pickoff=args.benign_per_pickoff,
                           pickoff_edge=args.pickoff_edge, seed=args.seed)
    elif args.mode == "ma":
        fills = ma_blotter(ticks, fast=args.fast, slow=args.slow, seed=args.seed)
    else:
        fills = random_blotter(ticks, n_trades=args.n_trades,
                               n_counterparties=args.counterparties, seed=args.seed)

    fills = fills.sort_values("timestamp").reset_index(drop=True)
    fills.to_csv(args.out, index=False)

    span = pd.to_datetime(fills["timestamp"])
    print(f"\nwrote {len(fills):,} fills to {args.out}")
    print(f"  mode: {args.mode}")
    print(f"  span: {span.min()} .. {span.max()}")
    print(f"  buy/sell: {(fills['side'] == 'buy').sum():,} / "
          f"{(fills['side'] == 'sell').sum():,}")
    print(f"  counterparties: {fills['counterparty_id'].nunique()}")
    if len(fills) < 10_000:
        print(f"\n  note: {len(fills):,} fills is on the thin side. Walk-forward folds "
              f"get small\n  and the model struggles below roughly 10,000 -- fetch more "
              f"days if you can.")
    print(f"\nnext:\n  python src/run_pipeline.py --ticks <your ticks.csv> "
          f"--provider generic \\\n           --blotter {args.out} --label-horizon 300")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
