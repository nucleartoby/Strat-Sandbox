from __future__ import annotations

import numpy as np
from scipy.stats import norm, t as student_t


FEATURE_NAMES = [
    "spread", "obi", "micro_dev", "rv_short", "rv_long", "vol_short",
    "vol_long", "size_rel_vol", "tick_rate", "spread_mean", "spread_rel",
    "dt_last_tick", "sec_of_day",]

FEATURE_COUNT = len(FEATURE_NAMES)

NS_PER_SEC = 1_000_000_000
NS_PER_DAY = 86_400 * NS_PER_SEC

LABEL_TOXIC = 1
LABEL_BENIGN = 0
LABEL_UNKNOWN = -1


def _buy_fraction(dp: float, sigma: float, dist: str, df: float) -> float:
    if not sigma > 0.0:
        return 0.5  # no dispersion estimate yet split evenly
    z = dp / sigma
    return float(norm.cdf(z)) if dist == "normal" else float(student_t.cdf(z, df))


def reference_vpin(ts, bid, ask, volume, bucket_volume, window=50, dist="normal",
                   df=0.25, sigma_mode="expanding", sigma_fixed=0.0,
                   sigma_warmup=20, pctile_warmup=30):
    ts = np.asarray(ts, dtype=np.int64)
    mid = (np.asarray(bid, float) + np.asarray(ask, float)) / 2.0
    volume = np.asarray(volume, float)
    if bucket_volume <= 0:
        raise ValueError("bucket_volume must be > 0")

    cum = np.cumsum(volume)
    total = cum[-1] if len(cum) else 0.0
    n_buckets = int(total // bucket_volume)
    if n_buckets == 0:
        return {k: np.empty(0) for k in
                ("ts_end", "vpin", "percentile", "imbalance", "v_buy", "v_sell", "price_end")}

    targets = bucket_volume * np.arange(1, n_buckets + 1)
    idx = np.searchsorted(cum, targets, side="left")
    idx = np.clip(idx, 0, len(mid) - 1)

    price_end = mid[idx]
    ts_end = ts[idx]

    prev_prices = np.empty(n_buckets)
    prev_prices[0] = mid[0]
    prev_prices[1:] = price_end[:-1]
    dp = price_end - prev_prices

    v_buy = np.empty(n_buckets)
    warmup = np.zeros(n_buckets, dtype=bool)

    if sigma_mode == "fixed":
        sig = np.full(n_buckets, float(sigma_fixed))
    else:
        sig = np.zeros(n_buckets)
        n_seen, mean, m2 = 0, 0.0, 0.0
        for k in range(n_buckets):
            sig[k] = np.sqrt(m2 / n_seen) if n_seen >= 2 else 0.0
            warmup[k] = n_seen < sigma_warmup
            n_seen += 1
            delta = dp[k] - mean
            mean += delta / n_seen
            m2 += delta * (dp[k] - mean)

    for k in range(n_buckets):
        v_buy[k] = bucket_volume * _buy_fraction(dp[k], sig[k], dist, df)

    v_sell = bucket_volume - v_buy
    imbalance = np.abs(v_buy - v_sell) / bucket_volume

    vpin = np.full(n_buckets, np.nan)
    if n_buckets >= window:
        csum = np.concatenate(([0.0], np.cumsum(imbalance)))
        vpin[window - 1:] = (csum[window:] - csum[:-window]) / window

    percentile = np.full(n_buckets, np.nan)
    finite = np.flatnonzero(~np.isnan(vpin))
    if len(finite):
        vals = vpin[finite]
        order = np.argsort(vals, kind="mergesort")
        ranks = np.empty(len(vals))
        ranks[order] = np.arange(1, len(vals) + 1)
        expanding = np.array([np.sum(vals[: i + 1] <= vals[i]) / (i + 1)
                              for i in range(len(vals))])
        expanding[: max(pctile_warmup - 1, 0)] = np.nan
        percentile[finite] = expanding

    return {
        "ts_end": ts_end, "vpin": vpin, "percentile": percentile,
        "imbalance": imbalance, "v_buy": v_buy, "v_sell": v_sell,
        "price_end": price_end, "warmup": warmup,}


def reference_markouts(fill_ts, fill_price, fill_side, mid_ts, mid, horizons_sec):
    fill_ts = np.asarray(fill_ts, dtype=np.int64)
    fill_price = np.asarray(fill_price, float)
    sign = np.asarray(fill_side, float)
    mid_ts = np.asarray(mid_ts, dtype=np.int64)
    mid = np.asarray(mid, float)

    out = np.full((len(fill_ts), len(horizons_sec)), np.nan)
    if len(mid_ts) == 0:
        return out
    last_ts = mid_ts[-1]

    for h, tau in enumerate(horizons_sec):
        target = fill_ts + np.int64(round(tau * NS_PER_SEC))
        j = np.searchsorted(mid_ts, target, side="right") - 1
        ok = (j >= 0) & (target <= last_ts)
        vals = np.full(len(fill_ts), np.nan)
        vals[ok] = sign[ok] * (fill_price[ok] - mid[j[ok]])
        out[:, h] = vals
    return out


def reference_mid_at_fill(fill_ts, mid_ts, mid):
    fill_ts = np.asarray(fill_ts, dtype=np.int64)
    mid_ts = np.asarray(mid_ts, dtype=np.int64)
    mid = np.asarray(mid, float)
    j = np.searchsorted(mid_ts, fill_ts, side="right") - 1
    out = np.full(len(fill_ts), np.nan)
    ok = j >= 0
    out[ok] = mid[j[ok]]
    return out


def reference_labels(fill_ts, fill_price, fill_side, mid_ts, mid, rule,
                     horizon_sec=30.0, theta=0.0, min_adverse=0.0, markouts=None,
                     center="mid_at_fill"):
    fill_ts = np.asarray(fill_ts, dtype=np.int64)
    fill_price = np.asarray(fill_price, float)
    side = np.asarray(fill_side, np.int8)
    mid_ts = np.asarray(mid_ts, dtype=np.int64)
    mid = np.asarray(mid, float)
    n = len(fill_ts)

    if rule == "markout_threshold":
        if markouts is None:
            raise ValueError("markout_threshold labeling needs the markouts argument")
        m = np.asarray(markouts, float)
        out = np.where(m < -theta, LABEL_TOXIC, LABEL_BENIGN).astype(np.int8)
        out[np.isnan(m)] = LABEL_UNKNOWN
        return out

    out = np.full(n, LABEL_UNKNOWN, dtype=np.int8)
    if len(mid_ts) == 0:
        return out
    horizon_ns = np.int64(round(horizon_sec * NS_PER_SEC))
    last_ts = mid_ts[-1]

    at = np.searchsorted(mid_ts, fill_ts, side="right") - 1
    end_ts = fill_ts + horizon_ns
    hi = np.searchsorted(mid_ts, end_ts, side="right") - 1

    def first_true(mask):
        """Index of the first True, or -1."""
        nz = np.flatnonzero(mask)
        return int(nz[0]) if len(nz) else -1

    for i in range(n):
        if end_ts[i] > last_ts or at[i] < 0:
            continue
        lo = at[i] + 1  # strictly after the fill
        if lo > hi[i]:
            continue
        path = mid[lo: hi[i] + 1]
        bought = side[i] >= 0
        px = fill_price[i]

        if rule == "crossback":
            barrier = px + min_adverse if bought else px - min_adverse
            hit = first_true(path >= barrier) if bought else first_true(path <= barrier)
            out[i] = LABEL_TOXIC if hit >= 0 else LABEL_BENIGN

        elif rule == "triple_barrier":
            c = mid[at[i]] if center == "mid_at_fill" else px
            t_up = first_true(path >= c + theta)
            t_down = first_true(path <= c - theta)
            t_fav, t_adv = (t_up, t_down) if bought else (t_down, t_up)
            if t_fav < 0:
                out[i] = LABEL_BENIGN
            elif t_adv < 0:
                out[i] = LABEL_TOXIC
            else:
                out[i] = LABEL_TOXIC if t_fav < t_adv else LABEL_BENIGN
        else:
            raise ValueError(f"unknown labeling rule: {rule}")
    return out


def reference_features(tick_ts, bid, ask, fill_ts, fill_size, bid_size=None,
                       ask_size=None, volume=None, short_window_sec=60.0,
                       long_window_sec=900.0):
    tick_ts = np.asarray(tick_ts, dtype=np.int64)
    bid = np.asarray(bid, float)
    ask = np.asarray(ask, float)
    n_t = len(tick_ts)
    fill_ts = np.asarray(fill_ts, dtype=np.int64)
    fill_size = np.asarray(fill_size, float)
    n_f = len(fill_ts)

    zeros = np.zeros(n_t)
    bid_size = zeros if bid_size is None else np.asarray(bid_size, float)
    ask_size = zeros if ask_size is None else np.asarray(ask_size, float)
    volume = zeros if volume is None else np.asarray(volume, float)

    out = np.full((n_f, FEATURE_COUNT), np.nan)
    if n_t == 0 or n_f == 0:
        return out

    mid = (bid + ask) / 2.0
    spread = ask - bid

    r2 = np.zeros(n_t)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = mid[1:] / mid[:-1]
    good = (mid[1:] > 0) & (mid[:-1] > 0)
    lr = np.zeros(n_t - 1)
    lr[good] = np.log(ratio[good])
    r2[1:] = lr ** 2

    c_r2 = np.concatenate(([0.0], np.cumsum(r2)))
    c_vol = np.concatenate(([0.0], np.cumsum(volume)))
    c_spr = np.concatenate(([0.0], np.cumsum(spread)))

    right = np.searchsorted(tick_ts, fill_ts, side="right") - 1
    valid = right >= 0

    short_ns = np.int64(round(short_window_sec * NS_PER_SEC))
    long_ns = np.int64(round(long_window_sec * NS_PER_SEC))
    left_s = np.searchsorted(tick_ts, fill_ts - short_ns, side="left")
    left_l = np.searchsorted(tick_ts, fill_ts - long_ns, side="left")

    v = np.flatnonzero(valid)
    if len(v) == 0:
        return out
    r = right[v]
    ls = np.minimum(left_s[v], r + 1)
    ll = np.minimum(left_l[v], r + 1)

    sum_r2_s = c_r2[r + 1] - c_r2[ls]
    sum_r2_l = c_r2[r + 1] - c_r2[ll]
    sum_vol_s = c_vol[r + 1] - c_vol[ls]
    sum_vol_l = c_vol[r + 1] - c_vol[ll]
    sum_spr_s = c_spr[r + 1] - c_spr[ls]
    count_s = (r + 1 - ls).astype(float)

    cur_spread = spread[r]
    depth = bid_size[r] + ask_size[r]

    out[v, FEATURE_NAMES.index("spread")] = cur_spread
    with np.errstate(divide="ignore", invalid="ignore"):
        obi = np.where(depth > 0, (bid_size[r] - ask_size[r]) / depth, 0.0)
        micro = np.where(depth > 0,
                         (bid[r] * ask_size[r] + ask[r] * bid_size[r]) / np.where(depth > 0, depth, 1.0),
                         mid[r])
        micro_dev = np.where((depth > 0) & (cur_spread > 0),
                             (micro - mid[r]) / np.where(cur_spread > 0, cur_spread, 1.0), 0.0)
    out[v, FEATURE_NAMES.index("obi")] = obi
    out[v, FEATURE_NAMES.index("micro_dev")] = micro_dev
    out[v, FEATURE_NAMES.index("rv_short")] = np.sqrt(np.maximum(sum_r2_s, 0.0))
    out[v, FEATURE_NAMES.index("rv_long")] = np.sqrt(np.maximum(sum_r2_l, 0.0))
    out[v, FEATURE_NAMES.index("vol_short")] = sum_vol_s
    out[v, FEATURE_NAMES.index("vol_long")] = sum_vol_l

    with np.errstate(divide="ignore", invalid="ignore"):
        out[v, FEATURE_NAMES.index("size_rel_vol")] = np.where(
            sum_vol_s > 0, fill_size[v] / np.where(sum_vol_s > 0, sum_vol_s, 1.0), np.nan)
    span = short_window_sec if short_window_sec > 0 else 1.0
    out[v, FEATURE_NAMES.index("tick_rate")] = count_s / span

    spread_mean = np.where(count_s > 0, sum_spr_s / np.where(count_s > 0, count_s, 1.0), np.nan)
    out[v, FEATURE_NAMES.index("spread_mean")] = spread_mean
    with np.errstate(divide="ignore", invalid="ignore"):
        out[v, FEATURE_NAMES.index("spread_rel")] = np.where(
            spread_mean > 0, cur_spread / np.where(spread_mean > 0, spread_mean, 1.0), np.nan)

    dt = np.full(len(v), np.nan)
    has_prev = r >= 1
    dt[has_prev] = (tick_ts[r[has_prev]] - tick_ts[r[has_prev] - 1]) / NS_PER_SEC
    out[v, FEATURE_NAMES.index("dt_last_tick")] = dt
    out[v, FEATURE_NAMES.index("sec_of_day")] = (tick_ts[r] % NS_PER_DAY) / NS_PER_SEC
    return out


def reference_counterparty_history(counterparty, labels, prior_rate=0.5, prior_weight=5.0):
    cp = np.asarray(counterparty, np.int32)
    y = np.asarray(labels, np.int8)
    rate = np.empty(len(cp))
    count = np.empty(len(cp))
    seen: dict[int, list[float]] = {}
    for i, c in enumerate(cp):
        toxic, total = seen.get(int(c), (0.0, 0.0))
        rate[i] = (toxic + prior_rate * prior_weight) / (total + prior_weight)
        count[i] = total
        if y[i] >= 0:  # update only after emitting and only for resolved fills
            seen[int(c)] = (toxic + float(y[i]), total + 1.0)
        else:
            seen[int(c)] = (toxic, total)
    return rate, count
