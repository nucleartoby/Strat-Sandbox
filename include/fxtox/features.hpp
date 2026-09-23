// Every value here is computable from quotes at or before the fill timestamp.
// That is not stylistic: a feature that peeks even one tick past
// the fill produces a model that looks excellent in backtest and is worth
// nothing live, and the leak is invisible in the metrics.
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <unordered_map>
#include <vector>

#include "fxtox/markout.hpp"
#include "fxtox/types.hpp"

namespace fxtox {


enum FeatureCol : int {
    kFeatSpread = 0,      // prevailing spread at the fill
    kFeatObi,             // top-of-book imbalance, (bid_sz - ask_sz)/(sum)
    kFeatMicroDev,        // (microprice - mid) / spread; which way the book leans
    kFeatRvShort,         // realized vol of mid over the short lookback
    kFeatRvLong,          // realized vol over the long lookback
    kFeatVolShort,        // traded volume over the short lookback
    kFeatVolLong,         // traded volume over the long lookback
    kFeatSizeRelVol,      // fill size / short-window volume
    kFeatTickRate,        // ticks per second over the short lookback
    kFeatSpreadMean,      // mean spread over the short lookback
    kFeatSpreadRel,       // current spread / mean spread; is it widening?
    kFeatDtLastTick,      // seconds since the previous quote
    kFeatSecOfDay,        // seconds since UTC midnight (session proxy)
    kFeatCount
};

inline const char* feature_name(int c) {
    switch (c) {
        case kFeatSpread:     return "spread";
        case kFeatObi:        return "obi";
        case kFeatMicroDev:   return "micro_dev";
        case kFeatRvShort:    return "rv_short";
        case kFeatRvLong:     return "rv_long";
        case kFeatVolShort:   return "vol_short";
        case kFeatVolLong:    return "vol_long";
        case kFeatSizeRelVol: return "size_rel_vol";
        case kFeatTickRate:   return "tick_rate";
        case kFeatSpreadMean: return "spread_mean";
        case kFeatSpreadRel:  return "spread_rel";
        case kFeatDtLastTick: return "dt_last_tick";
        case kFeatSecOfDay:   return "sec_of_day";
        default:              return "unknown";
    }
}

struct FeatureConfig {
    double short_window_sec = 60.0;
    double long_window_sec = 900.0;
};

constexpr nanos_t kNoTs = std::numeric_limits<nanos_t>::min();

namespace detail {

struct Sums {
    double sum_r2 = 0.0;
    double sum_vol = 0.0;
    double sum_spread = 0.0;
    std::size_t count = 0;

    void include(const Tick& t, double r2v) {
        sum_r2 += r2v;
        sum_vol += t.volume;
        sum_spread += t.spread();
        ++count;
    }

    void exclude(const Tick& t, double r2v) {
        sum_r2 -= r2v;
        sum_vol -= t.volume;
        sum_spread -= t.spread();
        --count;
        if (sum_r2 < 0.0) sum_r2 = 0.0;
        if (sum_vol < 0.0) sum_vol = 0.0;
        if (sum_spread < 0.0) sum_spread = 0.0;
    }
};

inline double sq_log_return(double prev_mid, double mid) {
    if (!(prev_mid > 0.0) || !(mid > 0.0)) return 0.0;
    const double r = std::log(mid / prev_mid);
    return r * r;
}

inline void fill_row(const Tick& cur, double size, const Sums& s_short,
                     const Sums& s_long, nanos_t prev_ts,
                     const FeatureConfig& cfg, double* row) {
    constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
    constexpr nanos_t kNsPerDay = 86400LL * 1000000000LL;

    const double spread = cur.spread();
    const double depth = cur.bid_size + cur.ask_size;

    row[kFeatSpread] = spread;
    row[kFeatObi] = cur.obi();

    // The microprice leans toward the side with less depth
    if (depth > 0.0 && spread > 0.0) {
        const double micro = (cur.bid * cur.ask_size + cur.ask * cur.bid_size) / depth;
        row[kFeatMicroDev] = (micro - cur.mid()) / spread;
    } else {
        row[kFeatMicroDev] = 0.0;
    }

    row[kFeatRvShort] = std::sqrt(s_short.sum_r2);
    row[kFeatRvLong] = std::sqrt(s_long.sum_r2);
    row[kFeatVolShort] = s_short.sum_vol;
    row[kFeatVolLong] = s_long.sum_vol;
    row[kFeatSizeRelVol] = s_short.sum_vol > 0.0 ? size / s_short.sum_vol : kNaN;

    const double span = cfg.short_window_sec > 0.0 ? cfg.short_window_sec : 1.0;
    row[kFeatTickRate] = static_cast<double>(s_short.count) / span;

    const double spread_mean =
        s_short.count ? s_short.sum_spread / static_cast<double>(s_short.count) : kNaN;
    row[kFeatSpreadMean] = spread_mean;
    row[kFeatSpreadRel] = (spread_mean > 0.0) ? spread / spread_mean : kNaN;

    row[kFeatDtLastTick] =
        prev_ts == kNoTs ? kNaN : static_cast<double>(cur.ts - prev_ts) / kNanosPerSec;

    nanos_t tod = cur.ts % kNsPerDay;
    if (tod < 0) tod += kNsPerDay;
    row[kFeatSecOfDay] = static_cast<double>(tod) / kNanosPerSec;
}

} // namespace detail

inline void build_features(const TickView& ticks, const FillView& fills,
                           const FeatureConfig& cfg, double* out) {
    constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
    const std::size_t n_ticks = ticks.n, n_fills = fills.n;
    if (n_fills == 0) return;
    if (n_ticks == 0) {
        for (std::size_t i = 0; i < n_fills * kFeatCount; ++i) out[i] = kNaN;
        return;
    }

    // Squared log returns precomputed once so windows only add/subtract
    std::vector<double> r2(n_ticks, 0.0);
    for (std::size_t i = 1; i < n_ticks; ++i)
        r2[i] = detail::sq_log_return(ticks[i - 1].mid(), ticks[i].mid());

    const nanos_t short_ns = static_cast<nanos_t>(cfg.short_window_sec * kNanosPerSec);
    const nanos_t long_ns = static_cast<nanos_t>(cfg.long_window_sec * kNanosPerSec);

    detail::Sums s_short, s_long;
    std::size_t left_short = 0, left_long = 0;
    std::size_t right = 0;

    auto evict_to = [&](nanos_t now) {
        const nanos_t cut_long = now - long_ns;
        while (left_long < right && ticks.ts[left_long] < cut_long)
            s_long.exclude(ticks[left_long], r2[left_long]), ++left_long;
        const nanos_t cut_short = now - short_ns;
        while (left_short < right && ticks.ts[left_short] < cut_short)
            s_short.exclude(ticks[left_short], r2[left_short]), ++left_short;
    };

    for (std::size_t i = 0; i < n_fills; ++i) {
        const Fill f = fills[i];
        double* row = out + i * kFeatCount;

        while (right < n_ticks && ticks.ts[right] <= f.ts) {
            const Tick t = ticks[right];
            s_short.include(t, r2[right]);
            s_long.include(t, r2[right]);
            ++right;
            evict_to(t.ts);
        }
        if (right == 0) { // fill predates every quote
            for (int c = 0; c < kFeatCount; ++c) row[c] = kNaN;
            continue;
        }
        evict_to(f.ts);

        const nanos_t prev_ts = right >= 2 ? ticks.ts[right - 2] : kNoTs;
        detail::fill_row(ticks[right - 1], f.size, s_short, s_long, prev_ts, cfg, row);
    }
}

class LiveFeatures {
  public:
    explicit LiveFeatures(const FeatureConfig& cfg)
        : cfg_(cfg),
          short_ns_(static_cast<nanos_t>(cfg.short_window_sec * kNanosPerSec)),
          long_ns_(static_cast<nanos_t>(cfg.long_window_sec * kNanosPerSec)) {}

    void on_tick(const Tick& t) {
        const double r2v = have_last_ ? detail::sq_log_return(last_mid_, t.mid()) : 0.0;
        last_mid_ = t.mid();
        have_last_ = true;

        buf_.push_back(Item{t, r2v});
        s_long_.include(t, r2v);
        s_short_.include(t, r2v);

        evict(t.ts);
    }

    bool ready() const noexcept { return !buf_.empty(); }
    const Tick& last_tick() const { return buf_.back().tick; }

    void snapshot(double size, nanos_t as_of, double* out) {
        if (buf_.empty()) {
            for (int c = 0; c < kFeatCount; ++c)
                out[c] = std::numeric_limits<double>::quiet_NaN();
            return;
        }
        evict(as_of);
        const nanos_t prev_ts = buf_.size() >= 2 ? buf_[buf_.size() - 2].tick.ts : kNoTs;
        detail::fill_row(buf_.back().tick, size, s_short_, s_long_, prev_ts, cfg_, out);
    }

    void snapshot(double size, double* out) {
        snapshot(size, buf_.empty() ? 0 : buf_.back().tick.ts, out);
    }

  private:
    struct Item { Tick tick; double r2; };

    void evict(nanos_t now) {
        const nanos_t cut_long = now - long_ns_;
        while (!buf_.empty() && buf_.front().tick.ts < cut_long) {
            s_long_.exclude(buf_.front().tick, buf_.front().r2);
            if (short_left_ == front_seq_) {
                s_short_.exclude(buf_.front().tick, buf_.front().r2);
                ++short_left_;
            }
            buf_.pop_front();
            ++front_seq_;
        }
        const nanos_t cut_short = now - short_ns_;
        for (std::size_t i = short_left_ - front_seq_;
             i < buf_.size() && buf_[i].tick.ts < cut_short; ++i) {
            s_short_.exclude(buf_[i].tick, buf_[i].r2);
            ++short_left_;
        }
    }

    FeatureConfig cfg_;
    nanos_t short_ns_;
    nanos_t long_ns_;
    std::deque<Item> buf_;
    detail::Sums s_short_, s_long_;
    std::uint64_t front_seq_ = 0;  // sequence number of buf_.front()
    std::uint64_t short_left_ = 0; // sequence number of the short window's left edge
    double last_mid_ = 0.0;
    bool have_last_ = false;
};


class LiveCounterpartyHistory {
  public:
    LiveCounterpartyHistory(double prior_rate = 0.5, double prior_weight = 5.0)
        : prior_rate_(prior_rate), prior_weight_(prior_weight) {}

    double rate_for(std::int32_t counterparty) const {
        const auto it = acc_.find(counterparty);
        const double toxic = it == acc_.end() ? 0.0 : it->second.toxic;
        const double total = it == acc_.end() ? 0.0 : it->second.total;
        return (toxic + prior_rate_ * prior_weight_) / (total + prior_weight_);
    }

    double fill_count(std::int32_t counterparty) const {
        const auto it = acc_.find(counterparty);
        return it == acc_.end() ? 0.0 : it->second.total;
    }

    void record(std::int32_t counterparty, std::int8_t label) {
        if (label < 0) return;
        Acc& a = acc_[counterparty];
        a.total += 1.0;
        a.toxic += static_cast<double>(label);
    }

    std::size_t counterparties() const noexcept { return acc_.size(); }

  private:
    struct Acc { double toxic = 0.0; double total = 0.0; };
    std::unordered_map<std::int32_t, Acc> acc_;
    double prior_rate_;
    double prior_weight_;
};


inline void counterparty_history(const FillView& fills,
                                 const std::int8_t* labels,
                                 double prior_rate, double prior_weight,
                                 double* out_rate, double* out_count) {
    struct Acc { double toxic = 0.0; double total = 0.0; };
    const std::size_t n_fills = fills.n;
    std::unordered_map<std::int32_t, Acc> acc;
    acc.reserve(n_fills / 8 + 16);

    for (std::size_t i = 0; i < n_fills; ++i) {
        Acc& a = acc[fills[i].counterparty];
        out_rate[i] = (a.toxic + prior_rate * prior_weight) / (a.total + prior_weight);
        out_count[i] = a.total;

        const std::int8_t y = labels ? labels[i] : static_cast<std::int8_t>(-1);
        if (y >= 0) {
            a.total += 1.0;
            a.toxic += static_cast<double>(y);
        }
    }
}

} // namespace fxtox
