// Reference: Easley, Lopez de Prado & O'Hara (2012), "Flow Toxicity and
// Liquidity in a High-Frequency World", RFS 25(5).
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include "fxtox/maths.hpp"
#include "fxtox/types.hpp"

namespace fxtox {

enum class BvcDist : int { Normal = 0, StudentT = 1 };

enum class BvcGranularity : int { Bucket = 0, Tick = 1 };

enum class SigmaMode : int { Expanding = 0, Fixed = 1 };

struct VpinConfig {
    double bucket_volume = 0.0;
    std::size_t window = 50;
    BvcDist dist = BvcDist::Normal;
    double df = 0.25;
    BvcGranularity granularity = BvcGranularity::Bucket;
    SigmaMode sigma_mode = SigmaMode::Expanding;
    double sigma_fixed = 0.0;     // used when sigma_mode == Fixed
    std::size_t sigma_warmup = 20;
    std::size_t pctile_bins = 65536;
    std::size_t pctile_warmup = 30;
};

// One VPIN reading emitted when a bucket closes
struct VpinReading {
    nanos_t ts_end;
    double vpin;
    double percentile;
    double imbalance;
    double v_buy;
    double v_sell;
    double price_end;
    bool warmup;
};

class OnlineQuantiler {
  public:
    explicit OnlineQuantiler(std::size_t bins = 65536, double lo = 0.0, double hi = 1.0)
        : tree_(bins + 1, 0), bins_(bins), lo_(lo), inv_span_(1.0 / (hi - lo)) {}

    double observe(double v) {
        add(bin_of(v));
        return static_cast<double>(prefix(bin_of(v))) / static_cast<double>(n_);
    }

    double percentile_of(double v) const {
        return n_ ? static_cast<double>(prefix(bin_of(v))) / static_cast<double>(n_)
                  : std::numeric_limits<double>::quiet_NaN();
    }

    std::uint64_t count() const noexcept { return n_; }

  private:
    std::size_t bin_of(double v) const noexcept {
        const double scaled = (v - lo_) * inv_span_ * static_cast<double>(bins_);
        if (!(scaled > 0.0)) return 0; // also catches NaN
        const std::size_t idx = static_cast<std::size_t>(scaled);
        return idx >= bins_ ? bins_ - 1 : idx;
    }

    void add(std::size_t bin) {
        ++n_;
        for (std::size_t i = bin + 1; i <= bins_; i += i & (~i + 1)) ++tree_[i];
    }

    std::uint64_t prefix(std::size_t bin) const {
        std::uint64_t s = 0;
        for (std::size_t i = bin + 1; i > 0; i -= i & (~i + 1)) s += tree_[i];
        return s;
    }

    std::vector<std::uint64_t> tree_;
    std::size_t bins_;
    double lo_;
    double inv_span_;
    std::uint64_t n_ = 0;
};

class VpinEngine {
  public:
    explicit VpinEngine(const VpinConfig& cfg)
        : cfg_(cfg),
          roll_(cfg.window ? cfg.window : 1),
          quant_(cfg.pctile_bins) {}

    template <typename OnReading>
    void on_tick(const Tick& t, OnReading&& cb) {
        const double price = t.mid();
        if (!have_open_) {
            open_ts_ = t.ts;
            open_price_ = price;
            prev_bucket_price_ = std::isnan(prev_bucket_price_) ? price : prev_bucket_price_;
            have_open_ = true;
        }

        double remaining = t.volume;
        if (!(remaining > 0.0)) { last_price_ = price; return; }

        while (remaining > 0.0) {
            const double room = cfg_.bucket_volume - filled_;
            const double take = std::min(room, remaining);

            if (cfg_.granularity == BvcGranularity::Tick) {
                const double dp = std::isnan(last_price_) ? 0.0 : price - last_price_;
                const double frac = buy_fraction(dp, tick_sigma());
                tick_v_buy_ += take * frac;
            }

            filled_ += take;
            remaining -= take;

            if (filled_ >= cfg_.bucket_volume - kVolEps) {
                cb(close_bucket(t.ts, price));
                // Any leftover volume opens next bucket at same tick
                if (remaining > 0.0) { open_ts_ = t.ts; open_price_ = price; }
            }
        }

        if (cfg_.granularity == BvcGranularity::Tick && !std::isnan(last_price_)) {
            tick_dp_.push(price - last_price_);
        }
        last_price_ = price;
    }

    void on_tick(const Tick& t, std::vector<VpinReading>& out) {
        on_tick(t, [&out](const VpinReading& r) { out.push_back(r); });
    }

    double vpin() const noexcept {
        return roll_.full() ? roll_.mean() : std::numeric_limits<double>::quiet_NaN();
    }
    std::uint64_t buckets_closed() const noexcept { return n_buckets_; }
    const VpinConfig& config() const noexcept { return cfg_; }

  private:
    static constexpr double kVolEps = 1e-9;

    double tick_sigma() const {
        if (cfg_.sigma_mode == SigmaMode::Fixed) return cfg_.sigma_fixed;
        return tick_dp_.count() >= 2 ? tick_dp_.stddev_pop() : 0.0;
    }

    double bucket_sigma() const {
        if (cfg_.sigma_mode == SigmaMode::Fixed) return cfg_.sigma_fixed;
        return bucket_dp_.count() >= 2 ? bucket_dp_.stddev_pop() : 0.0;
    }

    double buy_fraction(double dp, double sigma) const {
        if (!(sigma > 0.0)) return 0.5;
        const double z = dp / sigma;
        return cfg_.dist == BvcDist::Normal ? norm_cdf(z) : student_t_cdf(z, cfg_.df);
    }

    VpinReading close_bucket(nanos_t ts_end, double price_end) {
        const double V = cfg_.bucket_volume;
        double v_buy;
        bool warmup;

        if (cfg_.granularity == BvcGranularity::Tick) {
            v_buy = tick_v_buy_;
            warmup = cfg_.sigma_mode == SigmaMode::Expanding &&
                     tick_dp_.count() < cfg_.sigma_warmup;
        } else {
            const double dp = price_end - prev_bucket_price_;
            const double sigma = bucket_sigma();
            v_buy = V * buy_fraction(dp, sigma);
            warmup = cfg_.sigma_mode == SigmaMode::Expanding &&
                     bucket_dp_.count() < cfg_.sigma_warmup;
            bucket_dp_.push(dp);
        }

        const double v_sell = V - v_buy;
        const double imb = std::fabs(v_buy - v_sell) / V;

        const bool full = roll_.push(imb);
        ++n_buckets_;

        VpinReading r{};
        r.ts_end = ts_end;
        r.imbalance = imb;
        r.v_buy = v_buy;
        r.v_sell = v_sell;
        r.price_end = price_end;
        r.warmup = warmup;
        if (full) {
            r.vpin = roll_.mean();
            const double pct = quant_.observe(r.vpin);
            r.percentile = quant_.count() >= cfg_.pctile_warmup
                               ? pct : std::numeric_limits<double>::quiet_NaN();
        } else {
            r.vpin = std::numeric_limits<double>::quiet_NaN();
            r.percentile = std::numeric_limits<double>::quiet_NaN();
        }

        prev_bucket_price_ = price_end;
        filled_ -= V;
        if (filled_ < 0.0) filled_ = 0.0;
        tick_v_buy_ = 0.0;
        open_price_ = price_end;
        open_ts_ = ts_end;
        return r;
    }

    VpinConfig cfg_;
    RollingSum roll_;
    OnlineQuantiler quant_;
    Welford bucket_dp_;
    Welford tick_dp_;

    double filled_ = 0.0;
    double tick_v_buy_ = 0.0;
    double last_price_ = std::numeric_limits<double>::quiet_NaN();
    double prev_bucket_price_ = std::numeric_limits<double>::quiet_NaN();
    double open_price_ = 0.0;
    nanos_t open_ts_ = 0;
    bool have_open_ = false;
    std::uint64_t n_buckets_ = 0;
};

struct VpinBatchOut {
    nanos_t* ts_end = nullptr;
    double* vpin = nullptr;
    double* percentile = nullptr;
    double* imbalance = nullptr;
    double* v_buy = nullptr;
    double* v_sell = nullptr;
    double* price_end = nullptr;
    std::uint8_t* warmup = nullptr;
};

inline std::size_t compute_vpin_batch(const TickView& ticks, const VpinConfig& cfg,
                                      const VpinBatchOut& out, std::size_t cap) {
    VpinEngine engine(cfg);
    std::size_t k = 0;
    for (std::size_t i = 0; i < ticks.n && k < cap; ++i) {
        engine.on_tick(ticks[i], [&](const VpinReading& r) {
            if (k >= cap) return;
            if (out.ts_end) out.ts_end[k] = r.ts_end;
            if (out.vpin) out.vpin[k] = r.vpin;
            if (out.percentile) out.percentile[k] = r.percentile;
            if (out.imbalance) out.imbalance[k] = r.imbalance;
            if (out.v_buy) out.v_buy[k] = r.v_buy;
            if (out.v_sell) out.v_sell[k] = r.v_sell;
            if (out.price_end) out.price_end[k] = r.price_end;
            if (out.warmup) out.warmup[k] = r.warmup ? 1u : 0u;
            ++k;
        });
    }
    return k;
}

// Upper bound on the number of buckets a tick series can produce
inline std::size_t bucket_capacity(const TickView& ticks, double bucket_volume) {
    if (!(bucket_volume > 0.0) || !ticks.volume) return 0;
    double total = 0.0;
    for (std::size_t i = 0; i < ticks.n; ++i) total += ticks.volume[i];
    return static_cast<std::size_t>(total / bucket_volume) + 1;
}

} // namespace fxtox
