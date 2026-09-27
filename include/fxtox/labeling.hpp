#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "fxtox/markout.hpp"
#include "fxtox/search.hpp"
#include "fxtox/types.hpp"

namespace fxtox {

enum class LabelRule : int { MarkoutThreshold = 0, Crossback = 1, TripleBarrier = 2 };

enum class BarrierCenter : int { ExecPrice = 0, MidAtFill = 1 };

struct LabelConfig {
    LabelRule rule = LabelRule::Crossback;
    double horizon_sec = 30.0;  // G
    double theta = 0.0;         // barrier / threshold width in price units
    double min_adverse = 0.0;
    BarrierCenter center = BarrierCenter::MidAtFill;
};

// Labels are 1 (toxic), 0 (benign), or -1 (undecidable
constexpr std::int8_t kLabelToxic = 1;
constexpr std::int8_t kLabelBenign = 0;
constexpr std::int8_t kLabelUnknown = -1;

inline void label_fills(const FillView& fills,
                        const MidSeries& mids, const BlockIndex& index,
                        const LabelConfig& cfg,
                        const double* markouts, std::int8_t* out) {
    const std::size_t n_fills = fills.n;
    const nanos_t horizon_ns = static_cast<nanos_t>(cfg.horizon_sec * kNanosPerSec);
    const nanos_t last_ts = mids.n ? mids.ts[mids.n - 1] : 0;
    std::size_t cursor = 0;

    for (std::size_t i = 0; i < n_fills; ++i) {
        const Fill& f = fills[i];

        if (cfg.rule == LabelRule::MarkoutThreshold) {
            const double m = markouts ? markouts[i] : std::numeric_limits<double>::quiet_NaN();
            out[i] = std::isnan(m) ? kLabelUnknown
                                   : (m < -cfg.theta ? kLabelToxic : kLabelBenign);
            continue;
        }

        const nanos_t end_ts = f.ts + horizon_ns;
        if (mids.n == 0 || end_ts > last_ts) { out[i] = kLabelUnknown; continue; }

        const std::size_t at = prevailing_index(mids, f.ts, cursor);
        if (at == static_cast<std::size_t>(-1)) { out[i] = kLabelUnknown; continue; }
        cursor = at;
        const std::size_t lo = at + 1;
        const std::size_t hi = prevailing_index(mids, end_ts, at);
        if (hi == static_cast<std::size_t>(-1) || lo > hi) { out[i] = kLabelUnknown; continue; }

        const bool client_bought = f.side == Side::Buy;

        if (cfg.rule == LabelRule::Crossback) {
            const double barrier = client_bought ? f.price + cfg.min_adverse
                                                 : f.price - cfg.min_adverse;
            const std::size_t hit = client_bought ? index.first_ge(lo, hi, barrier)
                                                  : index.first_le(lo, hi, barrier);
            out[i] = (hit != BlockIndex::npos) ? kLabelToxic : kLabelBenign;
            continue;
        }

        const double center = (cfg.center == BarrierCenter::MidAtFill)
                                  ? mids.mid[at] : f.price;
        const double up = center + cfg.theta;
        const double down = center - cfg.theta;
        const std::size_t t_up = index.first_ge(lo, hi, up);
        const std::size_t t_down = index.first_le(lo, hi, down);

        const std::size_t t_fav = client_bought ? t_up : t_down;
        const std::size_t t_adv = client_bought ? t_down : t_up;

        if (t_fav == BlockIndex::npos) out[i] = kLabelBenign;       // never paid off
        else if (t_adv == BlockIndex::npos) out[i] = kLabelToxic;   // only ran our way
        else out[i] = (t_fav < t_adv) ? kLabelToxic : kLabelBenign;
    }
}

} // namespace fxtox
