#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include "fxtox/types.hpp"

namespace fxtox {

struct MidSeries {
    const nanos_t* ts = nullptr;
    const double* mid = nullptr;
    std::size_t n = 0;
};

inline std::size_t prevailing_index(const MidSeries& s, nanos_t t, std::size_t hint = 0) {
    if (s.n == 0 || t < s.ts[0]) return static_cast<std::size_t>(-1);
    std::size_t i = hint;
    while (i + 1 < s.n && s.ts[i + 1] <= t) ++i;
    if (s.ts[i] > t) { // hint overshot fall back to a bisect
        i = static_cast<std::size_t>(
                std::upper_bound(s.ts, s.ts + s.n, t) - s.ts);
        if (i == 0) return static_cast<std::size_t>(-1);
        --i;
    }
    return i;
}

// Compute LP markouts for every fill at every horizon.
inline void compute_markouts(const FillView& fills, const MidSeries& mids,
                             const double* horizons_sec, std::size_t n_h,
                             double* out) {
    constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
    const std::size_t n_fills = fills.n;
    if (n_fills == 0 || n_h == 0) return;

    std::vector<std::size_t> cursor(n_h, 0);
    const nanos_t last_ts = mids.n ? mids.ts[mids.n - 1] : 0;

    for (std::size_t i = 0; i < n_fills; ++i) {
        const Fill& f = fills[i];
        const double sgn = f.signum();
        for (std::size_t h = 0; h < n_h; ++h) {
            const nanos_t target =
                f.ts + static_cast<nanos_t>(horizons_sec[h] * kNanosPerSec);
            if (mids.n == 0 || target > last_ts) { out[i * n_h + h] = kNaN; continue; }
            const std::size_t j = prevailing_index(mids, target, cursor[h]);
            if (j == static_cast<std::size_t>(-1)) { out[i * n_h + h] = kNaN; continue; }
            cursor[h] = j;
            out[i * n_h + h] = sgn * (f.price - mids.mid[j]);
        }
    }
}

inline void mid_at_fill(const FillView& fills, const MidSeries& mids, double* out) {
    std::size_t cursor = 0;
    for (std::size_t i = 0; i < fills.n; ++i) {
        const std::size_t j = prevailing_index(mids, fills[i].ts, cursor);
        if (j == static_cast<std::size_t>(-1)) {
            out[i] = std::numeric_limits<double>::quiet_NaN();
        } else {
            cursor = j;
            out[i] = mids.mid[j];
        }
    }
}

inline double markout_curve(double tau, double half_spread, double alpha_mu, double lambda) {
    return half_spread - alpha_mu * (1.0 - std::exp(-lambda * tau));
}

} // namespace fxtox
