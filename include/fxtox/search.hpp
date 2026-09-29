#pragma once

#include <algorithm>
#include <cstddef>
#include <limits>
#include <vector>

namespace fxtox {

class BlockIndex {
  public:
    static constexpr std::size_t kBlock = 64;
    static constexpr std::size_t npos = static_cast<std::size_t>(-1);

    BlockIndex() = default;

    BlockIndex(const double* x, std::size_t n) { build(x, n); }

    void build(const double* x, std::size_t n) {
        x_ = x;
        n_ = n;
        const std::size_t nb = (n + kBlock - 1) / kBlock;
        bmin_.assign(nb, std::numeric_limits<double>::infinity());
        bmax_.assign(nb, -std::numeric_limits<double>::infinity());
        for (std::size_t i = 0; i < n; ++i) {
            const std::size_t b = i / kBlock;
            if (x[i] < bmin_[b]) bmin_[b] = x[i];
            if (x[i] > bmax_[b]) bmax_[b] = x[i];
        }
    }

    // First index i in [lo, hi] with x[i] >= thr else npos
    std::size_t first_ge(std::size_t lo, std::size_t hi, double thr) const {
        return scan(lo, hi, [&](std::size_t b) { return bmax_[b] >= thr; },
                    [&](std::size_t i) { return x_[i] >= thr; });
    }

    std::size_t first_le(std::size_t lo, std::size_t hi, double thr) const {
        return scan(lo, hi, [&](std::size_t b) { return bmin_[b] <= thr; },
                    [&](std::size_t i) { return x_[i] <= thr; });
    }

    std::size_t size() const noexcept { return n_; }

  private:
    template <typename BlockOk, typename ElemOk>
    std::size_t scan(std::size_t lo, std::size_t hi, BlockOk block_ok, ElemOk elem_ok) const {
        if (n_ == 0 || lo >= n_ || hi < lo) return npos;
        hi = std::min(hi, n_ - 1);

        std::size_t i = lo;
        while (i <= hi) {
            const std::size_t b = i / kBlock;
            const std::size_t block_end = std::min((b + 1) * kBlock - 1, hi);
            if (!block_ok(b)) { i = block_end + 1; continue; }
            for (; i <= block_end; ++i) {
                if (elem_ok(i)) return i;
            }
        }
        return npos;
    }

    const double* x_ = nullptr;
    std::size_t n_ = 0;
    std::vector<double> bmin_;
    std::vector<double> bmax_;
};

} // namespace fxtox
