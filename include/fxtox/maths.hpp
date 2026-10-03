#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace fxtox {

inline double norm_cdf(double z) noexcept {
    return 0.5 * std::erfc(-z * 0.70710678118654752440); // -z / sqrt(2)
}

namespace detail {

inline double betacf(double a, double b, double x) noexcept {
    constexpr int kMaxIter = 300;
    constexpr double kEps = 3.0e-16;
    constexpr double kTiny = 1.0e-300;

    const double qab = a + b, qap = a + 1.0, qam = a - 1.0;
    double c = 1.0;
    double d = 1.0 - qab * x / qap;
    if (std::fabs(d) < kTiny) d = kTiny;
    d = 1.0 / d;
    double h = d;

    for (int m = 1; m <= kMaxIter; ++m) {
        const int m2 = 2 * m;
        double aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if (std::fabs(d) < kTiny) d = kTiny;
        c = 1.0 + aa / c;
        if (std::fabs(c) < kTiny) c = kTiny;
        d = 1.0 / d;
        h *= d * c;

        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if (std::fabs(d) < kTiny) d = kTiny;
        c = 1.0 + aa / c;
        if (std::fabs(c) < kTiny) c = kTiny;
        d = 1.0 / d;
        const double del = d * c;
        h *= del;
        if (std::fabs(del - 1.0) < kEps) break;
    }
    return h;

}
inline double betai(double a, double b, double x) noexcept {
    if (x <= 0.0) return 0.0;
    if (x >= 1.0) return 1.0;
    const double lbeta = std::lgamma(a + b) - std::lgamma(a) - std::lgamma(b);
    const double front = std::exp(lbeta + a * std::log(x) + b * std::log1p(-x));
    return x < (a + 1.0) / (a + b + 2.0) ? front * betacf(a, b, x) / a
                                         : 1.0 - front * betacf(b, a, 1.0 - x) / b;
}

}
inline double student_t_cdf(double t, double df) noexcept {
    if (df <= 0.0) return norm_cdf(t);
    const double x = df / (df + t * t);
    const double p = 0.5 * detail::betai(0.5 * df, 0.5, x);
    return t > 0.0 ? 1.0 - p : p;
}

class Welford {
  public:
    void push(double x) noexcept {
        ++n_;
        const double delta = x - mean_;
        mean_ += delta / static_cast<double>(n_);
        m2_ += delta * (x - mean_);
    }

    std::uint64_t count() const noexcept { return n_; }
    double mean() const noexcept { return mean_; }
    double variance() const noexcept {
        return n_ > 1 ? m2_ / static_cast<double>(n_ - 1) : 0.0;
    }
    double variance_pop() const noexcept {
        return n_ > 0 ? m2_ / static_cast<double>(n_) : 0.0;
    }
    double stddev() const noexcept { return std::sqrt(variance()); }
    double stddev_pop() const noexcept { return std::sqrt(variance_pop()); }
    void reset() noexcept { n_ = 0; mean_ = 0.0; m2_ = 0.0; }

  private:
    std::uint64_t n_ = 0;
    double mean_ = 0.0;
    double m2_ = 0.0;
};

class RollingSum {
  public:
    explicit RollingSum(std::size_t window)
        : buf_(window, 0.0), window_(window) {}

    bool push(double x) noexcept {
        if (filled_ == window_) sum_ -= buf_[head_];
        else ++filled_;
        buf_[head_] = x;
        sum_ += x;
        head_ = (head_ + 1 == window_) ? 0 : head_ + 1;

        if (++since_refresh_ >= kRefreshEvery) refresh();
        return filled_ == window_;
    }

    double sum() const noexcept { return sum_; }
    double mean() const noexcept {
        return filled_ ? sum_ / static_cast<double>(filled_) : 0.0;
    }
    bool full() const noexcept { return filled_ == window_; }
    std::size_t size() const noexcept { return filled_; }
    std::size_t window() const noexcept { return window_; }

  private:
    static constexpr std::uint32_t kRefreshEvery = 1u << 16;

    void refresh() noexcept {
        double s = 0.0;
        for (std::size_t i = 0; i < filled_; ++i) s += buf_[i];
        sum_ = s;
        since_refresh_ = 0;
    }

    std::vector<double> buf_;
    std::size_t window_;
    std::size_t head_ = 0;
    std::size_t filled_ = 0;
    double sum_ = 0.0;
    std::uint32_t since_refresh_ = 0;
};

} // namespace fxtox
