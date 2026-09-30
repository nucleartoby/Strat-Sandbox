#pragma once

#include <cstddef>
#include <cstdint>

namespace fxtox {

// Nanoseconds since unix epoch to match numpy datetime64
using nanos_t = std::int64_t;

constexpr double kNanosPerSec = 1e9;

struct Tick {
    nanos_t ts;       // exchange/feed timestamp
    double  bid;
    double  ask;
    double  bid_size; // top of book depth 0 if the feed does not supply it
    double  ask_size;
    double  volume;   // traded or proxied volume attributed to this tick

    constexpr double mid() const noexcept { return 0.5 * (bid + ask); }
    constexpr double spread() const noexcept { return ask - bid; }

    constexpr double obi() const noexcept {
        const double denom = bid_size + ask_size;
        return denom > 0.0 ? (bid_size - ask_size) / denom : 0.0;
    }
};

enum class Side : std::int8_t { Sell = -1, Buy = 1 };

struct Fill {
    nanos_t ts;
    double  price;
    double  size;
    Side    side;
    std::int32_t counterparty;

    constexpr double signum() const noexcept { return static_cast<double>(side); }
};

struct Bucket {
    nanos_t ts_start;
    nanos_t ts_end;
    double  v_buy;
    double  v_sell;
    double  volume;
    double  price_start;
    double  price_end;

    constexpr double imbalance() const noexcept {
        const double d = v_buy - v_sell;
        const double a = d < 0.0 ? -d : d;
        return volume > 0.0 ? a / volume : 0.0;
    }
};

struct TickView {
    const nanos_t* ts = nullptr;
    const double* bid = nullptr;
    const double* ask = nullptr;
    const double* bid_size = nullptr;
    const double* ask_size = nullptr;
    const double* volume = nullptr;   // optional treated as 0 when absent
    std::size_t n = 0;

    Tick operator[](std::size_t i) const noexcept {
        return Tick{ts[i], bid[i], ask[i],
                    bid_size ? bid_size[i] : 0.0,
                    ask_size ? ask_size[i] : 0.0,
                    volume ? volume[i] : 0.0};
    }
    std::size_t size() const noexcept { return n; }
    bool empty() const noexcept { return n == 0; }
};

struct FillView {
    const nanos_t* ts = nullptr;
    const double* price = nullptr;
    const double* size = nullptr;
    const std::int8_t* side = nullptr;
    const std::int32_t* counterparty = nullptr;
    std::size_t n = 0;

    Fill operator[](std::size_t i) const noexcept {
        return Fill{ts[i], price[i], size ? size[i] : 0.0,
                    side[i] >= 0 ? Side::Buy : Side::Sell,
                    counterparty ? counterparty[i] : -1};
    }
    std::size_t size_of() const noexcept { return n; }
    bool empty() const noexcept { return n == 0; }
};

} // namespace fxtox
