#pragma once
#include <algorithm>
#include <cmath>
#include "fxtox/types.hpp"

namespace fxtox {

enum class QuoteAction : int { Tighten = 0, Normal = 1, Widen = 2, Suspend = 3 };

inline const char* quote_action_name(QuoteAction a) {
    switch (a) {
        case QuoteAction::Tighten: return "tighten";
        case QuoteAction::Normal:  return "normal";
        case QuoteAction::Widen:   return "widen";
        case QuoteAction::Suspend: return "suspend";
    }
    return "unknown";
}

struct GateConfig {
    double tighten_below = 0.35;
    double widen_above = 0.65;
    double suspend_above = 0.85;

    double hysteresis = 0.05;
    nanos_t min_dwell_ns = 1000000000LL; // 1s

    double tighten_factor = 0.9;
    double max_widen_factor = 5.0;

    double alpha_mu = 0.0;
    double widen_factor = 2.5; // fallback when alpha_mu is unset
};

struct GateDecision {
    QuoteAction action = QuoteAction::Normal;
    double spread = 0.0;       // spread to quote 0 when suspended
    double multiplier = 1.0;   // spread / base_spread
    bool changed = false;      // did the action change from the previous one
};

class RiskGate {
  public:
    explicit RiskGate(const GateConfig& cfg) : cfg_(cfg) {}

    GateDecision on_update(nanos_t ts, double toxicity_proba, double base_spread) {
        const double p = std::isnan(toxicity_proba) ? 0.5
                                                    : std::min(std::max(toxicity_proba, 0.0), 1.0);
        const QuoteAction target = classify(p);
        QuoteAction action = state_;

        if (target != state_) {
            const bool escalating = static_cast<int>(target) > static_cast<int>(state_);
            if (escalating || !have_state_ || ts - since_ts_ >= cfg_.min_dwell_ns) {
                action = target;
            }
        }

        GateDecision d;
        d.changed = !have_state_ || action != state_;
        if (d.changed) { since_ts_ = ts; }
        state_ = action;
        have_state_ = true;
        last_p_ = p;

        d.action = action;
        d.multiplier = multiplier_for(action, p, base_spread);
        d.spread = action == QuoteAction::Suspend ? 0.0 : base_spread * d.multiplier;
        return d;
    }

    QuoteAction state() const noexcept { return state_; }
    double last_probability() const noexcept { return last_p_; }

  private:
    QuoteAction classify(double p) const {
        const double h = cfg_.hysteresis;
        switch (state_) {
            case QuoteAction::Suspend:
                if (p > cfg_.suspend_above - h) return QuoteAction::Suspend;
                break;
            case QuoteAction::Widen:
                if (p >= cfg_.suspend_above) return QuoteAction::Suspend;
                if (p > cfg_.widen_above - h) return QuoteAction::Widen;
                break;
            default:
                break;
        }
        if (p >= cfg_.suspend_above) return QuoteAction::Suspend;
        if (p >= cfg_.widen_above) return QuoteAction::Widen;
        if (p <= cfg_.tighten_below) return QuoteAction::Tighten;
        return QuoteAction::Normal;
    }

    double multiplier_for(QuoteAction a, double p, double base_spread) const {
        if (a == QuoteAction::Suspend) return 0.0;
        if (a == QuoteAction::Tighten) return cfg_.tighten_factor;
        if (a == QuoteAction::Normal) return 1.0;

        if (cfg_.alpha_mu > 0.0 && base_spread > 0.0) {
            const double required = 2.0 * cfg_.alpha_mu * p;
            const double mult = required / base_spread;
            return std::min(std::max(mult, 1.0), cfg_.max_widen_factor);
        }
        return std::min(cfg_.widen_factor, cfg_.max_widen_factor);
    }

    GateConfig cfg_;
    QuoteAction state_ = QuoteAction::Normal;
    bool have_state_ = false;
    nanos_t since_ts_ = 0;
    double last_p_ = 0.5;
};

} // namespace fxtox
