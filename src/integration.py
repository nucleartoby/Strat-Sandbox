import numpy as np
import pandas as pd

import native


def attach_as_feature(features: pd.DataFrame, toxicity_proba,
                      column_name: str = "toxicity_score") -> pd.DataFrame:
    out = features.copy()
    out[column_name] = np.asarray(toxicity_proba, dtype=float)
    return out


def meta_label_gate(primary_signal: pd.Series, toxicity_proba,
                    act_threshold: float = 0.5, min_size: float = 0.0,
                    size_curve: str = "linear") -> pd.DataFrame:
    p = pd.Series(np.asarray(toxicity_proba, dtype=float),
                  index=primary_signal.index).clip(0.0, 1.0)

    if size_curve == "linear":
        mult = (1.0 - p / act_threshold).clip(0.0, 1.0)
    elif size_curve == "kelly":
        mult = (1.0 - 2.0 * p).clip(0.0, 1.0)
    elif size_curve == "binary":
        mult = (p < act_threshold).astype(float)
    else:
        raise ValueError(f"unknown size_curve: {size_curve!r}")

    act = p < act_threshold
    mult = mult.where(act, 0.0)
    mult = mult.where(mult >= min_size, 0.0)  # do not send dust

    return pd.DataFrame({
        "primary_signal": primary_signal,
        "toxicity_proba": p,
        "act": act,
        "size_multiplier": mult,
        "gated_signal": primary_signal.where(act, 0) * mult,})


def required_spread(toxicity_proba, alpha_mu: float) -> np.ndarray:
    p = np.clip(np.asarray(toxicity_proba, dtype=float), 0.0, 1.0)
    return 2.0 * alpha_mu * p


def make_risk_gate(alpha_mu: float = 0.0, widen_above: float = 0.65,
                   suspend_above: float = 0.85, tighten_below: float = 0.35,
                   hysteresis: float = 0.05, min_dwell_seconds: float = 1.0,
                   max_widen_factor: float = 5.0, **kwargs):
    return native.make_gate(
        alpha_mu=alpha_mu, widen_above=widen_above, suspend_above=suspend_above,
        tighten_below=tighten_below, hysteresis=hysteresis,
        min_dwell_ns=int(min_dwell_seconds * 1e9),
        max_widen_factor=max_widen_factor, **kwargs)


def simulate_gate(timestamps, toxicity_proba, base_spread, alpha_mu: float = 0.0,
                  **gate_kwargs) -> pd.DataFrame:
    gate = make_risk_gate(alpha_mu=alpha_mu, **gate_kwargs)
    ts = native._i64(timestamps)
    p = np.asarray(toxicity_proba, dtype=float)
    base = np.broadcast_to(np.asarray(base_spread, dtype=float), p.shape)

    actions, spreads, mults, changed = [], [], [], []
    for t, pi, b in zip(ts, p, base):
        d = gate.on_update(int(t), float(pi), float(b))
        actions.append(str(d.action).rsplit(".", 1)[-1].lower())
        spreads.append(d.spread)
        mults.append(d.multiplier)
        changed.append(d.changed)

    return pd.DataFrame({
        "timestamp": pd.to_datetime(ts, utc=True),
        "toxicity_proba": p, "base_spread": base,
        "action": actions, "quoted_spread": spreads,
        "multiplier": mults, "state_changed": changed,})


def gate_summary(sim: pd.DataFrame) -> pd.DataFrame:
    total = len(sim)
    out = sim.groupby("action").agg(
        n=("action", "size"),
        mean_proba=("toxicity_proba", "mean"),
        mean_multiplier=("multiplier", "mean"),).reset_index()
    
    out["share"] = out["n"] / max(total, 1)
    out.attrs["state_changes"] = int(sim["state_changed"].sum())
    out.attrs["changes_per_1k"] = 1000.0 * sim["state_changed"].sum() / max(total, 1)
    return out
