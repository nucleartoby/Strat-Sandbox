import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK = "#0b0b0b"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"
SERIES_1 = "#2a78d6" 
SERIES_2 = "#eb6834"
POS = "#2a78d6" 
NEG = "#e34948" 
STATUS = {"good": "#0ca30c", "warning": "#fab219", "critical": "#d03b3b"}


def _pips(x, pip: float) -> float:
    return float(x) / pip


def build_summary(*, symbol, ticks, fills, labels, fit, econ, cv_result, deciles,
              sweep, monotonicity, horizons, label_horizon, pip=1e-4,
              holdout=None, gate=None, problems=None) -> dict:
    folds = cv_result["fold_results"]
    short_h, long_h = horizons[0], horizons[-1]

    def mk(h):
        col = f"markout_{int(h)}s" if float(h).is_integer() else f"markout_{h}s"
        return fills[col].mean() if col in fills else np.nan

    keep_h = (label_horizon if label_horizon > short_h
              else next((h for h in horizons if h >= 30), long_h))
    spread_earned = mk(short_h)
    net_edge = mk(keep_h) if f"markout_{int(keep_h)}s" in fills else mk(long_h)
    adverse = spread_earned - net_edge

    best = None
    if gate is not None:
        if gate["uplift"] > 0:
            best = {"threshold": gate["threshold"], "blocked": gate["blocked_share"],
                    "uplift_pct": 100.0 * gate["uplift"],
                    "uplift_ci": tuple(100.0 * v for v in gate["uplift_ci"]),
                    "pnl_before": gate["pnl_ungated"], "pnl_after": gate["pnl_gated"]}
    elif holdout is None and sweep is not None and len(sweep):
        row = sweep.loc[sweep["pnl_improvement"].idxmax()]
        if row["pnl_improvement"] > 0:
            base = abs(row["pnl_ungated"]) or 1.0
            best = {"threshold": float(row["threshold"]),
                    "blocked": float(row["blocked_share"]),
                    "uplift_pct": 100.0 * row["pnl_improvement"] / base,
                    "pnl_before": float(row["pnl_ungated"]),
                    "pnl_after": float(row["pnl_gated"])}

    if holdout is not None:
        auc, skill = holdout["auc"], holdout["brier_skill"]
    else:
        auc = float(folds["auc"].mean()) if "auc" in folds else np.nan
        skill = float(folds["brier_skill"].mean()) if "brier_skill" in folds else np.nan

    worst = None
    if "counterparty_id" in fills and fills["counterparty_id"].nunique() > 1:
        col = f"markout_{int(label_horizon)}s"
        col = col if col in fills else f"markout_{int(long_h)}s"
        g = fills.groupby("counterparty_id")[col].agg(["mean", "size"])
        g = g[g["size"] >= 20].sort_values("mean")
        if len(g):
            worst = g.head(5)

    separates = bool(skill > 0 and monotonicity < -0.3)
    pays = best is not None and best["uplift_pct"] > 5
    bleeding = bool(net_edge < 0)

    if bleeding and (separates or pays):
        verdict, status = "LOSING TO TOXIC FLOW", "critical"
    elif bleeding:
        verdict, status = "LOSING MONEY (no toxic subset found)", "critical"
    elif separates and pays:
        verdict, status = "PROFITABLE, TOXIC SUBSET", "warning"
    elif separates or pays:
        verdict, status = "PROFITABLE, WEAK TOXIC SIGNAL", "warning"
    else:
        verdict, status = "BENIGN", "good"

    return {
        "symbol": symbol, "n_fills": len(fills), "n_ticks": len(ticks),
        "start": fills["timestamp"].min(), "end": fills["timestamp"].max(),
        "toxic_rate": float(labels[labels >= 0].mean()),
        "spread_earned_pips": _pips(spread_earned, pip),
        "adverse_pips": _pips(adverse, pip),
        "net_edge_pips": _pips(net_edge, pip),
        "short_horizon": float(short_h),
        "alpha_mu_pips": _pips(econ["alpha_mu"], pip),
        "alpha_mu_source": econ.get("alpha_mu_source", "fitted"),
        "auc": auc, "brier_skill": skill, "monotonicity": monotonicity,
        "gate": best, "worst_counterparties": worst,
        "verdict": verdict, "status": status,
        "separates": separates, "pays": pays,
        "features": cv_result.get("feature_names", []),
        "label_horizon": label_horizon, "keep_horizon": keep_h, "pip": pip,
        "auc_ci": holdout["auc_ci"] if holdout is not None else None,
        "problems": problems or [],}


def print_summary(s: dict) -> None:
    mark = {"good": "+", "warning": "!", "critical": "!!"}[s["status"]]
    width = 68
    print()
    print("=" * width)
    print(f"  {s['symbol']}   {s['start']:%Y-%m-%d %H:%M} -> {s['end']:%Y-%m-%d %H:%M}"
          f"   {s['n_fills']:,} fills")
    print("=" * width)
    print()
    print(f"  VERDICT   {mark} {s['verdict']}")
    print()
    adv = s["adverse_pips"]
    print(f"  {'markout @ %gs' % s['short_horizon']:<22}{s['spread_earned_pips']:+8.3f} pips/fill"
          f"   spread captured")
    print(f"  {'markout @ %gs' % s['keep_horizon']:<22}{s['net_edge_pips']:+8.3f} pips/fill"
          f"   what you keep")
    print(f"  {'':<22}{'-' * 13}")
    print(f"  {'adverse selection':<22}{adv:+8.3f} pips/fill"
          f"   {'eroded' if adv > 0 else 'none - markout improved'}")
    print()
    print(f"  {'toxic fills':<22}{s['toxic_rate']:7.1%}")
    if s["auc_ci"] is not None:
        lo, hi = s["auc_ci"]
        print(f"  {'model (holdout)':<22}AUC {s['auc']:.3f} [{lo:.3f}-{hi:.3f}]"
              f"   Brier skill {s['brier_skill']:+.3f}   monotonicity {s['monotonicity']:+.2f}")
    else:
        print(f"  {'model':<22}AUC {s['auc']:.3f}   Brier skill {s['brier_skill']:+.3f}"
              f"   monotonicity {s['monotonicity']:+.2f}")

    if s["gate"]:
        g = s["gate"]
        ci = (f" [{g['uplift_ci'][0]:+.0f}% to {g['uplift_ci'][1]:+.0f}%]"
              if "uplift_ci" in g else "")
        print(f"  {'gate':<22}block {g['blocked']:.0%} of fills at p>={g['threshold']:.2f}"
              f"  ->  PnL {g['uplift_pct']:+.0f}%{ci}")
    else:
        print(f"  {'gate':<22}no threshold improves PnL")
    for p in s["problems"]:
        print(f"  !! {p}")

    if s["worst_counterparties"] is not None:
        print()
        print("  worst counterparties (mean markout, pips)")
        for cp, row in s["worst_counterparties"].iterrows():
            print(f"    {cp:<12}{_pips(row['mean'], s['pip']):+7.3f}   "
                  f"{int(row['size']):,} fills")

    if s["features"]:
        print()
        print(f"  predictive features: {', '.join(s['features'])}")
    print()


def charts(*, path, summary, fills, buckets, deciles, sweep, fit, horizons,
           oof_proba=None) -> str:

    pip = summary["pip"]
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK_MUTED,
        "xtick.color": INK_MUTED, "ytick.color": INK_MUTED,
        "text.color": INK, "font.size": 9,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,})

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.6))
    for ax in axes.ravel():
        ax.set_axisbelow(True)

    fig.text(0.013, 0.963, "\u25a0", fontsize=15, color=STATUS[summary["status"]],
             va="center")
    fig.text(0.035, 0.963,
             f"{summary['symbol']}   {summary['verdict']}", fontsize=13,
             fontweight="bold", color=INK, va="center")
    fig.text(0.035, 0.928,
             f"net edge {summary['net_edge_pips']:+.3f} pips/fill   \u00b7   "
             f"{summary['n_fills']:,} fills   \u00b7   "
             f"{summary['toxic_rate']:.0%} labelled toxic",
             fontsize=9, color=INK_MUTED, va="center")
    
    ax = axes[0, 0]
    taus = np.asarray(fit["taus"], dtype=float)
    emp = np.asarray(fit["empirical_means"], dtype=float) / pip
    ax.axhline(0, color=AXIS, lw=1)
    ax.plot(taus, emp, marker="o", ms=5, lw=2, color=SERIES_1, label="realized")
    if fit.get("identified") and fit.get("converged"):
        grid = np.geomspace(max(taus.min(), 0.5), taus.max(), 100)
        ax.plot(grid, fit["curve"](grid) / pip, lw=1.5, ls="--",
                color=SERIES_2, label="fitted")
        ax.legend(frameon=False, fontsize=8, loc="best")
    ax.set_xscale("log")
    ax.set_xlabel("seconds after fill")
    ax.set_ylabel("LP markout (pips)")
    ax.set_title("Markout curve", loc="left", fontweight="bold", color=INK, pad=8)
    ax.margins(x=0.12, y=0.22)  # room for the endpoint labels
    for i, ha, dx, dy in ((0, "left", 6, 10), (len(emp) - 1, "right", -8, 12)):
        ax.annotate(f"{emp[i]:+.3f}", (taus[i], emp[i]), textcoords="offset points",
                    xytext=(dx, dy), ha=ha, fontsize=8, color=INK_MUTED)

    ax = axes[0, 1]
    if deciles is not None and len(deciles):
        vals = deciles["mean_pnl"].to_numpy() / pip
        xs = np.arange(len(vals))
        cols = [POS if v >= 0 else NEG for v in vals]
        ax.bar(xs, vals, color=cols, width=0.72)
        ax.axhline(0, color=AXIS, lw=1)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{p:.3f}" for p in deciles["mean_proba"]],
                           rotation=45, ha="right", fontsize=7)
        ax.set_xlabel("predicted toxicity (bin mean)")
        ax.set_ylabel("realized markout (pips)")
        ax.margins(y=0.18)
    ax.set_title(f"PnL by predicted toxicity   \u00b7   monotonicity "
                 f"{summary['monotonicity']:+.2f}", loc="left",
                 fontweight="bold", color=INK, pad=8)

    ax = axes[1, 0]
    if sweep is not None and len(sweep):
        up = 100.0 * sweep["pnl_improvement"] / (abs(sweep["pnl_ungated"]).replace(0, 1))
        ax.axhline(0, color=AXIS, lw=1)
        ax.plot(sweep["threshold"], up, marker="o", ms=5, lw=2, color=SERIES_1)
        if summary["gate"]:
            g = summary["gate"]
            ax.plot([g["threshold"]], [g["uplift_pct"]], marker="o", ms=12,
                    mfc="none", mec=STATUS["good"], mew=2, zorder=5)
            ax.annotate(f"chosen: block {g['blocked']:.0%} at p\u2265{g['threshold']:.2f}"
                        f"\nPnL {g['uplift_pct']:+.0f}%",
                        xy=(0.97, 0.93), xycoords="axes fraction",
                        ha="right", va="top", fontsize=8.5, color=INK,
                        bbox=dict(boxstyle="round,pad=0.4", fc=SURFACE,
                                  ec=GRID, lw=0.8))
        ax.set_xlabel("gate threshold (block fills above)")
        ax.set_ylabel("PnL change (%)")
        ax.margins(y=0.2)
    ax.set_title("Gate value", loc="left", fontweight="bold", color=INK, pad=8)

    ax = axes[1, 1]
    worst = summary["worst_counterparties"]
    if worst is not None and len(worst):
        names = list(worst.index)[::-1]
        vals = (worst["mean"].to_numpy() / pip)[::-1]
        counts = (worst["size"].to_numpy())[::-1]
        ys = np.arange(len(vals))
        ax.barh(ys, vals, color=[POS if v >= 0 else NEG for v in vals], height=0.66)
        ax.axvline(0, color=AXIS, lw=1)
        ax.set_yticks(ys)
        ax.set_yticklabels(names, fontsize=8.5)
        ax.set_xlabel("mean markout (pips)")
        ax.grid(axis="y", visible=False)
        ax.margins(x=0.30)
        for y, v, c in zip(ys, vals, counts):
            ax.annotate(f"{v:+.3f}  ({int(c):,} fills)", (0, y),
                        textcoords="offset points",
                        xytext=(6 if v < 0 else -6, 0), va="center",
                        ha="left" if v < 0 else "right",
                        fontsize=8, color=INK_MUTED)
        ax.set_title("Worst counterparties", loc="left", fontweight="bold",
                     color=INK, pad=8)
    else:
        ax.text(0.5, 0.5, "no counterparty ids in the blotter\n"
                          "per-counterparty toxicity unavailable",
                ha="center", va="center", color=INK_MUTED, fontsize=9)
        ax.set_axis_off()
        ax.set_title("Worst counterparties", loc="left", fontweight="bold",
                     color=INK, pad=8)

    fig.tight_layout(rect=(0, 0, 1, 0.905))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return str(path)
