import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import native
from cv import assert_no_leakage, purged_kfold_splits, split_report, walk_forward_splits
from data_cleaning import clean_ticks, coverage_report
from data_ingestion import (generate_synthetic_ticks, generate_synthetic_trades,
                            load_tick_data, load_trade_blotter)
from features import FEATURE_COLUMNS, build_feature_matrix, feature_health, select_matrix
from integration import (gate_summary, meta_label_gate, required_spread, simulate_gate)
from labeling import drop_undecidable, label_fills, label_report, suggest_theta
from markouts import compute_markouts, fit_markout_curve, markout_summary, spread_vs_adverse_selection
from model_export import export_model, verify_export
from modeling import (cross_validate, feature_importance, forward_select,
                      reliability_curve, summarize, univariate_scores)
from monitoring import (monotonicity_score, pnl_by_toxicity_decile, should_retrain,
                        sweep_gate_thresholds)
from report import charts as render_charts, build_summary, print_summary
from validation import validation_report
from vpin import compute_vpin, suggest_bucket_volume

HORIZONS = [1.0, 5.0, 30.0, 300.0, 1800.0]


def _nearest_markout_column(horizons, target):
    nearest = min(horizons, key=lambda h: abs(h - target))
    return f"markout_{int(nearest)}s" if float(nearest).is_integer() else f"markout_{nearest}s"


VERBOSE = False


def say(*a, **k):
    if VERBOSE:
        say(*a, **k)


def banner(text: str) -> None:
    say(f"\n{'=' * 78}\n{text}\n{'=' * 78}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ticks", help="tick CSV; omit to use synthetic data")
    ap.add_argument("--symbol", help="fetch ticks from Dukascopy, e.g. EURUSD "
                                     "(requires --start/--end)")
    ap.add_argument("--start", help="fetch start date, YYYY-MM-DD")
    ap.add_argument("--end", help="fetch end date, YYYY-MM-DD (inclusive)")
    ap.add_argument("--provider", default="auto",
                    choices=["auto", "dukascopy", "duka", "truefx", "histdata", "generic"])
    ap.add_argument("--blotter", help="trade blotter CSV; omit to synthesise fills")
    ap.add_argument("--n-ticks", type=int, default=150_000, help="synthetic tick count")
    ap.add_argument("--n-trades", type=int, default=6_000, help="synthetic fill count")
    ap.add_argument("--buckets-per-day", type=int, default=50)
    ap.add_argument("--vpin-window", type=int, default=50)
    ap.add_argument("--model", default="gbm", choices=["logreg", "gbm"])
    ap.add_argument("--vol-multiple", type=float, default=1.0,
                    help="barrier width as a multiple of the horizon's median move")
    ap.add_argument("--label-rule", default="triple_barrier",
                    choices=["triple_barrier", "crossback", "markout_threshold"],
                    help="how a fill is judged toxic. triple_barrier suits large "
                         "directional informed flow; markout_threshold suits small "
                         "persistent bleed such as latency arbitrage, where the "
                         "adverse move never reaches a volatility-scaled barrier "
                         "[triple_barrier]")
    ap.add_argument("--label-threshold", type=float, default=0.0,
                    help="markout_threshold: a fill is toxic when its markout is "
                         "worse than -this [0.0]")
    ap.add_argument("--label-horizon", type=float, default=300.0,
                    help="seconds a fill is followed to decide if it was toxic. "
                         "Match this to where the markout curve actually bends "
                         "(Phase 3 prints the half-life); too short and the "
                         "adverse move has not happened yet [300]")
    ap.add_argument("--horizons", type=float, nargs="+", default=HORIZONS,
                    help="markout horizons in seconds")
    ap.add_argument("--export", help="write the trained model to this .fxm path")
    ap.add_argument("--simulate", choices=["lp", "ma", "random"],
                    help="generate a blotter from the ticks instead of loading "
                         "one, so a whole run is a single command")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="print every phase's diagnostics; default is a summary")
    ap.add_argument("--charts", default="toxicity_report.png",
                    help="where to write the chart panel ('' to skip)")
    ap.add_argument("--pip", type=float, default=1e-4,
                    help="pip size for reporting; 1e-2 for JPY crosses [1e-4]")
    ap.add_argument("--skip-validation", action="store_true")
    ap.add_argument("--no-select-features", dest="select_features",
                    action="store_false",
                    help="train on every feature instead of forward-selecting")
    args = ap.parse_args(argv)

    global VERBOSE
    VERBOSE = args.verbose
    horizons = list(args.horizons)
    label_horizon = args.label_horizon

    if not args.blotter and not args.simulate:
        ap.error("need --blotter FILE, or --simulate lp to generate one")
    for label, path in (("--blotter", args.blotter), ("--ticks", args.ticks)):
        if path and not os.path.exists(path):
            ap.error(f"{label}: no such file: {path}\n"
                     f"  A blotter is your own fills, as "
                     f"timestamp,side,price,size[,counterparty_id].\n"
                     f"  Generate one from tick data with:  "
                     f"python src/make_blotter.py --help")
    if args.export:
        out_dir = os.path.dirname(os.path.abspath(args.export))
        if not os.path.isdir(out_dir):
            ap.error(f"--export: directory does not exist: {out_dir}")

    say(f"backend: {native.backend()}")

    banner("Phase 2 - acquire and prepare data")
    if args.symbol:
        if not (args.start and args.end):
            ap.error("--symbol needs --start and --end (YYYY-MM-DD)")
        from data_fetch import fetch_ticks
        say(f"fetching {args.symbol} {args.start}..{args.end} from Dukascopy "
              f"(cached under ~/.cache/fxtox)")
        raw = fetch_ticks(args.symbol, args.start, args.end)
        say(f"fetched {len(raw):,} ticks")
    elif args.ticks:
        raw = load_tick_data(args.ticks, provider=args.provider)
        say(f"loaded {len(raw):,} ticks from {args.ticks}")
        if raw.attrs.get("volume_is_proxy"):
            warnings.warn(
                "this source publishes no volume, so a proxy is in use. VPIN is "
                "defined in volume time; buckets built on a proxy measure "
                "something weaker than true volume imbalance.", RuntimeWarning)
    else:
        raw = generate_synthetic_ticks(n=args.n_ticks)
        say(f"generated {len(raw):,} synthetic ticks")

    ticks = clean_ticks(raw, verbose=VERBOSE)
    say(coverage_report(ticks).to_string(index=False))

    if args.blotter:
        fills = load_trade_blotter(args.blotter)
        say(f"loaded {len(fills):,} fills from {args.blotter}")
    elif args.simulate:
        from make_blotter import lp_blotter, ma_blotter, random_blotter
        builder = {"lp": lambda: lp_blotter(ticks),
                   "ma": lambda: ma_blotter(ticks),
                   "random": lambda: random_blotter(ticks, n_trades=args.n_trades)}
        fills = builder[args.simulate]().sort_values("timestamp").reset_index(drop=True)
        say(f"simulated {len(fills):,} fills (mode={args.simulate})")
    else:
        fills = generate_synthetic_trades(ticks, n_trades=args.n_trades)
        say(f"generated {len(fills):,} synthetic fills "
            f"({fills['is_informed'].mean():.0%} from informed counterparties)")

    banner("Phase 3 - core toxicity metrics")
    bucket_volume = suggest_bucket_volume(ticks, buckets_per_day=args.buckets_per_day,
                                          vpin_window=args.vpin_window)
    buckets = compute_vpin(ticks, bucket_volume, window=args.vpin_window,
                           dist="student_t", df=0.25)
    n_readings = int(buckets["vpin"].notna().sum())
    say(f"bucket volume {bucket_volume:,.0f} -> {len(buckets):,} buckets, "
          f"{n_readings:,} VPIN readings, mean VPIN {buckets['vpin'].mean():.3f}")
    if n_readings < 10:
        print("!! too few VPIN readings to be useful; lower --vpin-window or "
              "supply more data")
        return 1

    if not args.skip_validation:
        say()
        rep = validation_report(ticks, bucket_volume, window=args.vpin_window,
                                verbose=VERBOSE)
        if not rep["all_properties_passed"]:
            print("\n!! VPIN validation failed; stopping rather than building on it")
            return 1

    fills = compute_markouts(fills, ticks, horizons)
    say("\nmarkout profile (LP perspective):")
    say(markout_summary(fills, horizons).to_string(index=False))

    fit = fit_markout_curve(fills, horizons)
    say(f"\nfitted curve: half_spread={fit['half_spread']:.3e}  "
          f"alpha_mu={fit['alpha_mu']:.3e} (+/- {fit['stderr']['alpha_mu']:.1e})  "
          f"lambda={fit['lambda']:.4f}  half-life={fit['half_life_seconds']:.1f}s")
    if fit.get("identified") and np.isfinite(fit["half_life_seconds"]):
        hl = fit["half_life_seconds"]
        if label_horizon < hl / 2 or label_horizon > hl * 20:
            print(f"\n!! --label-horizon is {label_horizon:.0f}s but adverse selection "
                  f"has a {hl:.0f}s half-life.\n   Too short and the adverse move has "
                  f"not happened yet when the label is decided; far too long and the "
                  f"label\n   is dominated by unrelated drift. Consider "
                  f"--label-horizon {max(hl, 30):.0f}.")

    econ = spread_vs_adverse_selection(fit, fills)
    say(f"breakeven spread {econ['breakeven_spread']:.3e} vs realized half-spread "
          f"{econ['realized_half_spread']:.3e} -> "
          f"{'profitable' if econ['profitable'] else 'LOSS-MAKING at current spreads'}")

    banner("Phase 4 - ground-truth labels")
    if args.label_rule == "markout_threshold":
        col = _nearest_markout_column(horizons, label_horizon)
        say(f"rule=markout_threshold on {col}, toxic when markout < "
              f"-{args.label_threshold:.3e}")
        labels = label_fills(fills, ticks, rule="markout_threshold",
                             horizon_seconds=label_horizon,
                             theta=args.label_threshold, markout_column=col)
    else:
        cal = suggest_theta(fills, ticks, horizon_seconds=label_horizon,
                            vol_multiple=args.vol_multiple)
        say(f"rule={args.label_rule}, theta={cal['theta']:.3e} "
              f"({args.vol_multiple}x the {label_horizon:.0f}s median move of "
              f"{cal['horizon_vol']:.3e}) -> toxic rate {cal['toxic_rate']:.1%}")
        labels = label_fills(fills, ticks, rule=args.label_rule,
                             horizon_seconds=label_horizon, theta=cal["theta"],
                             min_adverse=cal["theta"])
    say(label_report(labels).to_string(index=False))

    if "is_informed" in fills:
        by = label_report(labels, fills, by="is_informed")
        say("\nlabel rate by planted ground truth:")
        say(by[["is_informed", "n", "toxic_rate"]].to_string(index=False))

    fills, labels = drop_undecidable(fills, labels)
    say(f"\n{len(fills):,} fills with resolved labels")

    banner("Phase 5 - pre-trade features")
    feats = build_feature_matrix(fills, ticks, buckets, labels=labels)
    X, y, fills_used = select_matrix(feats, labels, return_rows=True)
    if len(X) == 0:
        print("!! no usable rows: every fill has a missing feature. Check the "
              "feature-health table below for a column that is entirely NaN.")
        say(feature_health(feats).to_string(index=False))
        return 1
    say(f"feature matrix {X.shape}, toxic rate {y.mean():.1%}")
    say("\nfeature health:")
    say(feature_health(feats).to_string(index=False))

    banner("Phase 6 - walk-forward and purged cross-validation")
    event_times = fills_used["timestamp"]
    for name, splits in (
        ("purged k-fold", list(purged_kfold_splits(event_times, label_horizon,
                                                   n_splits=5, embargo_frac=0.01))),
        ("walk-forward", list(walk_forward_splits(event_times, label_horizon,
                                                  n_splits=4))),):
        assert_no_leakage(splits, event_times, label_horizon)
        say(f"\n{name} ({len(splits)} folds, leakage check passed):")
        say(split_report(splits, event_times, y)[
            ["fold", "n_train", "n_test", "test_toxic_rate"]].to_string(index=False))

    splits = list(walk_forward_splits(event_times, label_horizon, n_splits=4))

    say("\nunivariate feature scores (top 6):")
    say(univariate_scores(X, y, splits).head(6).to_string(index=False))

    if args.select_features:
        say("\nforward selection:")
        sel = forward_select(X, y, splits, verbose=VERBOSE)
        if sel["selected"]:
            say(f"selected {len(sel['selected'])}/{X.shape[1]}: {sel['selected']}")
            say(f"dropped: {sel['dropped']}")
            X = X[sel["selected"]]
        else:
            print("NO feature beats simply predicting the base rate.")
            say("  Most often this means the blotter has no counterparty ids, so the "
                  "one feature\n  that identifies who is picking you off is unavailable. "
                  "VPIN and the markout\n  curve above are still valid; the classifier "
                  "is not. Keeping all features so\n  the metrics below show it plainly.")

    results = {}
    for kind in dict.fromkeys(["logreg", args.model]):
        res = cross_validate(X, y, splits, kind=kind)
        results[kind] = res
        say(f"\n--- {kind} ---")
        say(res["fold_results"][
            ["fold", "n_train", "n", "auc", "brier", "brier_skill"]].to_string(index=False))
        say(summarize(res).to_string(index=False))

    best = args.model if args.model in results else "logreg"
    res = results[best]
    say(f"\nselected: {best}")
    say("\nreliability (out-of-fold):")
    oof = res["oof_proba"]
    mask = oof.notna()
    say(reliability_curve(y[mask], oof[mask]).to_string(index=False))
    say("\ntop features:")
    say(feature_importance(res).head(8).to_string(index=False))

    banner("Phase 7 - integration")
    proba = oof.fillna(oof.mean())
    meta = meta_label_gate(pd.Series(1, index=X.index), proba, act_threshold=0.5)
    say(f"meta-label gate: acts on {meta['act'].mean():.1%} of fills, "
          f"mean size multiplier {meta['size_multiplier'].mean():.2f}")
    say(f"required spread at p=0.8: {required_spread(0.8, fit['alpha_mu']):.3e} "
          f"(alpha_mu-priced)")

    sim = simulate_gate(fills_used["timestamp"], proba,
                        fills_used["spread"].to_numpy(), alpha_mu=fit["alpha_mu"])
    summary = gate_summary(sim)
    say("\nlive gate simulation:")
    say(summary[["action", "n", "share", "mean_proba", "mean_multiplier"]].to_string(index=False))
    say(f"state changes: {summary.attrs['state_changes']} "
          f"({summary.attrs['changes_per_1k']:.1f} per 1k fills)")

    banner("Phase 8 - export for the C++ engine")
    if args.export:
        info = export_model(res["model"], args.export, feature_names=list(X.columns))
        check = verify_export(res["model"], args.export, X.to_numpy())
        say(f"exported {info['kind']} to {args.export} "
              f"({os.path.getsize(args.export):,} bytes)")
        say(f"python vs C++ inference: max|diff| = {check['max_abs_diff']:.3e} "
              f"-> {'MATCH' if check['matches'] else 'MISMATCH'}")
        if not check["matches"]:
            print("!! export does not reproduce the Python model; do not deploy it")
            return 1
    else:
        say("pass --export PATH.fxm to write a model for the realtime engine")
        say("then:  ./build/vpin_realtime --bucket-volume "
              f"{bucket_volume:.0f} --model PATH.fxm --alpha-mu {fit['alpha_mu']:.3e} "
              "< ticks.csv")

    banner("Phase 9 - monitoring")
    pnl_col = _nearest_markout_column(horizons, label_horizon)
    deciles = pnl_by_toxicity_decile(fills_used, proba, pnl_column=pnl_col)
    say("realized markout by predicted-toxicity decile:")
    say(deciles[["decile", "n", "mean_proba", "mean_pnl"]].to_string(index=False))
    mono = monotonicity_score(deciles)
    say(f"\nmonotonicity (want close to -1): {mono:+.3f}")

    say("\ngate value by threshold:")
    sweep = sweep_gate_thresholds(fills_used, proba, pnl_column=pnl_col)
    say(sweep[["threshold", "blocked_share", "pnl_ungated", "pnl_gated",
                 "pnl_improvement"]].to_string(index=False))

    trigger = should_retrain(res["fold_results"])
    say(f"\nretrain trigger: {trigger['retrain']} ({trigger['reason']})")

    banner("done")

    pnl_col = _nearest_markout_column(horizons, label_horizon)
    summary = build_summary(
        symbol=args.symbol or os.path.basename(args.ticks or "synthetic"),
        ticks=ticks, fills=fills_used, labels=y, fit=fit, econ=econ,
        cv_result=res, deciles=deciles, sweep=sweep,
        monotonicity=mono, horizons=horizons, label_horizon=label_horizon,
        pip=args.pip)
    print_summary(summary)

    if args.charts:
        path = render_charts(path=args.charts, summary=summary, fills=fills_used,
                             buckets=buckets, deciles=deciles, sweep=sweep,
                             fit=fit, horizons=horizons, oof_proba=proba)
        print(f"  charts -> {path}")
    if not VERBOSE:
        print("  full diagnostics: re-run with -v")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
