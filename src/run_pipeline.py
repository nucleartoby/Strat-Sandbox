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
from features import (FEATURE_COLUMNS, build_feature_matrix, counterparty_table,
                      feature_health, select_matrix)
from integration import (gate_summary, meta_label_gate, required_spread, simulate_gate)
from labeling import drop_undecidable, label_fills, label_report, suggest_theta
from markouts import compute_markouts, fit_markout_curve, markout_summary, spread_vs_adverse_selection
from evaluation import (choose_threshold, gate_scores, holdout_scores, holdout_split,
                        hour_blocks)
from model_export import export_model, verify_export
from modeling import (cross_validate, evaluate, feature_importance, forward_select,
                      get_model, reliability_curve, summarize, univariate_scores)
from monitoring import (monotonicity_score, pnl_by_toxicity_decile, should_retrain,
                        sweep_gate_thresholds)
from report import charts as render_charts, build_summary, print_summary
from validation import validation_report
from vpin import compute_vpin, suggest_bucket_volume

HORIZONS = [1.0, 5.0, 30.0, 300.0, 1800.0]
GATE_THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.05), 2)


def _span(ts: pd.Series) -> str:
    return f"{ts.min():%Y-%m-%d %H:%M} -> {ts.max():%Y-%m-%d %H:%M}"


def _nearest_markout_column(horizons, target):
    nearest = min(horizons, key=lambda h: abs(h - target))
    return f"markout_{int(nearest)}s" if float(nearest).is_integer() else f"markout_{nearest}s"


VERBOSE = False


def say(*a, **k):
    if VERBOSE:
        print(*a, **k)


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
    ap.add_argument("--holdout", type=float, default=0.25,
                    help="fraction of fills, the latest in time, held out from all "
                         "selection and tuning and scored once at the end [0.25]")
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
    feats = build_feature_matrix(fills, ticks, buckets, labels=labels,
                                 label_delay_sec=label_horizon)
    X, y, fills_used = select_matrix(feats, labels, return_rows=True)
    if len(X) == 0:
        print("!! no usable rows: every fill has a missing feature. Check the "
              "feature-health table below for a column that is entirely NaN.")
        say(feature_health(feats).to_string(index=False))
        return 1
    say(f"feature matrix {X.shape}, toxic rate {y.mean():.1%}")
    say("\nfeature health:")
    say(feature_health(feats).to_string(index=False))

    banner("Phase 6 - model selection on the development period")
    event_times = fills_used["timestamp"]
    dev, test = holdout_split(event_times, label_horizon, test_frac=args.holdout)
    say(f"development {len(dev):,} fills ({_span(event_times.iloc[dev])})")
    say(f"holdout     {len(test):,} fills ({_span(event_times.iloc[test])}), "
        f"untouched until Phase 7")
    Xd, yd = X.iloc[dev].reset_index(drop=True), y.iloc[dev].reset_index(drop=True)
    dev_times = event_times.iloc[dev].reset_index(drop=True)
    fills_dev = fills_used.iloc[dev].reset_index(drop=True)
    for name, splits in (
        ("purged k-fold", list(purged_kfold_splits(dev_times, label_horizon,
                                                   n_splits=5, embargo_frac=0.01))),
        ("walk-forward", list(walk_forward_splits(dev_times, label_horizon,
                                                  n_splits=4))),):
        assert_no_leakage(splits, dev_times, label_horizon)
        say(f"\n{name} ({len(splits)} folds, leakage check passed):")
        say(split_report(splits, dev_times, yd)[
            ["fold", "n_train", "n_test", "test_toxic_rate"]].to_string(index=False))

    splits = list(walk_forward_splits(dev_times, label_horizon, n_splits=4))

    say("\nunivariate feature scores (top 6):")
    say(univariate_scores(Xd, yd, splits).head(6).to_string(index=False))

    cols = list(X.columns)
    if args.select_features:
        say("\nforward selection:")
        sel = forward_select(Xd, yd, splits, verbose=VERBOSE)
        if sel["selected"]:
            say(f"selected {len(sel['selected'])}/{X.shape[1]}: {sel['selected']}")
            say(f"dropped: {sel['dropped']}")
            cols = sel["selected"]
        else:
            print("NO feature beats simply predicting the base rate.")
            say("  Most often this means the blotter has no counterparty ids, so the "
                  "one feature\n  that identifies who is picking you off is unavailable. "
                  "VPIN and the markout\n  curve above are still valid; the classifier "
                  "is not. Keeping all features so\n  the metrics below show it plainly.")

    results = {}
    for kind in dict.fromkeys(["logreg", args.model]):
        res = cross_validate(Xd[cols], yd, splits, kind=kind)
        results[kind] = res
        say(f"\n--- {kind} ---")
        say(res["fold_results"][
            ["fold", "n_train", "n", "auc", "brier", "brier_skill"]].to_string(index=False))
        say(summarize(res).to_string(index=False))

    best = args.model if args.model in results else "logreg"
    res = results[best]
    say(f"\nselected: {best}")
    say("\nreliability (development out-of-fold):")
    oof = res["oof_proba"]
    scored = oof.notna().to_numpy()
    say(reliability_curve(yd[scored], oof[scored]).to_string(index=False))
    say("\ntop features:")
    say(feature_importance(res).head(8).to_string(index=False))

    pnl_col = _nearest_markout_column(horizons, label_horizon)
    dev_sweep = sweep_gate_thresholds(fills_dev.loc[scored], oof[scored].to_numpy(),
                                      thresholds=GATE_THRESHOLDS, pnl_column=pnl_col)
    threshold = choose_threshold(dev_sweep)
    say("\ngate threshold (chosen on development out-of-fold predictions): "
        + (f"block at p>={threshold:.2f}" if threshold is not None
           else "none improves PnL"))

    banner("Phase 7 - untouched holdout")
    model = res["model"]  # fit on the whole development period
    fills_test = fills_used.iloc[test].reset_index(drop=True)
    y_test = y.iloc[test].to_numpy()
    p_test = model.predict_proba(X.iloc[test][cols])[:, 1]
    blocks = hour_blocks(fills_test["timestamp"])
    ho = holdout_scores(y_test, p_test, blocks)
    fit_auc = evaluate(yd, model.predict_proba(Xd[cols])[:, 1])["auc"]
    oof_auc = res["oof_metrics"].get("auc", np.nan)
    say("AUC by stage (a large drop from in-sample to holdout means overfitting):")
    say(f"  in-sample (development fit)  {fit_auc:.3f}")
    say(f"  development out-of-fold      {oof_auc:.3f}")
    say(f"  holdout                      {ho['auc']:.3f}  "
        f"95% CI [{ho['auc_ci'][0]:.3f}, {ho['auc_ci'][1]:.3f}]")
    say(f"holdout Brier skill {ho['brier_skill']:+.4f}  "
        f"95% CI [{ho['brier_skill_ci'][0]:+.4f}, {ho['brier_skill_ci'][1]:+.4f}]")

    baselines = [("base rate (development)", np.full(len(y_test), yd.mean()))]
    if "cp_toxic_rate_hist" in X:
        baselines.append(("counterparty history alone",
                          X.iloc[test]["cp_toxic_rate_hist"].to_numpy()))
    if best != "logreg":
        baselines.append(("logreg, same features",
                          results["logreg"]["model"].predict_proba(X.iloc[test][cols])[:, 1]))
    baselines.append((f"{best} (selected)", p_test))
    say("\nholdout vs baselines:")
    say(pd.DataFrame([{"model": name, **{k: evaluate(y_test, p)[k]
                                         for k in ("auc", "brier_skill", "log_loss")}}
                      for name, p in baselines]).to_string(index=False))

    rng = np.random.default_rng(0)
    labels_perm = pd.Series(rng.permutation(labels.to_numpy()), index=labels.index)
    Xp, yp = select_matrix(build_feature_matrix(fills, ticks, buckets, labels=labels_perm,
                                                label_delay_sec=label_horizon), labels_perm)
    perm_model = get_model(best).fit(Xp.iloc[dev][cols], yp.iloc[dev])
    shuffle_auc = evaluate(yp.iloc[test].to_numpy(),
                           perm_model.predict_proba(Xp.iloc[test][cols])[:, 1])["auc"]
    say(f"\nshuffled-label check: holdout AUC {shuffle_auc:.3f} (want ~0.5)")

    say("\nreliability (holdout):")
    say(reliability_curve(y_test, p_test).to_string(index=False))

    gate = None
    if threshold is not None:
        gate = gate_scores(fills_test[pnl_col].to_numpy(), p_test, threshold, blocks)
        say(f"\nholdout gate at p>={threshold:.2f}: blocks {gate['blocked_share']:.1%} "
            f"of fills, PnL {gate['uplift']:+.1%} "
            f"(95% CI [{gate['uplift_ci'][0]:+.1%}, {gate['uplift_ci'][1]:+.1%}])")
        say(f"  mean markout of blocked fills {gate['mean_pnl_blocked'] / args.pip:+.3f} pips, "
            f"kept {gate['mean_pnl_kept'] / args.pip:+.3f} pips")
        say("  same gate judged at every markout horizon:")
        for h in horizons:
            c = _nearest_markout_column(horizons, h)
            g = gate_scores(fills_test[c].to_numpy(), p_test, threshold, blocks)
            say(f"    {c:<16}PnL {g['uplift']:+7.1%}  "
                f"95% CI [{g['uplift_ci'][0]:+.1%}, {g['uplift_ci'][1]:+.1%}]")

    problems = []
    if not ho["auc_ci"][0] > 0.5:
        problems.append("holdout AUC is not distinguishable from chance")
    if fit_auc - ho["auc"] > 0.05:
        problems.append(f"in-sample AUC exceeds holdout by {fit_auc - ho['auc']:.3f}: overfitting")
    if shuffle_auc > 0.55:
        problems.append(f"shuffled labels still score AUC {shuffle_auc:.3f}: leakage")
    if gate is not None and not gate["uplift_ci"][0] > 0:
        problems.append("the gate's holdout PnL gain is not significant")
    for p in problems:
        say(f"!! {p}")
    if not problems:
        say("\nno overfitting or leakage detected on the holdout")

    banner("Phase 8 - integration and export")
    p_test_s = pd.Series(p_test)
    meta = meta_label_gate(pd.Series(1, index=p_test_s.index), p_test_s, act_threshold=0.5)
    say(f"meta-label gate (holdout): acts on {meta['act'].mean():.1%} of fills, "
          f"mean size multiplier {meta['size_multiplier'].mean():.2f}")
    say(f"required spread at p=0.8: {required_spread(0.8, fit['alpha_mu']):.3e} "
          f"(alpha_mu-priced)")

    # One gate per counterparty
    gate_cfg = (dict(widen_above=threshold, suspend_above=threshold, tighten_below=0.0)
                if threshold is not None else {})
    groups = (fills_test.groupby("counterparty_id").indices.values()
              if "counterparty_id" in fills_test else [np.arange(len(fills_test))])
    sim = pd.concat([simulate_gate(fills_test["timestamp"].iloc[idx], p_test[idx],
                                   fills_test["spread"].to_numpy()[idx],
                                   alpha_mu=fit["alpha_mu"], **gate_cfg)
                     for idx in groups], ignore_index=True)
    summary = gate_summary(sim)
    say("\nlive gate simulation (holdout, one gate per counterparty):")
    say(summary[["action", "n", "share", "mean_proba", "mean_multiplier"]].to_string(index=False))
    say(f"state changes: {summary.attrs['state_changes']} "
          f"({summary.attrs['changes_per_1k']:.1f} per 1k fills)")

    engine_gate = (f"--widen-above {threshold:.2f} --suspend-above {threshold:.2f} "
                   f"--tighten-below 0" if threshold is not None else "")
    if args.export:
        final = get_model(best).fit(X[cols], y)
        info = export_model(final, args.export, feature_names=cols)
        check = verify_export(final, args.export, X[cols].to_numpy())
        say(f"\nexported {info['kind']} (refit on all {len(X):,} fills) to {args.export} "
              f"({os.path.getsize(args.export):,} bytes)")
        say(f"python vs C++ inference: max|diff| = {check['max_abs_diff']:.3e} "
              f"-> {'MATCH' if check['matches'] else 'MISMATCH'}")
        if not check["matches"]:
            print("!! export does not reproduce the Python model; do not deploy it")
            return 1
        cp_path = os.path.splitext(args.export)[0] + ".cp.csv"
        cp_table = counterparty_table(fills, labels)
        cp_table.to_csv(cp_path, index=False)
        say(f"counterparty history ({len(cp_table)} counterparties) -> {cp_path}")
        say("run the engine for one counterparty (code column in that file):")
        say(f"  ./build/vpin_realtime --bucket-volume {bucket_volume:.0f} --dist t "
            f"--model {args.export} --cp-history {cp_path} --counterparty CODE "
            f"{engine_gate} < ticks.csv")
    else:
        say("\npass --export PATH.fxm to write a model for the realtime engine")

    banner("Phase 9 - monitoring (holdout)")
    deciles = pnl_by_toxicity_decile(fills_test, p_test, pnl_column=pnl_col)
    say("realized markout by predicted-toxicity decile:")
    say(deciles[["decile", "n", "mean_proba", "mean_pnl"]].to_string(index=False))
    mono = monotonicity_score(deciles)
    say(f"\nmonotonicity (want close to -1): {mono:+.3f}")

    say("\ngate value by threshold (for reference; the threshold was fixed in Phase 6):")
    sweep = sweep_gate_thresholds(fills_test, p_test, thresholds=GATE_THRESHOLDS,
                                  pnl_column=pnl_col)
    say(sweep[["threshold", "blocked_share", "pnl_ungated", "pnl_gated",
                 "pnl_improvement"]].to_string(index=False))

    trigger = should_retrain(pd.DataFrame([ho]))
    say(f"\nretrain trigger: {trigger['retrain']} ({trigger['reason']})")

    banner("done")

    summary = build_summary(
        symbol=args.symbol or os.path.basename(args.ticks or "synthetic"),
        ticks=ticks, fills=fills_used, labels=y, fit=fit, econ=econ,
        cv_result=res, deciles=deciles, sweep=sweep,
        monotonicity=mono, horizons=horizons, label_horizon=label_horizon,
        pip=args.pip, holdout=ho, gate=gate, problems=problems)
    print_summary(summary)

    if args.charts:
        path = render_charts(path=args.charts, summary=summary, fills=fills_used,
                             buckets=buckets, deciles=deciles, sweep=sweep,
                             fit=fit, horizons=horizons, oof_proba=p_test_s)
        print(f"  charts -> {path}")
    if not VERBOSE:
        print("  full diagnostics: re-run with -v")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
