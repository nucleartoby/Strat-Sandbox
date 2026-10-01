# FX toxic-flow detection

A nine-phase pipeline for measuring and acting on adverse selection in FX market
making: VPIN from tick data, markout-based adverse-selection costs, supervised
toxicity labels, a calibrated classifier, and a quoting gate that runs the same
code in research and in production.

Prototyped in Python; the latency-critical path is C++17.

## Layout

```
include/fxtox/      header-only C++ core (the hot path)
  types.hpp         POD tick/fill/bucket types and zero-copy NumPy views
  math.hpp          normal & Student-t CDFs, Welford, drift-corrected rolling sum
  vpin.hpp          streaming volume bucketing, BVC, rolling VPIN, online quantiler
  markout.hpp       LP markouts against the prevailing mid
  search.hpp        block-skipping first-crossing queries over a price path
  labeling.hpp      crossback / triple-barrier / markout-threshold labels
  features.hpp      pre-trade features: one batch driver, one streaming driver
  model.hpp         .fxm loader + logistic / GBDT inference
  gate.hpp          quoting gate with hysteresis, dwell, and alpha_mu pricing
src/
  bindings.cpp      pybind11 module (fxtox_native)
  vpin_realtime.cpp the production engine binary
  native.py         dispatch: C++ if built, NumPy reference otherwise
  _reference.py     independent NumPy implementation (fallback + cross-check)
  data_fetch.py     Dukascopy downloader (duka's decoder, our transport)
  make_blotter.py   build a fill blotter from ticks (lp / ma / random)
  report.py         verdict summary + the four charts
  data_ingestion.py Dukascopy / duka / TrueFX / HistData loaders + synthetic data
  data_cleaning.py  crossed-quote removal, gap flagging, coverage report
  vpin.py           Phase 3 VPIN
  markouts.py       Phase 3 markouts and the adverse-selection curve fit
  labeling.py       Phase 4 labels and barrier selection
  features.py       Phase 5 feature assembly
  fracdiff.py       Phase 5 fractional differencing + ADF
  cv.py             Phase 6 purged k-fold and walk-forward, with embargo
  modeling.py       Phase 6 training, calibration, feature selection
  model_export.py   Phase 8 .fxm export
  integration.py    Phase 7 the three integration patterns
  monitoring.py     Phase 9 drift and gate-value tracking
  validation.py     Phase 3 validation + flowrisk cross-check
  run_pipeline.py   end-to-end orchestrator
tests/              C++ unit tests and the Python suite
bench/              C++ vs NumPy timings
```

## Setup

```bash
pip install -r requirements.txt
brew install libomp                       # macOS, for LightGBM
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

The extension lands in `src/`, so `import fxtox_native` works with no install
step. Without it everything still runs on the NumPy reference, just slower.

## Running it

```bash
# one command: fetch, simulate a book, run all nine phases, print a verdict
python src/run_pipeline.py --symbol EURUSD --start 2024-01-02 --end 2024-01-05 \
       --simulate lp --label-rule markout_threshold --label-horizon 30

python src/run_pipeline.py                            # synthetic, all 9 phases
python src/run_pipeline.py --symbol EURUSD --start 2024-01-02 --end 2024-01-05
python src/run_pipeline.py --ticks EURUSD.csv --provider dukascopy \
                           --blotter fills.csv --export model.fxm
./build/vpin_realtime --bucket-volume 3e6 --model model.fxm < ticks.csv
./build/vpin_realtime --bench 5000000 --bucket-volume 250000 --quiet
python bench/bench.py
```

Tests: `./build/fxtox_tests` and `pytest tests/`.

## Performance

500k ticks / 20k fills, Apple M-series, `-O3 -march=native`:

| stage | C++ | NumPy reference | speedup |
|---|---|---|---|
| VPIN (bucket + BVC + roll) | 1.5 ms | 51.5 ms | 34× |
| markouts (4 horizons) | 5.8 ms | 10.7 ms | 1.8× |
| labels (triple barrier) | 22.7 ms | 125.1 ms | 5.5× |
| features (13 columns) | 5.9 ms | 13.2 ms | 2.2× |
| **total** | **35.9 ms** | **200.5 ms** | **5.6×** |

The realtime engine sustains **~29 M ticks/s (34 ns/tick)** including feature
maintenance and gating. The reference is vectorised searchsorted/cumsum code,
not a strawman loop — VPIN is the outlier because its expanding sigma is
inherently sequential and cannot be vectorised at all.

## Design decisions worth knowing

**Everything is causal.** A full-sample sigma, a `rank(pct=True)` percentile, or
a scaler fitted outside the CV fold each leak the future into a backtest in a
way no metric will reveal. VPIN's sigma and percentile are both expanding, and
`tests/test_leakage.py` asserts that truncating the tick stream at the last fill
changes no feature.

**The live and research paths share code, not just intent.** `build_features`
and `LiveFeatures` call the same `fill_row`, with includes and evictions
interleaved identically — floating-point addition is not associative, so
matching the *set* of operations is not enough. The parity test asserts
bit-for-bit equality.

**VPIN's noise floor is 0.5, not 0** (in the default bucket granularity).
Under BVC a driftless walk has `dP ~ N(0, sigma)`, so the buy fraction
`Phi(dP/sigma)` is uniform and the imbalance `|2U-1|` has mean exactly ½. Raw
VPIN levels therefore say little on their own; the percentile is what a gate
should act on. Tick granularity has a different floor, near 0.06, because
summing many per-tick classifications averages toward `0.5·V` — so a threshold
tuned in one mode is meaningless in the other. Heavier-tailed BVC
(`dist="student_t"`) is *more* conservative, not less: `Phi_t(1) = 0.64` at
`df=0.25` against `0.84` for the normal.

**Barriers are centred on the mid, not the execution price.** A fill happens
half a spread from the mid, so centring a symmetric barrier on the execution
price puts the adverse side permanently closer. With a small `theta` that
asymmetry decides nearly every label: the toxic rate collapses to ~3% instead
of the ~50% coin flip a symmetric barrier should give, and it stops responding
monotonically to `theta`, which quietly breaks any attempt to calibrate it.

**The crossback rule has a direction trap.** A client buy fills at the *ask*, so
the mid already sits below the execution price at t=0. Asking whether it ever
gets there *from above* is trivially true and labels almost nothing toxic. The
rule fires when the market runs *through* the client's entry.

**Undecidable labels are dropped, not zeroed.** A fill whose horizon runs past
the end of the data has no label. Those are systematically the most recent
fills, so coercing them to benign teaches the model that recent flow is safe.

**The gate is priced off `alpha_mu`, not a widen factor.** The Phase 3 markout
fit already says what the flow costs. Required half-spread is
`alpha_mu * P(toxic)`, so the quoted spread is `2 * alpha_mu * p`. "Widen 2.5×"
is a guess; this is not.

**Models bind by feature name.** `.fxm` v2 stores feature names, and the engine
refuses a model whose inputs it cannot supply, naming the missing one.
Positional binding is a silent failure: the engine feeds spread into the slot
the model trained as realized vol and nothing downstream looks wrong.

**A converged fit is not an identified one.** When the markout profile is flat,
`alpha_mu` and `lambda` are jointly unidentifiable -- an arbitrarily large
`alpha_mu` with a near-zero `lambda` fits a flat line as well as anything else.
`curve_fit` returns without complaint, reporting a standard error larger than
the estimate and `lambda` pinned to its bound. That number then sets the
breakeven spread and, through it, the live gate's quoted spread: on one real run
a book earning 0.045 pips was told to quote 26. `fit_markout_curve` now reports
`identified`, and the economics fall back to the empirical decay.

**The labelling rule must match the character of the toxicity.** This is the
single easiest way to get a null result from data that has signal in it.
`triple_barrier` scales its barrier to the horizon's volatility, which suits
large directional informed flow. Latency-arbitrage bleed is small and fast --
roughly a half-spread, dissipating in tens of seconds -- so a volatility-scaled
300s barrier is an order of magnitude too wide to see it. Measured on a
simulated LP book over real EUR/USD:

| label | fast/slow toxic rate | counterparty-feature AUC |
|---|---|---|
| `triple_barrier` @300s, default θ | 0.430 / 0.424 | 0.497 (nothing) |
| `markout_threshold` @30s | 0.596 / 0.432 | 0.560 (oracle: 0.557) |

Use `--label-rule markout_threshold --label-horizon 30` for persistent bleed,
`triple_barrier` for directional flow. Check the markout curve's half-life
first; the pipeline warns when the horizon and the half-life disagree badly.

**Feature selection is part of the pipeline, not an afterthought.** Phase 5's
market-state features (spread, realized vol, tick rate) are identical for the
informed and uninformed fills happening at the same instant, so they often carry
nothing about whether a *given* fill is toxic. On the synthetic benchmark,
feeding all 19 features gives AUC 0.53 with *negative* Brier skill; forward
selection keeps `cp_toxic_rate_hist` alone and gets AUC 0.585 with positive
skill — against an oracle ceiling of 0.595. Selection is scored on Brier skill,
not AUC, because the output is consumed as a probability.

## Validation

`validation.py` runs two things. The **analytic properties** are the real check:
volume conservation, bounds, the 0.5 noise floor, one-sided flow driving VPIN to
~0.99, and invariance to how the same volume is chopped into ticks. These are
exact and derivable.

The **flowrisk cross-check** agrees closely, once the comparison is
like-for-like.

`flowrisk.BulkVPIN` applies BVC at **tick** granularity — each tick's own price
change, accumulated into buckets. The default here is **bucket** granularity,
one classification per bucket, which is the form in the paper. Both are
implemented (`granularity=`), and they are genuinely different estimators:
summing many `Φ(z_tick)·v_tick` terms averages toward `0.5·V`, so the tick form
has a noise floor near 0.06 where the bucket form sits at 0.5.

Compared across modes the number is uninterpretable — on one real EUR/USD day
it read **−0.47**, which looks damning and says nothing. Compared in matching
mode:

| data | spearman | top-quintile overlap | levels (ours / flowrisk) |
|---|---|---|---|
| synthetic | +0.99 | — | 0.0613 / 0.0606 |
| real EUR/USD, one day | +0.83 | 0.83 | 0.091 / 0.094 |

`compare_with_flowrisk` defaults to tick mode and flags a mismatched comparison
rather than letting it be read as a verdict. That makes it a real check on the
bucketing and BVC machinery — the parts most likely to hide an off-by-one.

The **analytic properties** remain the primary validation. All eight pass on
real EUR/USD ticks.

Separately, `tests/test_parity.py` compares the C++ against the NumPy reference.
The two share no code — monotone cursors and block-skipping searches versus
searchsorted and cumsum — so their agreement is real evidence.

## Known gaps

- **The live engine's counterparty history starts empty.** `record()` must be
  fed each fill's label once it resolves, which is a full label horizon after
  the trade. Until fills are fed back, `cp_toxic_rate_hist` reads the prior.
- **HistData's free tier is bid-only M1 bars** with zero volume. The loader
  imposes a synthetic spread and proxies volume, which is enough to exercise the
  pipeline and not enough to trust a markout fit. Use Dukascopy ticks for that.
- **Dukascopy throttles hard.** Concurrency defaults to 2 and the fetcher backs
  off, but a burst still provokes 429s and redirect loops for several minutes.
  Fetch a few days at a time; everything is cached, so re-running only retries
  the gaps.
- **Tick volume is quoted depth**, the sum of bid and ask size on each quote —
  real Dukascopy data, but a liquidity measure rather than executed volume.
  VPIN buckets are therefore in "quoted size" time, not trade time.
- **Feature selection runs inside the evaluation splits**, so its reported score
  has seen every fold. Treat the chosen subset as a decision to validate on
  genuinely held-out data.
- **No PULSE-style online Bayesian filter.** The design lists it as a step after
  the simpler model is validated; `logreg` and LightGBM are in place, the online
  filter is not.
