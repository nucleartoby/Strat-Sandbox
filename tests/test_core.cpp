#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "fxtox/features.hpp"
#include "fxtox/gate.hpp"
#include "fxtox/labeling.hpp"
#include "fxtox/markout.hpp"
#include "fxtox/model.hpp"
#include "fxtox/search.hpp"
#include "fxtox/vpin.hpp"

using namespace fxtox;

namespace {

int g_failures = 0;
int g_checks = 0;

void check(bool ok, const char* what, const char* file, int line) {
    ++g_checks;
    if (!ok) {
        ++g_failures;
        std::fprintf(stderr, "FAIL %s:%d: %s\n", file, line, what);
    }
}

void check_close(double a, double b, double tol, const char* what, const char* file, int line) {
    ++g_checks;
    const bool ok = std::fabs(a - b) <= tol * (1.0 + std::fabs(a) + std::fabs(b));
    if (!ok) {
        ++g_failures;
        std::fprintf(stderr, "FAIL %s:%d: %s (%.17g vs %.17g)\n", file, line, what, a, b);
    }
}

#define CHECK(cond) check((cond), #cond, __FILE__, __LINE__)
#define CHECK_CLOSE(a, b, tol) check_close((a), (b), (tol), #a " ~= " #b, __FILE__, __LINE__)

void test_norm_cdf() {
    CHECK_CLOSE(norm_cdf(0.0), 0.5, 1e-15);
    CHECK_CLOSE(norm_cdf(1.0), 0.8413447460685429, 1e-14);
    CHECK_CLOSE(norm_cdf(-1.96), 0.024997895148220435, 1e-13);
    CHECK_CLOSE(norm_cdf(3.0), 0.9986501019683699, 1e-14);
    CHECK(norm_cdf(-40.0) >= 0.0);
    CHECK(norm_cdf(40.0) <= 1.0);
}

void test_student_t_cdf() {
    // Symmetry and the analytic Cauchy case (df = 1).
    CHECK_CLOSE(student_t_cdf(0.0, 3.0), 0.5, 1e-14);
    CHECK_CLOSE(student_t_cdf(1.0, 1.0), 0.75, 1e-12);
    CHECK_CLOSE(student_t_cdf(-1.0, 1.0), 0.25, 1e-12);
    CHECK_CLOSE(student_t_cdf(2.0, 5.0) + student_t_cdf(-2.0, 5.0), 1.0, 1e-13);
    // The paper's fractional df must not blow up.
    const double p = student_t_cdf(1.5, 0.25);
    CHECK(p > 0.5 && p < 1.0);
}

void test_welford_and_rolling() {
    Welford w;
    const double xs[] = {2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0};
    for (double x : xs) w.push(x);
    CHECK_CLOSE(w.mean(), 5.0, 1e-14);
    CHECK_CLOSE(w.stddev_pop(), 2.0, 1e-14);      // known population sd
    CHECK_CLOSE(w.variance(), 32.0 / 7.0, 1e-14); // sample variance

    RollingSum r(3);
    CHECK(!r.push(1.0));
    CHECK(!r.push(2.0));
    CHECK(r.push(3.0));
    CHECK_CLOSE(r.mean(), 2.0, 1e-15);
    CHECK(r.push(4.0));
    CHECK_CLOSE(r.mean(), 3.0, 1e-15);
}

void test_quantiler() {
    OnlineQuantiler q(1024);
    for (int i = 0; i < 100; ++i) q.observe(i / 100.0);
    CHECK_CLOSE(q.percentile_of(0.495), 0.50, 0.02);
    CHECK_CLOSE(q.percentile_of(0.995), 1.00, 0.02);
    CHECK(q.percentile_of(-1.0) > 0.0);
    CHECK(q.count() == 100);

    OnlineQuantiler q2(256);
    CHECK_CLOSE(q2.observe(0.4), 1.0, 1e-12);
}


std::vector<Tick> make_ticks(std::size_t n, std::uint64_t seed = 7) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> g(0.0, 1.0);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    std::vector<Tick> v;
    v.reserve(n);
    double mid = 1.10;
    nanos_t ts = 1'700'000'000'000'000'000LL;
    for (std::size_t i = 0; i < n; ++i) {
        mid += 3e-5 * g(rng);
        const double sp = 8e-5 + 2e-5 * u(rng);
        v.push_back(Tick{ts, mid - sp / 2, mid + sp / 2,
                         1e6 * (0.5 + u(rng)), 1e6 * (0.5 + u(rng)),
                         1000.0 * (1.0 + std::floor(u(rng) * 20.0))});
        ts += static_cast<nanos_t>(1e6 * (1.0 + u(rng) * 10.0));
    }
    return v;
}

TickView view_of(const std::vector<Tick>& ticks, std::vector<nanos_t>& ts,
                 std::vector<double>& bid, std::vector<double>& ask,
                 std::vector<double>& bs, std::vector<double>& as,
                 std::vector<double>& vol) {
    const std::size_t n = ticks.size();
    ts.resize(n); bid.resize(n); ask.resize(n); bs.resize(n); as.resize(n); vol.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        ts[i] = ticks[i].ts; bid[i] = ticks[i].bid; ask[i] = ticks[i].ask;
        bs[i] = ticks[i].bid_size; as[i] = ticks[i].ask_size; vol[i] = ticks[i].volume;
    }
    TickView v;
    v.ts = ts.data(); v.bid = bid.data(); v.ask = ask.data();
    v.bid_size = bs.data(); v.ask_size = as.data(); v.volume = vol.data();
    v.n = n;
    return v;
}

void test_vpin_invariants() {
    const auto ticks = make_ticks(20000);
    double total = 0.0;
    for (const Tick& t : ticks) total += t.volume;
    const double V = total / 200.0;

    VpinConfig cfg;
    cfg.bucket_volume = V;
    cfg.window = 20;
    cfg.sigma_warmup = 5;

    VpinEngine engine(cfg);
    std::vector<VpinReading> out;
    for (const Tick& t : ticks) engine.on_tick(t, out);

    CHECK(out.size() == 200 || out.size() == 199);

    for (const VpinReading& r : out) {
        CHECK_CLOSE(r.v_buy + r.v_sell, V, 1e-12);
        CHECK(r.v_buy >= -1e-9 && r.v_sell >= -1e-9);
        CHECK(r.imbalance >= 0.0 && r.imbalance <= 1.0 + 1e-12);
        if (!std::isnan(r.vpin)) CHECK(r.vpin >= 0.0 && r.vpin <= 1.0 + 1e-12);
        if (!std::isnan(r.percentile)) CHECK(r.percentile > 0.0 && r.percentile <= 1.0);
    }

    std::size_t reported = 0, withheld = 0;
    for (const VpinReading& r : out) {
        if (std::isnan(r.vpin)) continue;
        if (std::isnan(r.percentile)) ++withheld; else ++reported;
    }
    CHECK(withheld == cfg.pctile_warmup - 1);
    CHECK(reported > 0);

    std::size_t first_finite = out.size();
    for (std::size_t i = 0; i < out.size(); ++i)
        if (!std::isnan(out[i].vpin)) { first_finite = i; break; }
    CHECK(first_finite == cfg.window - 1);
}

void test_vpin_exact_splitting() {
    VpinConfig cfg;
    cfg.bucket_volume = 100.0;
    cfg.window = 1;
    VpinEngine engine(cfg);
    std::vector<VpinReading> out;
    engine.on_tick(Tick{1000, 1.0999, 1.1001, 0, 0, 1000.0}, out);
    CHECK(out.size() == 10);
    for (const VpinReading& r : out) CHECK_CLOSE(r.v_buy + r.v_sell, 100.0, 1e-12);

    VpinEngine e2(cfg);
    std::vector<VpinReading> o2;
    for (int i = 0; i < 1000; ++i)
        e2.on_tick(Tick{1000 + i, 1.0999, 1.1001, 0, 0, 1.0}, o2);
    CHECK(o2.size() == 10);
}

void test_vpin_detects_directional_flow() {
    auto run = [](bool trending) {
        std::mt19937_64 rng(11);
        std::normal_distribution<double> g(0.0, 1.0);
        VpinConfig cfg;
        cfg.bucket_volume = 10000.0;
        cfg.window = 10;
        cfg.sigma_warmup = 3;
        VpinEngine e(cfg);
        std::vector<VpinReading> out;
        double mid = 1.10;
        for (int i = 0; i < 20000; ++i) {
            mid += (trending ? 2e-5 : 0.0) + 2e-5 * g(rng);
            e.on_tick(Tick{1'700'000'000'000'000'000LL + i * 1'000'000LL,
                           mid - 4e-5, mid + 4e-5, 0, 0, 100.0}, out);
        }
        double s = 0.0;
        std::size_t k = 0;
        for (const VpinReading& r : out)
            if (!std::isnan(r.vpin)) { s += r.vpin; ++k; }
        return k ? s / static_cast<double>(k) : 0.0;
    };
    const double trend = run(true), noise = run(false);
    CHECK(trend > noise);
    CHECK(trend > 0.3);
}


void test_block_index_vs_bruteforce() {
    std::mt19937_64 rng(3);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    std::vector<double> x(5000);
    for (double& v : x) v = u(rng);
    BlockIndex idx(x.data(), x.size());

    std::uniform_int_distribution<std::size_t> pick(0, x.size() - 1);
    for (int trial = 0; trial < 2000; ++trial) {
        std::size_t lo = pick(rng), hi = pick(rng);
        if (lo > hi) std::swap(lo, hi);
        const double thr = u(rng);

        std::size_t want_ge = BlockIndex::npos, want_le = BlockIndex::npos;
        for (std::size_t i = lo; i <= hi; ++i) {
            if (want_ge == BlockIndex::npos && x[i] >= thr) want_ge = i;
            if (want_le == BlockIndex::npos && x[i] <= thr) want_le = i;
            if (want_ge != BlockIndex::npos && want_le != BlockIndex::npos) break;
        }
        CHECK(idx.first_ge(lo, hi, thr) == want_ge);
        CHECK(idx.first_le(lo, hi, thr) == want_le);
    }
}

void test_prevailing_index() {
    std::vector<nanos_t> ts{10, 20, 30, 40, 50};
    std::vector<double> mid{1, 2, 3, 4, 5};
    MidSeries s{ts.data(), mid.data(), ts.size()};

    CHECK(prevailing_index(s, 5) == static_cast<std::size_t>(-1));
    CHECK(prevailing_index(s, 10) == 0);
    CHECK(prevailing_index(s, 25) == 1);
    CHECK(prevailing_index(s, 30) == 2);
    CHECK(prevailing_index(s, 999) == 4);
    CHECK(prevailing_index(s, 15, 4) == 0);
}


void test_markouts() {
    std::vector<nanos_t> ts;
    std::vector<double> mid;
    for (int i = 0; i <= 100; ++i) {
        ts.push_back(static_cast<nanos_t>(i) * 1'000'000'000LL);
        mid.push_back(1.1000 + i * 1e-4);
    }
    MidSeries ms{ts.data(), mid.data(), ts.size()};
    std::vector<nanos_t> f_ts{0};
    std::vector<double> f_px{1.1000}, f_sz{1.0};
    std::vector<std::int8_t> f_side{1};
    FillView fv{f_ts.data(), f_px.data(), f_sz.data(), f_side.data(), nullptr, 1};

    std::vector<double> horizons{1.0, 10.0, 30.0};
    std::vector<double> out(3);
    compute_markouts(fv, ms, horizons.data(), horizons.size(), out.data());

    CHECK_CLOSE(out[0], -1e-4, 1e-12);  // 1s  mid 1.1001
    CHECK_CLOSE(out[1], -10e-4, 1e-12); // 10s
    CHECK_CLOSE(out[2], -30e-4, 1e-12); // 30s

    f_side[0] = -1;
    compute_markouts(fv, ms, horizons.data(), horizons.size(), out.data());
    CHECK_CLOSE(out[0], 1e-4, 1e-12);

    std::vector<double> far{1000.0};
    std::vector<double> out2(1);
    compute_markouts(fv, ms, far.data(), 1, out2.data());
    CHECK(std::isnan(out2[0]));
}


void test_labeling_crossback() {
    std::vector<nanos_t> ts;
    std::vector<double> mid;
    for (int i = 0; i <= 60; ++i) {
        ts.push_back(static_cast<nanos_t>(i) * 1'000'000'000LL);
        mid.push_back(1.1000 + i * 1e-4);
    }
    MidSeries ms{ts.data(), mid.data(), ts.size()};
    BlockIndex idx(mid.data(), mid.size());

    std::vector<nanos_t> f_ts{0};
    std::vector<double> f_px{1.1000}, f_sz{1.0};
    std::vector<std::int8_t> f_side{1}; // client buys
    FillView fv{f_ts.data(), f_px.data(), f_sz.data(), f_side.data(), nullptr, 1};

    LabelConfig cfg;
    cfg.rule = LabelRule::Crossback;
    cfg.horizon_sec = 30.0;
    std::vector<std::int8_t> out(1);

    label_fills(fv, ms, idx, cfg, nullptr, out.data());
    CHECK(out[0] == kLabelToxic);

    f_side[0] = -1;
    label_fills(fv, ms, idx, cfg, nullptr, out.data());
    CHECK(out[0] == kLabelBenign);

    f_side[0] = 1;
    cfg.min_adverse = 10e-4;
    label_fills(fv, ms, idx, cfg, nullptr, out.data());
    CHECK(out[0] == kLabelToxic);
    cfg.min_adverse = 100e-4;
    label_fills(fv, ms, idx, cfg, nullptr, out.data());
    CHECK(out[0] == kLabelBenign);
    cfg.min_adverse = 0.0;

    cfg.horizon_sec = 600.0;
    label_fills(fv, ms, idx, cfg, nullptr, out.data());
    CHECK(out[0] == kLabelUnknown);
}

void test_crossback_direction_on_flat_market() {
    std::vector<nanos_t> ts;
    std::vector<double> mid;
    for (int i = 0; i <= 60; ++i) {
        ts.push_back(static_cast<nanos_t>(i) * 1'000'000'000LL);
        mid.push_back(1.1000); // dead flat
    }
    MidSeries ms{ts.data(), mid.data(), ts.size()};
    BlockIndex idx(mid.data(), mid.size());

    std::vector<nanos_t> f_ts{0};
    std::vector<double> f_px{1.10005}, f_sz{1.0};
    std::vector<std::int8_t> f_side{1};
    FillView fv{f_ts.data(), f_px.data(), f_sz.data(), f_side.data(), nullptr, 1};

    LabelConfig cfg;
    cfg.rule = LabelRule::Crossback;
    cfg.horizon_sec = 30.0;
    std::vector<std::int8_t> out(1);
    label_fills(fv, ms, idx, cfg, nullptr, out.data());
    CHECK(out[0] == kLabelBenign);

    f_px[0] = 1.09995;
    f_side[0] = -1;
    label_fills(fv, ms, idx, cfg, nullptr, out.data());
    CHECK(out[0] == kLabelBenign);
}

void test_labeling_triple_barrier() {
    std::vector<nanos_t> ts;
    std::vector<double> mid;
    for (int i = 0; i <= 60; ++i) {
        ts.push_back(static_cast<nanos_t>(i) * 1'000'000'000LL);
        mid.push_back(i < 5 ? 1.1000 - i * 1e-5 : 1.1000 + (i - 5) * 1e-4);
    }
    MidSeries ms{ts.data(), mid.data(), ts.size()};
    BlockIndex idx(mid.data(), mid.size());

    std::vector<nanos_t> f_ts{0};
    std::vector<double> f_px{1.1000}, f_sz{1.0};
    std::vector<std::int8_t> f_side{1};
    FillView fv{f_ts.data(), f_px.data(), f_sz.data(), f_side.data(), nullptr, 1};

    LabelConfig cfg;
    cfg.rule = LabelRule::TripleBarrier;
    cfg.horizon_sec = 30.0;
    std::vector<std::int8_t> out(1);

    cfg.theta = 5e-4;
    label_fills(fv, ms, idx, cfg, nullptr, out.data());
    CHECK(out[0] == kLabelToxic);

    cfg.theta = 2e-5;
    label_fills(fv, ms, idx, cfg, nullptr, out.data());
    CHECK(out[0] == kLabelBenign);
}


void test_feature_parity_batch_vs_live() {
    const auto ticks = make_ticks(6000, 21);
    std::vector<nanos_t> ts; std::vector<double> bid, ask, bs, as, vol;
    const TickView tv = view_of(ticks, ts, bid, ask, bs, as, vol);

    std::vector<nanos_t> f_ts;
    std::vector<double> f_px, f_sz;
    std::vector<std::int8_t> f_side;
    for (std::size_t i = 0; i < ticks.size(); i += 50) {
        f_ts.push_back(ticks[i].ts);
        f_px.push_back(ticks[i].ask);
        f_sz.push_back(1e6);
        f_side.push_back(1);
    }
    FillView fv{f_ts.data(), f_px.data(), f_sz.data(), f_side.data(), nullptr, f_ts.size()};

    FeatureConfig cfg;
    cfg.short_window_sec = 5.0;
    cfg.long_window_sec = 30.0;

    std::vector<double> batch(f_ts.size() * kFeatCount);
    build_features(tv, fv, cfg, batch.data());

    LiveFeatures live(cfg);
    std::vector<double> row(kFeatCount);
    std::size_t fi = 0;
    for (std::size_t i = 0; i < ticks.size() && fi < f_ts.size(); ++i) {
        live.on_tick(ticks[i]);
        while (fi < f_ts.size() && f_ts[fi] == ticks[i].ts) {
            live.snapshot(f_sz[fi], f_ts[fi], row.data());
            for (int c = 0; c < kFeatCount; ++c) {
                const double b = batch[fi * kFeatCount + c];
                const double l = row[c];
                if (std::isnan(b) && std::isnan(l)) continue;
                check(b == l, feature_name(c), __FILE__, __LINE__);
            }
            ++fi;
        }
    }
    CHECK(fi == f_ts.size());
}

void test_counterparty_history_is_causal() {
    std::vector<std::int32_t> cp{1, 1, 1, 2, 1};
    std::vector<std::int8_t> y{1, 1, 1, 0, 0};
    std::vector<nanos_t> ts(cp.size(), 0);
    std::vector<double> px(cp.size(), 1.0), sz(cp.size(), 1.0);
    std::vector<std::int8_t> side(cp.size(), 1);
    FillView fv{ts.data(), px.data(), sz.data(), side.data(), cp.data(), cp.size()};

    std::vector<double> rate(cp.size()), count(cp.size());
    counterparty_history(fv, y.data(), 0.5, 2.0, rate.data(), count.data());

    CHECK_CLOSE(rate[0], 0.5, 1e-12);
    CHECK_CLOSE(count[0], 0.0, 1e-12);
    CHECK_CLOSE(rate[1], (1.0 + 1.0) / (1.0 + 2.0), 1e-12);
    CHECK_CLOSE(rate[3], 0.5, 1e-12);
    CHECK_CLOSE(count[4], 3.0, 1e-12);
    CHECK_CLOSE(rate[4], (3.0 + 1.0) / (3.0 + 2.0), 1e-12);
}


void write_names(std::FILE* f, const std::vector<std::string>& names) {
    const std::uint32_t n = static_cast<std::uint32_t>(names.size());
    std::fwrite(&n, sizeof n, 1, f);
    for (const std::string& s : names) {
        const std::uint32_t len = static_cast<std::uint32_t>(s.size());
        std::fwrite(&len, sizeof len, 1, f);
        std::fwrite(s.data(), 1, s.size(), f);
    }
}

std::string write_logistic_model() {
    const std::string path = "/tmp/fxtox_test_model.fxm";
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) { CHECK(false); return path; }
    const std::uint32_t version = kModelVersion, kind = 0, nfeat = 3, flags = kFlagStandardize;
    const double sig = 1.0;
    std::fwrite("FXTM", 1, 4, f);
    std::fwrite(&version, sizeof version, 1, f);
    std::fwrite(&kind, sizeof kind, 1, f);
    std::fwrite(&nfeat, sizeof nfeat, 1, f);
    std::fwrite(&flags, sizeof flags, 1, f);
    std::fwrite(&sig, sizeof sig, 1, f);
    write_names(f, {"alpha", "beta", "gamma"});
    const double mean[3] = {0.0, 1.0, 2.0};
    const double scale[3] = {1.0, 2.0, 4.0};
    std::fwrite(mean, sizeof(double), 3, f);
    std::fwrite(scale, sizeof(double), 3, f);
    const double bias = -0.5;
    const double w[3] = {1.0, -2.0, 0.25};
    std::fwrite(&bias, sizeof bias, 1, f);
    std::fwrite(w, sizeof(double), 3, f);
    std::fclose(f);
    return path;
}

void test_model_logistic() {
    const std::string path = write_logistic_model();
    const Model m = Model::load(path);
    CHECK(m.n_features() == 3);

    const double x[3] = {1.0, 3.0, 6.0};
    // z = (1-0)/1, (3-1)/2, (6-2)/4 = 1, 1, 1
    // raw = -0.5 + 1*1 + (-2)*1 + 0.25*1 = -1.25
    CHECK_CLOSE(m.raw_score(x), -1.25, 1e-12);
    CHECK_CLOSE(m.predict_proba(x), 1.0 / (1.0 + std::exp(1.25)), 1e-12);

    std::vector<int> idx;
    std::string missing;
    CHECK(m.bind({"noise", "gamma", "alpha", "beta"}, idx, missing));
    CHECK(idx.size() == 3);
    CHECK(idx[0] == 2 && idx[1] == 3 && idx[2] == 1);

    CHECK(!m.bind({"alpha", "beta"}, idx, missing));
    CHECK(missing == "gamma");
    std::remove(path.c_str());
}

void test_model_gbdt() {
    const std::string path = "/tmp/fxtox_test_gbdt.fxm";
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) { CHECK(false); return; }
    const std::uint32_t version = kModelVersion, kind = 1, nfeat = 2, flags = 0, ntrees = 2;
    const double sig = 1.0, base = 0.1;
    std::fwrite("FXTM", 1, 4, f);
    std::fwrite(&version, sizeof version, 1, f);
    std::fwrite(&kind, sizeof kind, 1, f);
    std::fwrite(&nfeat, sizeof nfeat, 1, f);
    std::fwrite(&flags, sizeof flags, 1, f);
    std::fwrite(&sig, sizeof sig, 1, f);
    write_names(f, {"x0", "x1"});
    std::fwrite(&base, sizeof base, 1, f);
    std::fwrite(&ntrees, sizeof ntrees, 1, f);

    auto write_tree = [&](double lo, double hi) {
        const std::uint32_t n_nodes = 3;
        std::fwrite(&n_nodes, sizeof n_nodes, 1, f);
        const std::int32_t nodes[3][4] = {{0, 1, 2, 1}, {-1, -1, -1, 1}, {-1, -1, -1, 1}};
        const double vals[3] = {0.5, lo, hi};
        for (int k = 0; k < 3; ++k) {
            std::fwrite(nodes[k], sizeof(std::int32_t), 4, f);
            std::fwrite(&vals[k], sizeof(double), 1, f);
        }
    };
    write_tree(-1.0, 1.0);
    write_tree(-2.0, 2.0);
    std::fclose(f);

    const Model m = Model::load(path);
    CHECK(m.kind() == ModelKind::Gbdt);
    CHECK(m.n_trees() == 2);

    const double lo[2] = {0.0, 0.0};
    CHECK_CLOSE(m.raw_score(lo), 0.1 - 1.0 - 2.0, 1e-12);
    const double hi[2] = {1.0, 0.0};
    CHECK_CLOSE(m.raw_score(hi), 0.1 + 1.0 + 2.0, 1e-12);

    const double nan_x[2] = {std::nan(""), 0.0};
    CHECK_CLOSE(m.raw_score(nan_x), 0.1 - 1.0 - 2.0, 1e-12);
    std::remove(path.c_str());
}

void test_model_rejects_garbage() {
    const std::string path = "/tmp/fxtox_test_bad.fxm";
    std::FILE* f = std::fopen(path.c_str(), "wb");
    std::fwrite("NOPE", 1, 4, f);
    std::fclose(f);
    bool threw = false;
    try { Model::load(path); } catch (const std::exception&) { threw = true; }
    CHECK(threw);
    std::remove(path.c_str());

    threw = false;
    try { Model::load("/tmp/fxtox_definitely_missing.fxm"); }
    catch (const std::exception&) { threw = true; }
    CHECK(threw);

    const std::string old_path = "/tmp/fxtox_test_v1.fxm";
    f = std::fopen(old_path.c_str(), "wb");
    const std::uint32_t old_version = 1, kind = 0, nfeat = 1, flags = 0;
    const double sig = 1.0;
    std::fwrite("FXTM", 1, 4, f);
    std::fwrite(&old_version, sizeof old_version, 1, f);
    std::fwrite(&kind, sizeof kind, 1, f);
    std::fwrite(&nfeat, sizeof nfeat, 1, f);
    std::fwrite(&flags, sizeof flags, 1, f);
    std::fwrite(&sig, sizeof sig, 1, f);
    std::fclose(f);
    threw = false;
    try { Model::load(old_path); } catch (const std::exception&) { threw = true; }
    CHECK(threw);
    std::remove(old_path.c_str());
}

void test_gate_hysteresis() {
    GateConfig cfg;
    cfg.widen_above = 0.65;
    cfg.suspend_above = 0.85;
    cfg.tighten_below = 0.35;
    cfg.hysteresis = 0.05;
    cfg.min_dwell_ns = 1'000'000'000LL;
    RiskGate gate(cfg);

    nanos_t t = 0;
    CHECK(gate.on_update(t, 0.50, 1e-4).action == QuoteAction::Normal);

    t += 1'000'000;
    CHECK(gate.on_update(t, 0.70, 1e-4).action == QuoteAction::Widen);

    t += 1'000'000;
    CHECK(gate.on_update(t, 0.63, 1e-4).action == QuoteAction::Widen);

    t += 1'000'000;
    CHECK(gate.on_update(t, 0.50, 1e-4).action == QuoteAction::Widen);

    // Past the dwell, it relaxes.
    t += 2'000'000'000LL;
    CHECK(gate.on_update(t, 0.50, 1e-4).action == QuoteAction::Normal);

    t += 1'000'000;
    CHECK(gate.on_update(t, 0.90, 1e-4).action == QuoteAction::Suspend);
    CHECK_CLOSE(gate.on_update(t, 0.90, 1e-4).spread, 0.0, 1e-15);
}

void test_gate_prices_adverse_selection() {
    GateConfig cfg;
    cfg.alpha_mu = 5e-5;
    cfg.max_widen_factor = 10.0;
    RiskGate gate(cfg);

    const GateDecision d = gate.on_update(0, 0.9, 3e-5);
    CHECK(d.action == QuoteAction::Suspend);
    GateConfig cfg2 = cfg;
    cfg2.suspend_above = 0.99;
    RiskGate g2(cfg2);
    const GateDecision d2 = g2.on_update(0, 0.9, 3e-5);
    CHECK(d2.action == QuoteAction::Widen);
    CHECK_CLOSE(d2.multiplier, 3.0, 1e-12);
    CHECK_CLOSE(d2.spread, 9e-5, 1e-12);

    RiskGate g3(cfg2);
    const GateDecision d3 = g3.on_update(0, 0.70, 1e-3);
    CHECK(d3.multiplier >= 1.0);
}

} // namespace

int main() {
    test_norm_cdf();
    test_student_t_cdf();
    test_welford_and_rolling();
    test_quantiler();
    test_vpin_invariants();
    test_vpin_exact_splitting();
    test_vpin_detects_directional_flow();
    test_block_index_vs_bruteforce();
    test_prevailing_index();
    test_markouts();
    test_labeling_crossback();
    test_crossback_direction_on_flat_market();
    test_labeling_triple_barrier();
    test_feature_parity_batch_vs_live();
    test_counterparty_history_is_causal();
    test_model_logistic();
    test_model_gbdt();
    test_model_rejects_garbage();
    test_gate_hysteresis();
    test_gate_prices_adverse_selection();

    std::printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
