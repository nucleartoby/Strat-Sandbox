#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "fxtox/features.hpp"
#include "fxtox/gate.hpp"
#include "fxtox/model.hpp"
#include "fxtox/vpin.hpp"

using namespace fxtox;

namespace {

struct Options {
    double bucket_volume = 0.0;
    std::size_t window = 50;
    BvcDist dist = BvcDist::Normal;
    double df = 0.25;
    std::size_t sigma_warmup = 20;
    std::size_t pctile_warmup = 30;
    double short_window_sec = 60.0;
    double long_window_sec = 900.0;
    double quote_size = 1'000'000.0; // notional the gate prices for
    std::int32_t counterparty = -1;  // whose history to score against
    double base_spread = 0.0;        // 0 => use the live spread
    std::string model_path;
    std::string cp_history_path;
    GateConfig gate;
    std::size_t bench = 0;
    bool quiet = false;
};

[[noreturn]] void usage(int code) {
    std::fprintf(code ? stderr : stdout,
        "vpin_realtime -- streaming VPIN + toxicity gate\n\n"
        "  --bucket-volume V     volume per VPIN bucket (required)\n"
        "  --window N            buckets averaged into one VPIN reading [50]\n"
        "  --dist normal|t       BVC distribution [normal]\n"
        "  --df X                Student-t degrees of freedom [0.25]\n"
        "  --sigma-warmup N      buckets before sigma is trusted [20]\n"
        "  --pctile-warmup N     readings before the percentile is reported [30]\n"
        "  --short-window S      short feature lookback, seconds [60]\n"
        "  --long-window S       long feature lookback, seconds [900]\n"
        "  --model PATH          .fxm classifier; without it the gate runs on\n"
        "                        the VPIN percentile directly\n"
        "  --quote-size X        notional the live feature vector assumes [1e6]\n"
        "  --counterparty N      counterparty code to score against [-1]\n"
        "  --cp-history PATH     seed counterparty history from the pipeline's\n"
        "                        .cp.csv (code,toxic,total columns)\n"
        "  --base-spread X       reference spread; 0 uses the live spread [0]\n"
        "  --alpha-mu X          adverse-selection cost from the markout fit [0]\n"
        "  --widen-above P       widen threshold [0.65]\n"
        "  --suspend-above P     suspend threshold [0.85]\n"
        "  --tighten-below P     tighten threshold [0.35]\n"
        "  --bench N             run N synthetic ticks and report throughput\n"
        "  --quiet               suppress per-bucket output\n"
        "  -h, --help            this message\n");
    std::exit(code);
}

double need_num(int argc, char** argv, int& i, const char* flag) {
    if (++i >= argc) { std::fprintf(stderr, "%s requires a value\n", flag); usage(2); }
    return std::atof(argv[i]);
}

Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "-h" || a == "--help") usage(0);
        else if (a == "--bucket-volume") o.bucket_volume = need_num(argc, argv, i, "--bucket-volume");
        else if (a == "--window") o.window = static_cast<std::size_t>(need_num(argc, argv, i, "--window"));
        else if (a == "--df") o.df = need_num(argc, argv, i, "--df");
        else if (a == "--sigma-warmup") o.sigma_warmup = static_cast<std::size_t>(need_num(argc, argv, i, "--sigma-warmup"));
        else if (a == "--pctile-warmup") o.pctile_warmup = static_cast<std::size_t>(need_num(argc, argv, i, "--pctile-warmup"));
        else if (a == "--short-window") o.short_window_sec = need_num(argc, argv, i, "--short-window");
        else if (a == "--long-window") o.long_window_sec = need_num(argc, argv, i, "--long-window");
        else if (a == "--quote-size") o.quote_size = need_num(argc, argv, i, "--quote-size");
        else if (a == "--counterparty") o.counterparty = static_cast<std::int32_t>(need_num(argc, argv, i, "--counterparty"));
        else if (a == "--base-spread") o.base_spread = need_num(argc, argv, i, "--base-spread");
        else if (a == "--alpha-mu") o.gate.alpha_mu = need_num(argc, argv, i, "--alpha-mu");
        else if (a == "--widen-above") o.gate.widen_above = need_num(argc, argv, i, "--widen-above");
        else if (a == "--suspend-above") o.gate.suspend_above = need_num(argc, argv, i, "--suspend-above");
        else if (a == "--tighten-below") o.gate.tighten_below = need_num(argc, argv, i, "--tighten-below");
        else if (a == "--bench") o.bench = static_cast<std::size_t>(need_num(argc, argv, i, "--bench"));
        else if (a == "--quiet") o.quiet = true;
        else if (a == "--dist") {
            if (++i >= argc) usage(2);
            o.dist = (std::strcmp(argv[i], "t") == 0) ? BvcDist::StudentT : BvcDist::Normal;
        } else if (a == "--model") {
            if (++i >= argc) usage(2);
            o.model_path = argv[i];
        } else if (a == "--cp-history") {
            if (++i >= argc) usage(2);
            o.cp_history_path = argv[i];
        } else {
            std::fprintf(stderr, "unknown option: %s\n", a.c_str());
            usage(2);
        }
    }
    if (!(o.bucket_volume > 0.0)) {
        std::fprintf(stderr, "--bucket-volume is required and must be > 0\n");
        usage(2);
    }
    return o;
}

bool parse_line(const char* p, const char* end, Tick& t, const int* col) {
    double vals[6] = {0, 0, 0, 0, 0, 0};
    int field = 0;
    const char* cur = p;
    while (cur <= end && field < 32) {
        const char* comma = static_cast<const char*>(std::memchr(cur, ',', static_cast<std::size_t>(end - cur)));
        const char* stop = comma ? comma : end;
        for (int k = 0; k < 6; ++k) {
            if (col[k] == field) {
                double v = 0.0;
#if defined(__cpp_lib_to_chars) && __cpp_lib_to_chars >= 201611L
                auto r = std::from_chars(cur, stop, v);
                if (r.ec != std::errc()) return false;
#else
                char buf[64];
                const std::size_t len = std::min<std::size_t>(static_cast<std::size_t>(stop - cur), 63);
                std::memcpy(buf, cur, len);
                buf[len] = '\0';
                char* endp = nullptr;
                v = std::strtod(buf, &endp);
                if (endp == buf) return false;
#endif
                vals[k] = v;
            }
        }
        if (!comma) break;
        cur = comma + 1;
        ++field;
    }
    t.ts = static_cast<nanos_t>(vals[0]);
    t.bid = vals[1];
    t.ask = vals[2];
    t.volume = vals[3];
    t.bid_size = vals[4];
    t.ask_size = vals[5];
    return true;
}

bool map_header(const std::string& header, int* col) {
    static const char* kNames[6] = {"ts_ns", "bid", "ask", "volume", "bid_size", "ask_size"};
    for (int k = 0; k < 6; ++k) col[k] = -1;

    std::size_t start = 0;
    int idx = 0;
    while (start <= header.size()) {
        const std::size_t comma = header.find(',', start);
        std::string name = header.substr(start, comma == std::string::npos ? std::string::npos
                                                                           : comma - start);
        // trim
        while (!name.empty() && (name.back() == '\r' || name.back() == ' ')) name.pop_back();
        while (!name.empty() && name.front() == ' ') name.erase(name.begin());
        for (int k = 0; k < 6; ++k)
            if (name == kNames[k]) col[k] = idx;
        if (comma == std::string::npos) break;
        start = comma + 1;
        ++idx;
    }
    return col[0] >= 0 && col[1] >= 0 && col[2] >= 0 && col[3] >= 0;
}

// Reads the code, toxic and total columns of the pipeline
bool load_cp_history(const std::string& path, LiveCounterpartyHistory& h, std::size_t& n) {
    std::FILE* f = std::fopen(path.c_str(), "r");
    if (!f) return false;
    char buf[4096];
    int c_code = -1, c_toxic = -1, c_total = -1;
    bool header = true;
    n = 0;
    while (std::fgets(buf, sizeof buf, f)) {
        std::vector<std::string> cells;
        std::string cell;
        for (const char* p = buf; *p && *p != '\n' && *p != '\r'; ++p) {
            if (*p == ',') { cells.push_back(cell); cell.clear(); }
            else cell += *p;
        }
        cells.push_back(cell);
        if (header) {
            for (int k = 0; k < static_cast<int>(cells.size()); ++k) {
                if (cells[k] == "code") c_code = k;
                else if (cells[k] == "toxic") c_toxic = k;
                else if (cells[k] == "total") c_total = k;
            }
            header = false;
            if (c_code < 0 || c_toxic < 0 || c_total < 0) { std::fclose(f); return false; }
            continue;
        }
        const int need = std::max(c_code, std::max(c_toxic, c_total));
        if (static_cast<int>(cells.size()) <= need) continue;
        h.seed(static_cast<std::int32_t>(std::atoi(cells[c_code].c_str())),
               std::atof(cells[c_toxic].c_str()), std::atof(cells[c_total].c_str()));
        ++n;
    }
    std::fclose(f);
    return true;
}

std::vector<Tick> synth(std::size_t n, std::uint64_t seed = 42) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> gauss(0.0, 1.0);
    std::uniform_real_distribution<double> uni(0.0, 1.0);

    std::vector<Tick> out;
    out.reserve(n);
    double mid = 1.1000;
    double drift = 0.0;
    nanos_t ts = 1'700'000'000'000'000'000LL;

    for (std::size_t i = 0; i < n; ++i) {
        if (uni(rng) < 0.00002) drift = (uni(rng) < 0.5 ? -1.0 : 1.0) * 2e-6; // informed burst
        if (uni(rng) < 0.0005) drift = 0.0;
        mid += drift + 3e-5 * gauss(rng);
        const double spread = 8e-5 + 2e-5 * std::fabs(gauss(rng));
        Tick t;
        t.ts = ts;
        t.bid = mid - 0.5 * spread;
        t.ask = mid + 0.5 * spread;
        t.bid_size = 1e6 * (0.5 + uni(rng));
        t.ask_size = 1e6 * (0.5 + uni(rng));
        t.volume = 1000.0 * (1.0 + std::floor(uni(rng) * 50.0));
        out.push_back(t);
        ts += static_cast<nanos_t>(1e6 * (1.0 + uni(rng) * 20.0)); // 1-21 ms apart
    }
    return out;
}

} // namespace

int main(int argc, char** argv) {
    const Options o = parse(argc, argv);

    VpinConfig vcfg;
    vcfg.bucket_volume = o.bucket_volume;
    vcfg.window = o.window;
    vcfg.dist = o.dist;
    vcfg.df = o.df;
    vcfg.sigma_mode = SigmaMode::Expanding; // the only honest choice live
    vcfg.sigma_warmup = o.sigma_warmup;

    FeatureConfig fcfg;
    fcfg.short_window_sec = o.short_window_sec;
    fcfg.long_window_sec = o.long_window_sec;

    vcfg.pctile_warmup = o.pctile_warmup;

    VpinEngine engine(vcfg);
    LiveFeatures feats(fcfg);
    RiskGate gate(o.gate);

    std::unique_ptr<Model> model;
    if (!o.model_path.empty()) {
        try {
            model = std::make_unique<Model>(Model::load(o.model_path));
        } catch (const std::exception& e) {
            std::fprintf(stderr, "model load failed: %s\n", e.what());
            return 1;
        }
    }

    std::vector<std::string> available;
    for (int c = 0; c < kFeatCount; ++c) available.emplace_back(feature_name(c));
    available.emplace_back("cp_toxic_rate_hist");
    available.emplace_back("cp_fill_count");
    available.emplace_back("size");
    available.emplace_back("vpin");
    available.emplace_back("vpin_pctile");
    available.emplace_back("is_client_buy");
    const std::size_t kIsClientBuy = available.size() - 1;

    std::vector<int> binding;
    if (model) {
        std::string missing;
        if (!model->bind(available, binding, missing)) {
            std::fprintf(stderr,
                "model needs feature %s, which this engine does not compute.\n"
                "Available: ", missing.c_str());
            for (const std::string& n : available) std::fprintf(stderr, "%s ", n.c_str());
            std::fprintf(stderr, "\n");
            return 1;
        }
    }

    std::vector<double> pool(available.size(), 0.0);
    std::vector<double> fvec(model ? model->n_features() : 0, 0.0);
    LiveCounterpartyHistory cp_history;
    if (!o.cp_history_path.empty()) {
        std::size_t n_cp = 0;
        if (!load_cp_history(o.cp_history_path, cp_history, n_cp)) {
            std::fprintf(stderr, "could not read --cp-history %s (needs code,toxic,total "
                                 "columns)\n", o.cp_history_path.c_str());
            return 1;
        }
        std::fprintf(stderr, "counterparty history: %zu counterparties; code %d has "
                             "rate %.3f over %.0f fills\n", n_cp, o.counterparty,
                     cp_history.rate_for(o.counterparty), cp_history.fill_count(o.counterparty));
    }
    std::uint64_t n_ticks = 0, n_buckets = 0, n_suspend = 0, n_widen = 0;

    auto process = [&](const Tick& t) {
        ++n_ticks;
        feats.on_tick(t);
        engine.on_tick(t, [&](const VpinReading& r) {
            ++n_buckets;
            if (std::isnan(r.vpin)) return;

            double p;
            if (model) {
                feats.snapshot(o.quote_size, t.ts, pool.data());
                pool[kFeatCount + 0] = cp_history.rate_for(o.counterparty);
                pool[kFeatCount + 1] = cp_history.fill_count(o.counterparty);
                pool[kFeatCount + 2] = o.quote_size;
                pool[kFeatCount + 3] = r.vpin;
                pool[kFeatCount + 4] = r.percentile;

                p = 0.0;
                for (double side : {1.0, 0.0}) {
                    pool[kIsClientBuy] = side;
                    for (std::size_t j = 0; j < binding.size(); ++j)
                        fvec[j] = pool[static_cast<std::size_t>(binding[j])];
                    const double q = model->predict_proba(fvec.data());
                    if (std::isnan(q)) return;
                    p = std::max(p, q);
                }
            } else {
                if (std::isnan(r.percentile)) return;
                p = r.percentile;
            }

            const double base = o.base_spread > 0.0 ? o.base_spread : t.spread();
            const GateDecision d = gate.on_update(t.ts, p, base);
            if (d.action == QuoteAction::Suspend) ++n_suspend;
            else if (d.action == QuoteAction::Widen) ++n_widen;

            if (!o.quiet) {
                std::printf("%lld,%.6f,%.6f,%.8f,%.6f,%s,%.8f,%.3f\n",
                            static_cast<long long>(r.ts_end), r.vpin, r.percentile,
                            r.price_end, p, quote_action_name(d.action), d.spread,
                            d.multiplier);
            }
        });
    };

    const auto t0 = std::chrono::steady_clock::now();

    if (o.bench) {
        const std::vector<Tick> ticks = synth(o.bench);
        const auto b0 = std::chrono::steady_clock::now();
        for (const Tick& t : ticks) process(t);
        const auto b1 = std::chrono::steady_clock::now();
        const double sec = std::chrono::duration<double>(b1 - b0).count();
        std::fprintf(stderr,
            "bench: %zu ticks in %.3f s = %.2f M ticks/s (%.1f ns/tick), "
            "%llu buckets, widen=%llu suspend=%llu\n",
            o.bench, sec, o.bench / sec / 1e6, sec * 1e9 / static_cast<double>(o.bench),
            static_cast<unsigned long long>(n_buckets),
            static_cast<unsigned long long>(n_widen),
            static_cast<unsigned long long>(n_suspend));
        return 0;
    }

    if (!o.quiet) std::printf("ts_end,vpin,percentile,price,toxicity,action,spread,multiplier\n");

    std::ios::sync_with_stdio(false);
    std::string line;
    if (!std::getline(std::cin, line)) {
        std::fprintf(stderr, "empty input\n");
        return 1;
    }
    int col[6];
    if (!map_header(line, col)) {
        std::fprintf(stderr,
            "header must name at least ts_ns,bid,ask,volume (got: %s)\n", line.c_str());
        return 1;
    }

    Tick t{};
    std::uint64_t bad = 0;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        if (!parse_line(line.data(), line.data() + line.size(), t, col)) { ++bad; continue; }
        if (!(t.ask > t.bid)) { ++bad; continue; } // crossed/locked: drop, as in Phase 2
        process(t);
    }

    const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::fprintf(stderr,
        "processed %llu ticks (%llu rejected) -> %llu buckets in %.3f s; "
        "widen=%llu suspend=%llu\n",
        static_cast<unsigned long long>(n_ticks), static_cast<unsigned long long>(bad),
        static_cast<unsigned long long>(n_buckets), sec,
        static_cast<unsigned long long>(n_widen),
        static_cast<unsigned long long>(n_suspend));
    return 0;
}
