#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stdexcept>
#include <string>
#include <vector>

#include "fxtox/features.hpp"
#include "fxtox/gate.hpp"
#include "fxtox/labeling.hpp"
#include "fxtox/markout.hpp"
#include "fxtox/model.hpp"
#include "fxtox/search.hpp"
#include "fxtox/types.hpp"
#include "fxtox/vpin.hpp"

namespace py = pybind11;
using namespace fxtox;

namespace {

template <typename T>
using arr = py::array_t<T, py::array::c_style | py::array::forcecast>;

using arr_d = arr<double>;
using arr_i64 = arr<std::int64_t>;
using arr_i8 = arr<std::int8_t>;
using arr_i32 = arr<std::int32_t>;

void require(bool ok, const char* msg) {
    if (!ok) throw std::invalid_argument(msg);
}

struct Keeper {
    std::vector<py::object> held;
    template <typename A>
    const typename A::value_type* take(const py::object& o, std::size_t expect, const char* name) {
        if (o.is_none()) return nullptr;
        A a = o.cast<A>();
        if (static_cast<std::size_t>(a.size()) != expect)
            throw std::invalid_argument(std::string(name) + ": length mismatch");
        const auto* p = static_cast<const typename A::value_type*>(a.request().ptr);
        held.push_back(std::move(a));
        return p;
    }
};

} // namespace

PYBIND11_MODULE(fxtox_native, m) {
    m.doc() = "Latency-critical core of the FX toxic-flow pipeline (C++17).";
    m.attr("FEATURE_COUNT") = static_cast<int>(kFeatCount);

    m.def("feature_names", [] {
        std::vector<std::string> names;
        names.reserve(kFeatCount);
        for (int c = 0; c < kFeatCount; ++c) names.emplace_back(feature_name(c));
        return names;
    }, "Feature-matrix column names, in column order.");

    py::enum_<BvcDist>(m, "BvcDist")
        .value("NORMAL", BvcDist::Normal)
        .value("STUDENT_T", BvcDist::StudentT);

    py::enum_<BvcGranularity>(m, "BvcGranularity")
        .value("BUCKET", BvcGranularity::Bucket)
        .value("TICK", BvcGranularity::Tick);

    py::enum_<SigmaMode>(m, "SigmaMode")
        .value("EXPANDING", SigmaMode::Expanding)
        .value("FIXED", SigmaMode::Fixed);

    py::enum_<BarrierCenter>(m, "BarrierCenter")
        .value("EXEC_PRICE", BarrierCenter::ExecPrice)
        .value("MID_AT_FILL", BarrierCenter::MidAtFill);

    py::enum_<LabelRule>(m, "LabelRule")
        .value("MARKOUT_THRESHOLD", LabelRule::MarkoutThreshold)
        .value("CROSSBACK", LabelRule::Crossback)
        .value("TRIPLE_BARRIER", LabelRule::TripleBarrier);

    py::enum_<QuoteAction>(m, "QuoteAction")
        .value("TIGHTEN", QuoteAction::Tighten)
        .value("NORMAL", QuoteAction::Normal)
        .value("WIDEN", QuoteAction::Widen)
        .value("SUSPEND", QuoteAction::Suspend);

    m.def("norm_cdf", py::vectorize(norm_cdf));
    m.def("student_t_cdf", py::vectorize(student_t_cdf));

    m.def("compute_vpin",
        [](arr_i64 ts, arr_d bid, arr_d ask, arr_d volume, double bucket_volume,
           std::size_t window, BvcDist dist, double df, BvcGranularity gran,
           SigmaMode sigma_mode, double sigma_fixed, std::size_t sigma_warmup,
           std::size_t pctile_bins, std::size_t pctile_warmup) {
            const std::size_t n = static_cast<std::size_t>(ts.size());
            require(bid.size() == ts.size() && ask.size() == ts.size() &&
                    volume.size() == ts.size(), "compute_vpin: column length mismatch");
            require(bucket_volume > 0.0, "compute_vpin: bucket_volume must be > 0");

            TickView tv;
            tv.ts = static_cast<const nanos_t*>(ts.request().ptr);
            tv.bid = static_cast<const double*>(bid.request().ptr);
            tv.ask = static_cast<const double*>(ask.request().ptr);
            tv.volume = static_cast<const double*>(volume.request().ptr);
            tv.n = n;

            VpinConfig cfg;
            cfg.bucket_volume = bucket_volume;
            cfg.window = window;
            cfg.dist = dist;
            cfg.df = df;
            cfg.granularity = gran;
            cfg.sigma_mode = sigma_mode;
            cfg.sigma_fixed = sigma_fixed;
            cfg.sigma_warmup = sigma_warmup;
            cfg.pctile_bins = pctile_bins;
            cfg.pctile_warmup = pctile_warmup;

            const std::size_t cap = bucket_capacity(tv, bucket_volume);
            arr_i64 o_ts(cap);
            arr_d o_vpin(cap), o_pct(cap), o_imb(cap), o_vb(cap), o_vs(cap), o_px(cap);
            arr<std::uint8_t> o_warm(cap);

            VpinBatchOut out;
            out.ts_end = static_cast<nanos_t*>(o_ts.request().ptr);
            out.vpin = static_cast<double*>(o_vpin.request().ptr);
            out.percentile = static_cast<double*>(o_pct.request().ptr);
            out.imbalance = static_cast<double*>(o_imb.request().ptr);
            out.v_buy = static_cast<double*>(o_vb.request().ptr);
            out.v_sell = static_cast<double*>(o_vs.request().ptr);
            out.price_end = static_cast<double*>(o_px.request().ptr);
            out.warmup = static_cast<std::uint8_t*>(o_warm.request().ptr);

            std::size_t k;
            {
                py::gil_scoped_release release;
                k = compute_vpin_batch(tv, cfg, out, cap);
            }

            auto trim_d = [&](arr_d& a) { a.resize({k}, false); return a; };
            py::dict d;
            o_ts.resize({k}, false);
            o_warm.resize({k}, false);
            d["ts_end"] = o_ts;
            d["vpin"] = trim_d(o_vpin);
            d["percentile"] = trim_d(o_pct);
            d["imbalance"] = trim_d(o_imb);
            d["v_buy"] = trim_d(o_vb);
            d["v_sell"] = trim_d(o_vs);
            d["price_end"] = trim_d(o_px);
            d["warmup"] = o_warm;
            return d;
        },
        py::arg("ts"), py::arg("bid"), py::arg("ask"), py::arg("volume"),
        py::arg("bucket_volume"), py::arg("window") = 50,
        py::arg("dist") = BvcDist::Normal, py::arg("df") = 0.25,
        py::arg("granularity") = BvcGranularity::Bucket,
        py::arg("sigma_mode") = SigmaMode::Expanding,
        py::arg("sigma_fixed") = 0.0, py::arg("sigma_warmup") = 20,
        py::arg("pctile_bins") = 65536, py::arg("pctile_warmup") = 30,
        "Volume-bucket + BVC + rolling VPIN over a tick series.");

    m.def("compute_markouts",
        [](arr_i64 f_ts, arr_d f_price, arr_i8 f_side, arr_i64 m_ts, arr_d m_mid,
           arr_d horizons) {
            const std::size_t nf = static_cast<std::size_t>(f_ts.size());
            const std::size_t nm = static_cast<std::size_t>(m_ts.size());
            const std::size_t nh = static_cast<std::size_t>(horizons.size());
            require(f_price.size() == f_ts.size() && f_side.size() == f_ts.size(),
                    "compute_markouts: fill column length mismatch");
            require(m_mid.size() == m_ts.size(),
                    "compute_markouts: mid column length mismatch");

            FillView fv;
            fv.ts = static_cast<const nanos_t*>(f_ts.request().ptr);
            fv.price = static_cast<const double*>(f_price.request().ptr);
            fv.side = static_cast<const std::int8_t*>(f_side.request().ptr);
            fv.n = nf;

            MidSeries ms;
            ms.ts = static_cast<const nanos_t*>(m_ts.request().ptr);
            ms.mid = static_cast<const double*>(m_mid.request().ptr);
            ms.n = nm;

            arr_d out({nf, nh});
            {
                py::gil_scoped_release release;
                compute_markouts(fv, ms, static_cast<const double*>(horizons.request().ptr),
                                 nh, static_cast<double*>(out.request().ptr));
            }
            return out;
        },
        py::arg("fill_ts"), py::arg("fill_price"), py::arg("fill_side"),
        py::arg("mid_ts"), py::arg("mid"), py::arg("horizons_sec"),
        "LP markouts, [n_fills x n_horizons]. Uses the prevailing mid at t+tau.");

    m.def("mid_at_fill",
        [](arr_i64 f_ts, arr_i64 m_ts, arr_d m_mid) {
            const std::size_t nf = static_cast<std::size_t>(f_ts.size());
            FillView fv;
            fv.ts = static_cast<const nanos_t*>(f_ts.request().ptr);
            fv.n = nf;
            MidSeries ms;
            ms.ts = static_cast<const nanos_t*>(m_ts.request().ptr);
            ms.mid = static_cast<const double*>(m_mid.request().ptr);
            ms.n = static_cast<std::size_t>(m_ts.size());
            arr_d out(nf);
            {
                py::gil_scoped_release release;
                mid_at_fill(fv, ms, static_cast<double*>(out.request().ptr));
            }
            return out;
        },
        py::arg("fill_ts"), py::arg("mid_ts"), py::arg("mid"),
        "Prevailing mid at each fill timestamp.");

    m.def("label_fills",
        [](arr_i64 f_ts, arr_d f_price, arr_i8 f_side, arr_i64 m_ts, arr_d m_mid,
           LabelRule rule, double horizon_sec, double theta, double min_adverse,
           BarrierCenter center, py::object markouts) {
            const std::size_t nf = static_cast<std::size_t>(f_ts.size());
            require(f_price.size() == f_ts.size() && f_side.size() == f_ts.size(),
                    "label_fills: fill column length mismatch");
            require(m_mid.size() == m_ts.size(), "label_fills: mid column length mismatch");

            Keeper keep;
            const double* mk = keep.take<arr_d>(markouts, nf, "markouts");
            require(rule != LabelRule::MarkoutThreshold || mk != nullptr,
                    "label_fills: MARKOUT_THRESHOLD requires the markouts argument");

            FillView fv;
            fv.ts = static_cast<const nanos_t*>(f_ts.request().ptr);
            fv.price = static_cast<const double*>(f_price.request().ptr);
            fv.side = static_cast<const std::int8_t*>(f_side.request().ptr);
            fv.n = nf;

            MidSeries ms;
            ms.ts = static_cast<const nanos_t*>(m_ts.request().ptr);
            ms.mid = static_cast<const double*>(m_mid.request().ptr);
            ms.n = static_cast<std::size_t>(m_ts.size());

            LabelConfig cfg;
            cfg.rule = rule;
            cfg.horizon_sec = horizon_sec;
            cfg.theta = theta;
            cfg.min_adverse = min_adverse;
            cfg.center = center;

            arr_i8 out(nf);
            {
                py::gil_scoped_release release;
                BlockIndex index(ms.mid, ms.n);
                label_fills(fv, ms, index, cfg, mk,
                            static_cast<std::int8_t*>(out.request().ptr));
            }
            return out;
        },
        py::arg("fill_ts"), py::arg("fill_price"), py::arg("fill_side"),
        py::arg("mid_ts"), py::arg("mid"), py::arg("rule"),
        py::arg("horizon_sec") = 30.0, py::arg("theta") = 0.0,
        py::arg("min_adverse") = 0.0,
        py::arg("center") = BarrierCenter::MidAtFill,
        py::arg("markouts") = py::none(),
        "Ground-truth labels: 1 toxic, 0 benign, -1 undecidable.");

    m.def("build_features",
        [](arr_i64 t_ts, arr_d bid, arr_d ask, arr_i64 f_ts, arr_d f_size,
           py::object bid_size, py::object ask_size, py::object volume,
           double short_window_sec, double long_window_sec) {
            const std::size_t nt = static_cast<std::size_t>(t_ts.size());
            const std::size_t nf = static_cast<std::size_t>(f_ts.size());
            require(bid.size() == t_ts.size() && ask.size() == t_ts.size(),
                    "build_features: quote column length mismatch");
            require(f_size.size() == f_ts.size(),
                    "build_features: fill column length mismatch");

            Keeper keep;
            TickView tv;
            tv.ts = static_cast<const nanos_t*>(t_ts.request().ptr);
            tv.bid = static_cast<const double*>(bid.request().ptr);
            tv.ask = static_cast<const double*>(ask.request().ptr);
            tv.bid_size = keep.take<arr_d>(bid_size, nt, "bid_size");
            tv.ask_size = keep.take<arr_d>(ask_size, nt, "ask_size");
            tv.volume = keep.take<arr_d>(volume, nt, "volume");
            tv.n = nt;

            FillView fv;
            fv.ts = static_cast<const nanos_t*>(f_ts.request().ptr);
            fv.size = static_cast<const double*>(f_size.request().ptr);
            fv.n = nf;

            FeatureConfig cfg;
            cfg.short_window_sec = short_window_sec;
            cfg.long_window_sec = long_window_sec;

            arr_d out({nf, static_cast<std::size_t>(kFeatCount)});
            {
                py::gil_scoped_release release;
                build_features(tv, fv, cfg, static_cast<double*>(out.request().ptr));
            }
            return out;
        },
        py::arg("tick_ts"), py::arg("bid"), py::arg("ask"), py::arg("fill_ts"),
        py::arg("fill_size"), py::arg("bid_size") = py::none(),
        py::arg("ask_size") = py::none(), py::arg("volume") = py::none(),
        py::arg("short_window_sec") = 60.0, py::arg("long_window_sec") = 900.0,
        "Pre-trade feature matrix, [n_fills x FEATURE_COUNT].");

    m.def("counterparty_history",
        [](arr_i32 cp, arr_i8 labels, double prior_rate, double prior_weight) {
            const std::size_t n = static_cast<std::size_t>(cp.size());
            require(labels.size() == cp.size(),
                    "counterparty_history: length mismatch");

            // Only the counterparty column is read from the fills
            std::vector<nanos_t> dummy_ts(n, 0);
            std::vector<double> dummy_px(n, 0.0);
            std::vector<std::int8_t> dummy_side(n, 1);
            FillView fv;
            fv.ts = dummy_ts.data();
            fv.price = dummy_px.data();
            fv.side = dummy_side.data();
            fv.counterparty = static_cast<const std::int32_t*>(cp.request().ptr);
            fv.n = n;

            arr_d rate(n), count(n);
            {
                py::gil_scoped_release release;
                counterparty_history(fv, static_cast<const std::int8_t*>(labels.request().ptr),
                                     prior_rate, prior_weight,
                                     static_cast<double*>(rate.request().ptr),
                                     static_cast<double*>(count.request().ptr));
            }
            return py::make_tuple(rate, count);
        },
        py::arg("counterparty"), py::arg("labels"), py::arg("prior_rate") = 0.5,
        py::arg("prior_weight") = 5.0,
        "Causal per-counterparty toxic rate and prior fill count.");

    py::class_<Model>(m, "Model")
        .def_static("load", &Model::load, py::arg("path"))
        .def_property_readonly("n_features", &Model::n_features)
        .def_property_readonly("n_trees", &Model::n_trees)
        .def_property_readonly("feature_names", &Model::feature_names)
        .def("bind", [](const Model& self, const std::vector<std::string>& available) {
            std::vector<int> idx;
            std::string missing;
            if (!self.bind(available, idx, missing))
                throw std::invalid_argument(
                    "model needs a feature the caller does not provide: " + missing);
            return idx;
        }, py::arg("available"),
           "Positions of this model's inputs within `available`, in model order.")
        .def("predict_proba", [](const Model& self, arr_d X) {
            require(X.ndim() == 2, "predict_proba: expected a 2-D array");
            require(static_cast<std::uint32_t>(X.shape(1)) == self.n_features(),
                    "predict_proba: feature-count mismatch");
            const std::size_t n = static_cast<std::size_t>(X.shape(0));
            arr_d out(n);
            {
                py::gil_scoped_release release;
                self.predict_proba(static_cast<const double*>(X.request().ptr), n,
                                   static_cast<double*>(out.request().ptr));
            }
            return out;
        }, py::arg("X"));

    py::class_<GateConfig>(m, "GateConfig")
        .def(py::init<>())
        .def_readwrite("tighten_below", &GateConfig::tighten_below)
        .def_readwrite("widen_above", &GateConfig::widen_above)
        .def_readwrite("suspend_above", &GateConfig::suspend_above)
        .def_readwrite("hysteresis", &GateConfig::hysteresis)
        .def_readwrite("min_dwell_ns", &GateConfig::min_dwell_ns)
        .def_readwrite("tighten_factor", &GateConfig::tighten_factor)
        .def_readwrite("max_widen_factor", &GateConfig::max_widen_factor)
        .def_readwrite("alpha_mu", &GateConfig::alpha_mu)
        .def_readwrite("widen_factor", &GateConfig::widen_factor);

    py::class_<GateDecision>(m, "GateDecision")
        .def_readonly("action", &GateDecision::action)
        .def_readonly("spread", &GateDecision::spread)
        .def_readonly("multiplier", &GateDecision::multiplier)
        .def_readonly("changed", &GateDecision::changed)
        .def("__repr__", [](const GateDecision& d) {
            return std::string("<GateDecision ") + quote_action_name(d.action) +
                   " spread=" + std::to_string(d.spread) + ">";
        });

    py::class_<RiskGate>(m, "RiskGate")
        .def(py::init<const GateConfig&>(), py::arg("config"))
        .def("on_update", &RiskGate::on_update,
             py::arg("ts"), py::arg("toxicity_proba"), py::arg("base_spread"))
        .def_property_readonly("state", &RiskGate::state);

    py::class_<LiveFeatures>(m, "LiveFeatures")
        .def(py::init([](double short_window_sec, double long_window_sec) {
                 FeatureConfig c;
                 c.short_window_sec = short_window_sec;
                 c.long_window_sec = long_window_sec;
                 return new LiveFeatures(c);
             }),
             py::arg("short_window_sec") = 60.0, py::arg("long_window_sec") = 900.0)
        .def("on_tick", [](LiveFeatures& self, nanos_t ts, double bid, double ask,
                           double bid_size, double ask_size, double volume) {
                 self.on_tick(Tick{ts, bid, ask, bid_size, ask_size, volume});
             },
             py::arg("ts"), py::arg("bid"), py::arg("ask"), py::arg("bid_size") = 0.0,
             py::arg("ask_size") = 0.0, py::arg("volume") = 0.0)
        .def("snapshot", [](LiveFeatures& self, double size, py::object as_of) {
                 arr_d out(static_cast<std::size_t>(kFeatCount));
                 double* p = static_cast<double*>(out.request().ptr);
                 if (as_of.is_none()) self.snapshot(size, p);
                 else self.snapshot(size, as_of.cast<nanos_t>(), p);
                 return out;
             }, py::arg("size") = 0.0, py::arg("as_of") = py::none())
        .def_property_readonly("ready", &LiveFeatures::ready);
}
