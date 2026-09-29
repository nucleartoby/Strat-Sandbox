#pragma once

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace fxtox {

enum class ModelKind : std::uint32_t { Logistic = 0, Gbdt = 1 };

constexpr std::uint32_t kModelVersion = 2;
constexpr std::uint32_t kFlagStandardize = 1u << 0;

struct Node {
    std::int32_t feature = -1;
    std::int32_t left = -1;
    std::int32_t right = -1;
    std::int32_t default_left = 1;
    double value = 0.0;
};

inline double sigmoid(double x) noexcept {
    if (x >= 0.0) return 1.0 / (1.0 + std::exp(-x));
    const double e = std::exp(x);
    return e / (1.0 + e);
}

class Model {
  public:
    static Model load(const std::string& path) {
        std::FILE* f = std::fopen(path.c_str(), "rb");
        if (!f) throw std::runtime_error("fxtox::Model: cannot open " + path);
        try {
            Model m = read(f);
            std::fclose(f);
            return m;
        } catch (...) {
            std::fclose(f);
            throw;
        }
    }

    // Raw score before the link function.
    double raw_score(const double* x) const {
        if (kind_ == ModelKind::Logistic) {
            double s = bias_;
            for (std::uint32_t j = 0; j < n_features_; ++j) s += w_[j] * standardize(x, j);
            return s;
        }
        double s = base_score_;
        for (std::uint32_t t = 0; t < n_trees_; ++t) s += descend(t, x);
        return s;
    }

    double predict_proba(const double* x) const {
        return sigmoid(sigmoid_scale_ * raw_score(x));
    }

    void predict_proba(const double* X, std::size_t n, double* out) const {
        for (std::size_t i = 0; i < n; ++i) out[i] = predict_proba(X + i * n_features_);
    }

    std::uint32_t n_features() const noexcept { return n_features_; }
    ModelKind kind() const noexcept { return kind_; }
    std::uint32_t n_trees() const noexcept { return n_trees_; }
    const std::vector<std::string>& feature_names() const noexcept { return names_; }

    bool bind(const std::vector<std::string>& available,
              std::vector<int>& out_index, std::string& missing) const {
        out_index.clear();
        if (names_.empty()) {
            if (available.size() < n_features_) {
                missing = "<unnamed model expects " + std::to_string(n_features_) +
                          " features, only " + std::to_string(available.size()) + " available>";
                return false;
            }
            for (std::uint32_t j = 0; j < n_features_; ++j) out_index.push_back(static_cast<int>(j));
            return true;
        }
        for (const std::string& want : names_) {
            std::size_t k = 0;
            for (; k < available.size(); ++k) if (available[k] == want) break;
            if (k == available.size()) { missing = want; return false; }
            out_index.push_back(static_cast<int>(k));
        }
        return true;
    }

  private:
    double standardize(const double* x, std::uint32_t j) const {
        return standardize_ ? (x[j] - mean_[j]) / scale_[j] : x[j];
    }

    double descend(std::uint32_t tree, const double* x) const {
        std::int32_t i = static_cast<std::int32_t>(tree_offset_[tree]);
        const Node* nodes = nodes_.data();
        while (nodes[i].feature >= 0) {
            const Node& nd = nodes[i];
            const double v = x[nd.feature];
            const bool go_left = std::isnan(v) ? (nd.default_left != 0) : (v <= nd.value);
            i = go_left ? nd.left : nd.right;
        }
        return nodes[i].value;
    }

    template <typename T>
    static T get(std::FILE* f) {
        T v{};
        if (std::fread(&v, sizeof(T), 1, f) != 1)
            throw std::runtime_error("fxtox::Model: truncated model file");
        return v;
    }

    template <typename T>
    static void get_n(std::FILE* f, T* dst, std::size_t n) {
        if (n && std::fread(dst, sizeof(T), n, f) != n)
            throw std::runtime_error("fxtox::Model: truncated model file");
    }

    static Model read(std::FILE* f) {
        char magic[4];
        get_n(f, magic, 4);
        if (std::memcmp(magic, "FXTM", 4) != 0)
            throw std::runtime_error("fxtox::Model: bad magic (not an .fxm file)");

        const auto version = get<std::uint32_t>(f);
        if (version != kModelVersion)
            throw std::runtime_error("fxtox::Model: unsupported version " +
                                     std::to_string(version));

        Model m;
        m.kind_ = static_cast<ModelKind>(get<std::uint32_t>(f));
        m.n_features_ = get<std::uint32_t>(f);
        const auto flags = get<std::uint32_t>(f);
        m.sigmoid_scale_ = get<double>(f);
        m.standardize_ = (flags & kFlagStandardize) != 0;

        const auto n_names = get<std::uint32_t>(f);
        m.names_.reserve(n_names);
        for (std::uint32_t j = 0; j < n_names; ++j) {
            const auto len = get<std::uint32_t>(f);
            if (len > 4096) throw std::runtime_error("fxtox::Model: implausible name length");
            std::string name(len, '\0');
            get_n(f, &name[0], len);
            m.names_.push_back(std::move(name));
        }
        if (n_names && n_names != m.n_features_)
            throw std::runtime_error("fxtox::Model: name count does not match feature count");

        if (m.standardize_) {
            m.mean_.resize(m.n_features_);
            m.scale_.resize(m.n_features_);
            get_n(f, m.mean_.data(), m.n_features_);
            get_n(f, m.scale_.data(), m.n_features_);
            for (double& s : m.scale_) if (!(s > 0.0)) s = 1.0; // zero variance column
        }

        if (m.kind_ == ModelKind::Logistic) {
            m.bias_ = get<double>(f);
            m.w_.resize(m.n_features_);
            get_n(f, m.w_.data(), m.n_features_);
            return m;
        }

        m.base_score_ = get<double>(f);
        m.n_trees_ = get<std::uint32_t>(f);
        m.tree_offset_.reserve(m.n_trees_);
        for (std::uint32_t t = 0; t < m.n_trees_; ++t) {
            const auto n_nodes = get<std::uint32_t>(f);
            const std::size_t base = m.nodes_.size();
            m.tree_offset_.push_back(base);
            m.nodes_.resize(base + n_nodes);
            for (std::uint32_t k = 0; k < n_nodes; ++k) {
                Node& nd = m.nodes_[base + k];
                nd.feature = get<std::int32_t>(f);
                nd.left = get<std::int32_t>(f);
                nd.right = get<std::int32_t>(f);
                nd.default_left = get<std::int32_t>(f);
                nd.value = get<double>(f);
                if (nd.feature >= 0) {
                    nd.left += static_cast<std::int32_t>(base);
                    nd.right += static_cast<std::int32_t>(base);
                }
            }
        }
        return m;
    }

    ModelKind kind_ = ModelKind::Logistic;
    std::uint32_t n_features_ = 0;
    std::uint32_t n_trees_ = 0;
    bool standardize_ = false;
    double sigmoid_scale_ = 1.0;
    double bias_ = 0.0;
    double base_score_ = 0.0;
    std::vector<std::string> names_;
    std::vector<double> mean_, scale_, w_;
    std::vector<Node> nodes_;
    std::vector<std::size_t> tree_offset_;
};

} // namespace fxtox
