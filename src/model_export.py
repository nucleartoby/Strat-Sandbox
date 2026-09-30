from pathlib import Path

from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline

import numpy as np
import native
import struct


MAGIC = b"FXTM"
VERSION = 2
KIND_LOGISTIC = 0
KIND_GBDT = 1
FLAG_STANDARDIZE = 1 << 0


def _unwrap(model):
    if isinstance(model, CalibratedClassifierCV):
        raise NotImplementedError(
            "CalibratedClassifierCV cannot be exported to .fxm: the sigmoid/isotonic "
            "calibration map is not part of the format, so the engine would run "
            "uncalibrated probabilities while your reports show calibrated ones. "
            "Either export the uncalibrated model, or apply the calibration on the "
            "C++ side and export its parameters alongside.")
    scaler = None
    if isinstance(model, Pipeline):
        scaler = model.named_steps.get("scale")
        model = model.named_steps["clf"]
    return model, scaler


def export_model(model, path, feature_names=None) -> dict:
    est, scaler = _unwrap(model)
    path = Path(path)

    if hasattr(est, "coef_"):
        info = _export_logistic(est, scaler, path, feature_names)
    elif hasattr(est, "booster_"):
        info = _export_lightgbm(est, scaler, path, feature_names)
    else:
        raise TypeError(
            f"cannot export {type(est).__name__}; expected LogisticRegression or LGBMClassifier")
    info["path"] = str(path)
    info["feature_names"] = list(feature_names) if feature_names is not None else None
    return info


def _write_header(fh, kind: int, n_features: int, flags: int, sigmoid_scale: float,
                  feature_names=None):
    fh.write(MAGIC)
    fh.write(struct.pack("<IIII", VERSION, kind, n_features, flags))
    fh.write(struct.pack("<d", sigmoid_scale))

    names = list(feature_names) if feature_names is not None else []
    if names and len(names) != n_features:
        raise ValueError(
            f"got {len(names)} feature names for {n_features} model inputs")
    fh.write(struct.pack("<I", len(names)))
    for name in names:
        encoded = str(name).encode("utf-8")
        fh.write(struct.pack("<I", len(encoded)))
        fh.write(encoded)


def _write_standardizer(fh, scaler, n_features: int):
    mean = np.asarray(scaler.mean_, dtype=np.float64)
    scale = np.asarray(scaler.scale_, dtype=np.float64)
    scale = np.where(scale > 0, scale, 1.0)
    fh.write(mean.tobytes())
    fh.write(scale.tobytes())


def _export_logistic(est, scaler, path: Path, feature_names=None) -> dict:
    w = np.ravel(est.coef_).astype(np.float64)
    bias = float(np.ravel(est.intercept_)[0])
    n_features = len(w)
    flags = FLAG_STANDARDIZE if scaler is not None else 0

    with open(path, "wb") as fh:
        _write_header(fh, KIND_LOGISTIC, n_features, flags, 1.0, feature_names)
        if scaler is not None:
            _write_standardizer(fh, scaler, n_features)
        fh.write(struct.pack("<d", bias))
        fh.write(w.tobytes())
    return {"kind": "logistic", "n_features": n_features, "standardized": scaler is not None}


def _export_lightgbm(est, scaler, path: Path, feature_names=None) -> dict:
    booster = est.booster_
    dump = booster.dump_model()
    n_features = int(dump["max_feature_idx"]) + 1
    if feature_names is not None and len(feature_names) != n_features:
        raise ValueError(
            f"model was trained on {n_features} columns but {len(feature_names)} "
            "names were supplied")
    trees = dump["tree_info"]

    sigmoid_scale = 1.0
    objective = dump.get("objective", "")
    if "sigmoid:" in objective:
        sigmoid_scale = float(objective.split("sigmoid:")[1].split()[0])

    if scaler is not None:
        raise ValueError(
            "a StandardScaler in front of a tree model changes every split threshold; "
            "train LightGBM on raw features (get_model('gbm') already does)")

    with open(path, "wb") as fh:
        _write_header(fh, KIND_GBDT, n_features, 0, sigmoid_scale, feature_names)
        fh.write(struct.pack("<d", 0.0))
        fh.write(struct.pack("<I", len(trees)))
        for tree in trees:
            nodes = _flatten_tree(tree["tree_structure"])
            fh.write(struct.pack("<I", len(nodes)))
            for feat, left, right, default_left, value in nodes:
                fh.write(struct.pack("<iiii", feat, left, right, default_left))
                fh.write(struct.pack("<d", value))

    return {"kind": "gbdt", "n_features": n_features, "n_trees": len(trees),
            "sigmoid_scale": sigmoid_scale}


def _flatten_tree(root) -> list:
    nodes: list = []

    def visit(node) -> int:
        idx = len(nodes)
        if "leaf_value" in node:
            nodes.append((-1, -1, -1, 1, float(node["leaf_value"])))
            return idx

        dt = node.get("decision_type", "<=")
        if dt != "<=":
            raise NotImplementedError(
                f"decision_type {dt!r} is not supported by the .fxm format; "
                "train without categorical features, or one-hot them first")
        default_left = 1 if node.get("default_left", True) else 0
        nodes.append((int(node["split_feature"]), -1, -1, default_left,
                      float(node["threshold"])))
        left = visit(node["left_child"])
        right = visit(node["right_child"])
        feat, _, _, dl, val = nodes[idx]
        nodes[idx] = (feat, left, right, dl, val)
        return idx

    visit(root)
    return nodes


def verify_export(model, path, X, atol: float = 1e-9) -> dict:
    if not native.HAVE_NATIVE:
        raise RuntimeError("fxtox_native is required to verify an export; build it with cmake")

    py_proba = np.asarray(model.predict_proba(np.asarray(X, dtype=np.float64))[:, 1])
    cpp_proba = np.asarray(native.load_model(path).predict_proba(
        np.ascontiguousarray(np.asarray(X, dtype=np.float64))))

    diff = np.abs(py_proba - cpp_proba)
    return {
        "max_abs_diff": float(diff.max()) if len(diff) else 0.0,
        "mean_abs_diff": float(diff.mean()) if len(diff) else 0.0,
        "n": len(diff),
        "matches": bool(len(diff) and diff.max() <= atol),
        "atol": atol,}
