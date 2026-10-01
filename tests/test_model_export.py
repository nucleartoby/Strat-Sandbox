"""The exported .fxm must reproduce the Python model exactly."""
import os

import numpy as np
import pytest

import native
from model_export import export_model, verify_export
from modeling import HAVE_LGBM, get_model

pytestmark = pytest.mark.skipif(
    not native.HAVE_NATIVE, reason="fxtox_native is required to load .fxm models")


@pytest.fixture(scope="module")
def dataset():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(1500, 13))
    logit = X[:, 0] + 0.6 * X[:, 3] - 0.4 * X[:, 7]
    y = (logit + rng.normal(0, 0.5, len(X)) > 0).astype(int)
    return X, y


def test_logistic_round_trip(dataset, tmp_path):
    X, y = dataset
    model = get_model("logreg")
    model.fit(X, y)
    path = tmp_path / "logreg.fxm"

    info = export_model(model, path)
    assert info["kind"] == "logistic"
    assert info["standardized"] is True  # the pipeline's scaler must be carried

    check = verify_export(model, path, X)
    assert check["matches"], f"max diff {check['max_abs_diff']:.3e}"
    assert check["max_abs_diff"] < 1e-12


@pytest.mark.skipif(not HAVE_LGBM, reason="LightGBM unavailable (needs libomp on macOS)")
def test_gbdt_round_trip(dataset, tmp_path):
    X, y = dataset
    model = get_model("gbm", n_estimators=60)
    model.fit(X, y)
    path = tmp_path / "gbm.fxm"

    info = export_model(model, path)
    assert info["kind"] == "gbdt"
    assert info["n_trees"] == 60

    check = verify_export(model, path, X)
    assert check["matches"], f"max diff {check['max_abs_diff']:.3e}"


@pytest.mark.skipif(not HAVE_LGBM, reason="LightGBM unavailable")
def test_gbdt_handles_missing_values(dataset, tmp_path):
    """NaN must follow LightGBM's default_left, not crash or silently go right."""
    X, y = dataset
    model = get_model("gbm", n_estimators=40)
    model.fit(X, y)
    path = tmp_path / "gbm_nan.fxm"
    export_model(model, path)

    X_nan = X[:200].copy()
    X_nan[::5, 2] = np.nan
    check = verify_export(model, path, X_nan)
    assert check["matches"], f"NaN routing diverges: max diff {check['max_abs_diff']:.3e}"


def test_refuses_to_export_a_calibrated_wrapper(dataset, tmp_path):
    """Exporting the inner estimator would ship uncalibrated probabilities."""
    from sklearn.calibration import CalibratedClassifierCV
    X, y = dataset
    model = CalibratedClassifierCV(get_model("logreg"), method="sigmoid", cv=3)
    model.fit(X, y)
    with pytest.raises(NotImplementedError, match="calibration"):
        export_model(model, tmp_path / "bad.fxm")


def test_rejects_a_corrupt_file(tmp_path):
    path = tmp_path / "garbage.fxm"
    path.write_bytes(b"NOPE" + b"\x00" * 64)
    with pytest.raises(Exception):
        native.load_model(path)


def test_feature_count_matches_the_engine(dataset, tmp_path):
    """A model exported against a different feature set must be detectable."""
    X, y = dataset
    model = get_model("logreg")
    model.fit(X, y)
    path = tmp_path / "m.fxm"
    export_model(model, path)
    loaded = native.load_model(path)
    assert loaded.n_features == X.shape[1] == native.FEATURE_COUNT


def test_feature_names_round_trip(dataset, tmp_path):
    """Names must survive export so the engine can bind by name, not position."""
    X, y = dataset
    names = [f"feat_{i}" for i in range(X.shape[1])]
    model = get_model("logreg")
    model.fit(X, y)
    path = tmp_path / "named.fxm"
    export_model(model, path, feature_names=names)

    loaded = native.load_model(path)
    assert list(loaded.feature_names) == names

    # Binding tolerates extra features and any ordering.
    available = ["unrelated"] + list(reversed(names))
    idx = loaded.bind(available)
    assert [available[i] for i in idx] == names


def test_binding_refuses_a_missing_feature(dataset, tmp_path):
    """Positional binding would silently feed the model the wrong columns."""
    X, y = dataset
    model = get_model("logreg")
    model.fit(X, y)
    path = tmp_path / "named2.fxm"
    export_model(model, path, feature_names=[f"f{i}" for i in range(X.shape[1])])

    loaded = native.load_model(path)
    with pytest.raises(Exception, match="f12"):
        loaded.bind([f"f{i}" for i in range(X.shape[1] - 1)])


def test_rejects_mismatched_name_count(dataset, tmp_path):
    X, y = dataset
    model = get_model("logreg")
    model.fit(X, y)
    with pytest.raises(ValueError, match="names"):
        export_model(model, tmp_path / "bad.fxm", feature_names=["only_one"])
