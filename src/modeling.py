import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, log_loss, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier


def get_model(kind: str = "logreg", **overrides):
    if kind == "gbm":
        params = dict(n_estimators=120, num_leaves=7, max_depth=3,
                      learning_rate=0.05, min_child_samples=60,
                      subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                      reg_lambda=1.0, verbosity=-1, random_state=0)
        params.update(overrides)
        return LGBMClassifier(**params)
    params = dict(max_iter=2000, C=1.0)
    params.update(overrides)
    return Pipeline([("scale", StandardScaler()),
                     ("clf", LogisticRegression(**params))])


def evaluate(y_true, proba) -> dict:
    y_true = np.asarray(y_true)
    proba = np.asarray(proba, dtype=float)
    out = {
        "n": len(y_true),
        "base_rate": float(y_true.mean()) if len(y_true) else np.nan,
        "brier": np.nan, "auc": np.nan, "avg_precision": np.nan, "log_loss": np.nan,}
    
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return out
    
    out["auc"] = float(roc_auc_score(y_true, proba))
    out["avg_precision"] = float(average_precision_score(y_true, proba))
    out["brier"] = float(brier_score_loss(y_true, proba))
    out["log_loss"] = float(log_loss(y_true, np.clip(proba, 1e-9, 1 - 1e-9)))
    base = float(y_true.mean())
    out["brier_baseline"] = float(np.mean((y_true - base) ** 2))
    out["brier_skill"] = 1.0 - out["brier"] / out["brier_baseline"] if out["brier_baseline"] > 0 else np.nan
    return out


def cross_validate(X: pd.DataFrame, y: pd.Series, splits, kind: str = "logreg",
                   calibrate: bool = False, **model_kwargs) -> dict:
    splits = list(splits)
    if not splits:
        raise ValueError("no CV splits supplied")

    rows, oof = [], pd.Series(np.nan, index=X.index, dtype=float)
    for i, (train_idx, test_idx) in enumerate(splits):
        y_train = y.iloc[train_idx]
        if y_train.nunique() < 2:
            rows.append({"fold": i, "n": len(test_idx), "note": "single-class train fold"})
            continue

        model = get_model(kind, **model_kwargs)
        if calibrate:
            model = CalibratedClassifierCV(model, method="sigmoid", cv=3)
        model.fit(X.iloc[train_idx], y_train)

        proba = model.predict_proba(X.iloc[test_idx])[:, 1]
        oof.iloc[test_idx] = proba
        row = {"fold": i, "n_train": len(train_idx)}
        row.update(evaluate(y.iloc[test_idx], proba))
        rows.append(row)

    final = get_model(kind, **model_kwargs)
    if calibrate:
        final = CalibratedClassifierCV(final, method="sigmoid", cv=3)
    final.fit(X, y)

    return {
        "fold_results": pd.DataFrame(rows),
        "model": final,
        "oof_proba": oof,
        "oof_metrics": evaluate(y[oof.notna()], oof[oof.notna()]) if oof.notna().any() else {},
        "kind": kind,
        "feature_names": list(X.columns),}


def univariate_scores(X: pd.DataFrame, y: pd.Series, splits, kind: str = "logreg",
                     **model_kwargs) -> pd.DataFrame:
    splits = list(splits)
    rows = []
    for col in X.columns:
        res = cross_validate(X[[col]], y, splits, kind=kind, **model_kwargs)
        fr = res["fold_results"]
        rows.append({
            "feature": col,
            "auc": fr["auc"].mean() if "auc" in fr else np.nan,
            "brier_skill": fr["brier_skill"].mean() if "brier_skill" in fr else np.nan,
            "auc_min": fr["auc"].min() if "auc" in fr else np.nan,})
    return pd.DataFrame(rows).sort_values("brier_skill", ascending=False).reset_index(drop=True)


def forward_select(X: pd.DataFrame, y: pd.Series, splits, kind: str = "logreg",
                   metric: str = "brier_skill", max_features: int = 8,
                   min_gain: float = 0.002, min_score: float = 0.0,
                   verbose: bool = False, **model_kwargs) -> dict:
    splits = list(splits)
    remaining = list(X.columns)
    chosen: list = []
    best_score = -np.inf
    history = []

    while remaining and len(chosen) < max_features:
        scored = []
        for col in remaining:
            res = cross_validate(X[chosen + [col]], y, splits, kind=kind, **model_kwargs)
            fr = res["fold_results"]
            score = fr[metric].mean() if metric in fr else np.nan
            scored.append((score, col))
        scored = [(s, c) for s, c in scored if not np.isnan(s)]
        if not scored:
            break

        score, col = max(scored, key=lambda t: t[0])
        if score < min_score:
            if verbose:
                print(f"  stop: best remaining feature {col!r} scores {score:+.4f}, "
                      f"below the {min_score:+.4f} floor")
            break
        gain = score - best_score if np.isfinite(best_score) else np.inf
        if gain < min_gain:
            if verbose:
                print(f"  stop: best remaining feature {col!r} gains only {gain:+.4f}")
            break

        chosen.append(col)
        remaining.remove(col)
        best_score = score
        history.append({"n_features": len(chosen), "added": col, metric: score})
        if verbose:
            print(f"  + {col:24s} {metric}={score:+.4f}")

    return {
        "selected": chosen,
        "score": best_score,
        "metric": metric,
        "history": pd.DataFrame(history),
        "dropped": remaining,}


def reliability_curve(y_true, proba, n_bins: int = 10) -> pd.DataFrame:
    y_true = np.asarray(y_true)
    proba = np.asarray(proba, dtype=float)
    ok = ~np.isnan(proba)
    y_true, proba = y_true[ok], proba[ok]
    if len(np.unique(y_true)) < 2:
        return pd.DataFrame(columns=["mean_predicted", "fraction_positive", "count"])

    frac_pos, mean_pred = calibration_curve(y_true, proba, n_bins=n_bins, strategy="quantile")
    edges = np.quantile(proba, np.linspace(0, 1, n_bins + 1))
    counts = np.histogram(proba, bins=np.unique(edges))[0]
    return pd.DataFrame({
        "mean_predicted": mean_pred,
        "fraction_positive": frac_pos,
        "count": counts[: len(mean_pred)],
        "calibration_error": np.abs(mean_pred - frac_pos),})


def feature_importance(result: dict) -> pd.DataFrame:
    model, names = result["model"], result["feature_names"]
    est = model
    if isinstance(est, CalibratedClassifierCV):
        est = est.calibrated_classifiers_[0].estimator
    if isinstance(est, Pipeline):
        est = est.named_steps["clf"]

    if hasattr(est, "coef_"):
        vals = np.ravel(est.coef_)
        col = "coefficient"
    elif hasattr(est, "feature_importances_"):
        vals = np.asarray(est.feature_importances_, dtype=float)
        col = "gain"
    else:
        return pd.DataFrame(columns=["feature", "importance"])

    out = pd.DataFrame({"feature": names[: len(vals)], col: vals})
    return out.reindex(out[col].abs().sort_values(ascending=False).index).reset_index(drop=True)


def summarize(result: dict) -> pd.DataFrame:
    df = result["fold_results"]
    cols = [c for c in ("auc", "brier", "brier_skill", "avg_precision", "log_loss")
            if c in df]
    if not cols:
        return pd.DataFrame()
    return pd.DataFrame({
        "metric": cols,
        "mean": [df[c].mean() for c in cols],
        "std": [df[c].std() for c in cols],
        "min": [df[c].min() for c in cols],
        "max": [df[c].max() for c in cols],})
