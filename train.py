"""P0-FR03 — model training, validation and the quality gate.

All learned preprocessing lives inside a sklearn Pipeline, fitted on the
training split only. Baselines are `DummyRegressor(mean)` and
`LinearRegression`; `RandomForestRegressor` is added when the training set is
large enough.
"""

from __future__ import annotations

import math


def split_indices(df, cfg: dict, np):
    split = cfg.get("split") or {}
    method = str(split.get("method", "random")).lower()
    seed = int(split.get("seed", 42))
    test_size = float(split.get("test_size", 0.2))

    if method == "random":
        from sklearn.model_selection import train_test_split

        idx = np.arange(len(df))
        tr, te = train_test_split(idx, test_size=test_size, random_state=seed, shuffle=True)
        return tr, te, {
            "method": "random",
            "seed": seed,
            "test_size": test_size,
            "warning": None if "group_column" in split else
                       "未提供 group_column：如存在同批次/同论文重复测量，指标可能偏乐观",
        }

    if method == "group":
        col = split.get("group_column")
        if not col:
            raise ValueError("split.method=group 但缺少 split.group_column")
        if col not in df.columns:
            raise ValueError(f"split.group_column '{col}' 不在数据列中")
        from sklearn.model_selection import GroupShuffleSplit

        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        tr, te = next(gss.split(df, groups=df[col].astype(str).to_numpy()))
        return tr, te, {"method": "group", "group_column": col, "seed": seed,
                        "test_size": test_size, "warning": None}

    raise ValueError(f"split.method='{method}' 未实现（当前支持 random / group）")


def build_estimators(cfg: dict, n_train: int, seed: int, np) -> dict:
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.dummy import DummyRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    model_cfg = cfg.get("model") or {}
    transform = str(model_cfg.get("target_transform", "none")).lower()

    def wrap(inner):
        if transform in ("none", "", "identity"):
            return inner
        if transform == "log10":
            return TransformedTargetRegressor(regressor=inner, func=np.log10,
                                              inverse_func=lambda v: np.power(10.0, v))
        if transform == "log":
            return TransformedTargetRegressor(regressor=inner, func=np.log,
                                              inverse_func=np.exp)
        raise ValueError(f"model.target_transform='{transform}' 不支持（none / log / log10）")

    estimators = {"dummy_mean": DummyRegressor(strategy="mean")}
    estimators["linear"] = wrap(
        Pipeline([("scaler", StandardScaler()), ("reg", LinearRegression())])
    )
    if n_train >= int(model_cfg.get("rf_min_train", 50)):
        estimators["random_forest"] = wrap(
            Pipeline([
                ("scaler", StandardScaler()),
                ("reg", RandomForestRegressor(
                    n_estimators=int(model_cfg.get("rf_estimators", 300)),
                    min_samples_leaf=int(model_cfg.get("rf_min_samples_leaf", 2)),
                    random_state=seed, n_jobs=1,
                )),
            ])
        )
    return estimators


def _score(est, X, y, np) -> dict:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    pred = est.predict(X)
    mae = float(mean_absolute_error(y, pred))
    rmse = float(math.sqrt(mean_squared_error(y, pred)))
    r2 = float(r2_score(y, pred)) if len(y) >= 2 else float("nan")
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "pred": pred}


def evaluate(estimators: dict, Xtr, ytr, Xte, yte, seed: int, np) -> tuple[dict, dict]:
    from sklearn.model_selection import KFold, cross_val_score

    n_train = len(ytr)
    k = int(max(2, min(5, n_train // 10))) if n_train >= 30 else 0
    kf = KFold(n_splits=k, shuffle=True, random_state=seed) if k >= 2 else None

    results, predictions = {}, {}
    for name, est in estimators.items():
        est.fit(Xtr, ytr)
        tr = _score(est, Xtr, ytr, np)
        te = _score(est, Xte, yte, np)
        cv_mae = cv_sd = None
        if kf is not None:
            try:
                scores = cross_val_score(est, Xtr, ytr, cv=kf,
                                         scoring="neg_mean_absolute_error")
                cv_mae, cv_sd = float(-scores.mean()), float(scores.std())
            except Exception:  # noqa: BLE001 - CV is informative, not blocking
                cv_mae, cv_sd = None, None
        results[name] = {
            "n_train": int(n_train),
            "n_test": int(len(yte)),
            "cv_folds": k,
            "metrics_train": {"MAE": tr["MAE"], "RMSE": tr["RMSE"], "R2": tr["R2"]},
            "metrics_test": {"MAE": te["MAE"], "RMSE": te["RMSE"], "R2": te["R2"]},
            "cv_mae_mean": cv_mae,
            "cv_mae_std": cv_sd,
            "error_bias_test": float(np.mean(te["pred"] - yte)),
        }
        predictions[name] = {"train": tr["pred"], "test": te["pred"]}
    return results, predictions


def choose_best(results: dict) -> str:
    candidates = {k: v for k, v in results.items() if k != "dummy_mean"}
    finite = {k: v for k, v in candidates.items()
              if math.isfinite(v["metrics_test"]["MAE"])}
    if not finite:
        return "dummy_mean"
    return min(finite, key=lambda k: finite[k]["metrics_test"]["MAE"])


def quality_gate(best: str, results: dict, min_improvement_pct: float) -> tuple[bool, str]:
    if best == "dummy_mean":
        return False, "没有模型在测试集上优于均值基线"
    dummy_mae = results["dummy_mean"]["metrics_test"]["MAE"]
    best_mae = results[best]["metrics_test"]["MAE"]
    if not math.isfinite(best_mae) or dummy_mae <= 0:
        return False, "测试集 MAE 不可比较（非有限值或基线为 0）"
    improvement = (dummy_mae - best_mae) / dummy_mae * 100.0
    if improvement < min_improvement_pct:
        return False, (
            f"最佳模型 {best} 的测试 MAE {best_mae:.6g} 仅比均值基线 {dummy_mae:.6g} "
            f"好 {improvement:.2f}%，低于门槛 {min_improvement_pct:.1f}%"
        )
    return True, f"最佳模型 {best} 测试 MAE 比均值基线低 {improvement:.2f}%"
