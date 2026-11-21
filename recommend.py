"""P0-FR04 — bounded candidate search, risk flagging and ranking.

Candidates are only generated inside the intersection of the researcher's
bounds and the observed training range. Nothing is extrapolated silently, and
no candidate is ever described as a measured result.
"""

from __future__ import annotations

import pandas as pd

from common import LABEL_UNVERIFIED


def support_bounds(df, cfg: dict) -> tuple[dict, list]:
    bounds, problems = {}, []
    for var, spec in cfg["variables"].items():
        if spec.get("role") != "controllable":
            continue
        lo, hi = [float(x) for x in spec["bounds"]]
        series = pd.to_numeric(df[var], errors="coerce").dropna()
        if series.empty:
            problems.append(f"{var}: 数据中没有可用数值")
            continue
        dmin, dmax = float(series.min()), float(series.max())
        eff_lo, eff_hi = max(lo, dmin), min(hi, dmax)
        if eff_hi <= eff_lo:
            problems.append(
                f"{var}: 用户边界 [{lo}, {hi}] 与数据范围 [{dmin}, {dmax}] 无交集"
            )
        bounds[var] = {
            "unit": spec.get("unit"),
            "user_bounds": [lo, hi],
            "data_range": [dmin, dmax],
            "effective_bounds": [eff_lo, eff_hi],
        }
    return bounds, problems


def generate_candidates(cfg: dict, bounds: dict, seed: int, np) -> tuple[pd.DataFrame, str]:
    rec = cfg.get("recommendation") or {}
    budget = int(rec.get("candidate_budget", 200))
    levels = int(rec.get("grid_levels", 5))
    variables = list(bounds)
    if not variables:
        return pd.DataFrame(), "none"

    total = levels ** len(variables)
    if total <= budget:
        grids = [np.linspace(bounds[v]["effective_bounds"][0],
                             bounds[v]["effective_bounds"][1], levels) for v in variables]
        combos = np.array(np.meshgrid(*grids, indexing="ij")).reshape(len(variables), -1).T
        mode = "grid"
    else:
        rng = np.random.default_rng(seed)
        low = np.array([bounds[v]["effective_bounds"][0] for v in variables])
        high = np.array([bounds[v]["effective_bounds"][1] for v in variables])
        combos = rng.uniform(low, high, size=(budget, len(variables)))
        mode = "random"

    frame = pd.DataFrame(np.round(combos, 6), columns=variables)
    return frame.drop_duplicates().reset_index(drop=True), mode


def _direction(cfg: dict) -> tuple[str, float | None]:
    obj = cfg.get("objective") or {}
    return str(obj.get("direction", "maximize")).lower(), obj.get("target_value")


def score_and_rank(cand: pd.DataFrame, model, df, cfg: dict, feature_names: list,
                   bounds: dict, np) -> pd.DataFrame:
    direction, target_value = _direction(cfg)
    rec = cfg.get("recommendation") or {}
    top_k = int(rec.get("top_k", 3))

    for var, spec in cfg["variables"].items():
        if spec.get("role") == "context":
            cand[var] = float(spec["fixed_value"])

    X = cand[feature_names].astype(float)
    cand["predicted_target"] = model.predict(X)

    train_X = df[feature_names].astype(float)
    mu = train_X.mean()
    sd = train_X.std(ddof=0).replace(0, 1.0)
    Ztr = (train_X - mu) / sd
    Zc = (X - mu) / sd
    dist = np.sqrt(((Zc.to_numpy()[:, None, :] - Ztr.to_numpy()[None, :, :]) ** 2).sum(axis=2))
    cand["_nn_distance"] = dist.min(axis=1)

    y_train = pd.to_numeric(df[cfg["target_column"]], errors="coerce").dropna()
    y_min, y_max = float(y_train.min()), float(y_train.max())
    var_stats = {
        v: (float(pd.to_numeric(df[v], errors="coerce").quantile(0.05)),
            float(pd.to_numeric(df[v], errors="coerce").quantile(0.95)))
        for v in bounds
    }

    flags = []
    for _, row in cand.iterrows():
        f = []
        if row["_nn_distance"] > 1.5:
            f.append("SPARSE_NEIGHBORHOOD")
        if any(not (var_stats[v][0] <= row[v] <= var_stats[v][1]) for v in bounds):
            f.append("OUTSIDE_TRAIN_P05_P95")
        if row["predicted_target"] > y_max or row["predicted_target"] < y_min:
            f.append("PREDICTED_OUTSIDE_OBSERVED_RANGE")
        if len(y_train) < 50:
            f.append("SMALL_SAMPLE")
        flags.append(f)
    cand["risk_flags"] = [";".join(sorted(set(f))) or "NONE" for f in flags]

    if direction == "maximize":
        cand = cand.sort_values("predicted_target", ascending=False)
    elif direction == "minimize":
        cand = cand.sort_values("predicted_target", ascending=True)
    elif direction == "target":
        if target_value is None:
            raise ValueError("objective.direction=target 但缺少 objective.target_value")
        cand = cand.assign(_gap=(cand["predicted_target"] - float(target_value)).abs()) \
                   .sort_values("_gap")
    else:
        raise ValueError(f"objective.direction='{direction}' 不支持（maximize / minimize / target）")

    # Diversity-aware greedy selection in the normalised bound space.
    ranges = {v: max(bounds[v]["effective_bounds"][1] - bounds[v]["effective_bounds"][0], 1e-12)
              for v in bounds}
    selected, selected_rows = [], []
    for sep in (float(rec.get("min_separation", 0.15)), 0.05, 0.0):
        for _, row in cand.iterrows():
            if len(selected) >= top_k:
                break
            if row.name in selected:
                continue
            ok = True
            for _, prev in selected_rows:
                d = max(abs(row[v] - prev[v]) / ranges[v] for v in bounds)
                if d < sep:
                    ok = False
                    break
            if ok:
                selected.append(row.name)
                selected_rows.append((row.name, row))
        if len(selected) >= top_k:
            break

    picked = cand.loc[selected].copy()
    picked["rank"] = range(1, len(picked) + 1)
    picked["data_label"] = LABEL_UNVERIFIED
    picked["units"] = "; ".join(f"{v}:{bounds[v]['unit']}" for v in bounds)
    picked["reason"] = [
        _reason(r, bounds, var_stats, y_min, y_max, direction)
        for _, r in picked.iterrows()
    ]
    return picked.drop(columns=["_nn_distance", "_gap"], errors="ignore").reset_index(drop=True)


def _reason(row, bounds, var_stats, y_min, y_max, direction) -> str:
    parts = [f"predicted target {row['predicted_target']:.4g}"]
    if row["predicted_target"] > y_max:
        parts.append("above observed maximum (extrapolation)")
    elif row["predicted_target"] < y_min:
        parts.append("below observed minimum (extrapolation)")
    else:
        parts.append("within observed target range")
    outside = [v for v in bounds if not (var_stats[v][0] <= row[v] <= var_stats[v][1])]
    if outside:
        parts.append("outside central 90% of training data: " + ", ".join(outside))
    if row["risk_flags"] != "NONE":
        parts.append("risk: " + row["risk_flags"])
    parts.append(f"objective={direction}")
    return "; ".join(parts)
