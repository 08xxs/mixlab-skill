#!/usr/bin/env python3
"""PRD V2.0 §8.1 layer 3 — replay validation for candidate recommendations.

Design
------
For each replay fold:
  1. split the historical data into a training pool and a held-out pool
     (group split by paper/DOI when configured, to avoid source leakage);
  2. choose a model by cross-validation **on the training pool only**;
  3. restrict the held-out pool to the researcher's bounds, so a "recommendation"
     can never point outside the allowed region;
  4. simulate one "next round": pick k conditions with
       - the model (same ranking + diversity rule as recommend.py),
       - random sampling (200 seeded draws -> distribution),
       - "go near the best training run" (nearest-neighbour heuristic);
  5. score each strategy by the **measured** target of the selected held-out rows.

The oracle (true best k in the pool) is reported only as an upper bound.

Exit codes: 0 completed, 1 runtime error, 2 invalid input/config.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import train as train_mod
from common import (C_ACCENT, C_GREY, C_PRIMARY, now_utc, read_table,
                    save_fig, sha256_file, style_axes, write_json)

SKILL_NAME = "mixlab-integration-check"
SKILL_VERSION = "0.3.0"
RANDOM_DRAWS = 200


def make_folds(df, cfg: dict, n_repeats: int, seed: int):
    split = cfg.get("split") or {}
    test_size = float(split.get("test_size", 0.25))
    gcol = split.get("group_column")
    folds = []
    for r in range(n_repeats):
        s = seed + r
        if gcol:
            from sklearn.model_selection import GroupShuffleSplit

            gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=s)
            tr, te = next(gss.split(df, groups=df[gcol].astype(str).to_numpy()))
        else:
            from sklearn.model_selection import train_test_split

            tr, te = train_test_split(np.arange(len(df)), test_size=test_size,
                                      random_state=s, shuffle=True)
        folds.append((tr, te, s))
    return folds


def fit_best_on_train(train_df, feature_names, target, cfg, seed):
    """Model selection uses CV on the training pool only (no test leakage)."""
    from sklearn.model_selection import KFold, cross_val_score

    estimators = train_mod.build_estimators(cfg, len(train_df), seed, np)
    estimators.pop("dummy_mean", None)
    X, y = train_df[feature_names], train_df[target].to_numpy(dtype=float)
    k = int(max(2, min(5, len(train_df) // 20)))
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)

    best_name, best_mae, best_est = None, float("inf"), None
    for name, est in estimators.items():
        try:
            scores = cross_val_score(est, X, y, cv=kf, scoring="neg_mean_absolute_error")
            mae = float(-scores.mean())
        except Exception:  # noqa: BLE001
            continue
        if mae < best_mae:
            best_name, best_mae, best_est = name, mae, est
    if best_est is None:
        return None, None, None
    best_est.fit(X, y)
    return best_name, best_est, best_mae


def select_diverse(pool: pd.DataFrame, score_col: str, k: int, bounds: dict,
                   min_sep: float) -> list:
    ordered = pool.sort_values(score_col, ascending=False)
    ranges = {v: max(b - a, 1e-12) for v, (a, b) in bounds.items()}
    chosen = []
    for sep in (min_sep, 0.05, 0.0):
        for idx, row in ordered.iterrows():
            if len(chosen) >= k:
                break
            if any(idx == c for c, _ in chosen):
                continue
            ok = True
            for _, prev in chosen:
                if max(abs(row[v] - prev[v]) / ranges[v] for v in bounds) < sep:
                    ok = False
                    break
            if ok:
                chosen.append((idx, row))
        if len(chosen) >= k:
            break
    return [c for c, _ in chosen]


def main() -> int:
    ap = argparse.ArgumentParser(description="Replay validation of candidate recommendations")
    ap.add_argument("--input", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--outdir", default="outputs/replay")
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    started = time.perf_counter()
    outdir = Path(args.outdir).resolve()
    (outdir / "figures").mkdir(parents=True, exist_ok=True)
    status, exit_code, reason = "FAILED_RUNTIME", 1, None

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        input_path = Path(args.input).resolve()
        config_path = Path(args.config).resolve()
        if not input_path.exists():
            status, exit_code, reason = "BLOCKED_INPUT", 2, f"输入文件不存在: {input_path}"
            return _finish(args, outdir, status, exit_code, reason, {}, started)
        if not config_path.exists():
            status, exit_code, reason = "BLOCKED_INPUT", 2, f"配置文件不存在: {config_path}"
            return _finish(args, outdir, status, exit_code, reason, {}, started)

        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        target = cfg["target_column"]
        variables = cfg["variables"]
        controllable = {v: s for v, s in variables.items() if s.get("role") == "controllable"}
        if not controllable:
            status, exit_code, reason = "BLOCKED_INPUT", 2, "没有 controllable 变量"
            return _finish(args, outdir, status, exit_code, reason, {}, started)

        df, io_meta = read_table(input_path, pd)
        feature_names = list(variables)
        work = df[feature_names + [target]].copy()
        work[target] = pd.to_numeric(work[target], errors="coerce")
        for c in feature_names:
            work[c] = pd.to_numeric(work[c], errors="coerce")
        gcol = (cfg.get("split") or {}).get("group_column")
        if gcol and gcol in df.columns:
            work[gcol] = df[gcol]
        work = work.dropna(subset=feature_names + [target])

        bounds = {v: tuple(map(float, s["bounds"])) for v, s in controllable.items()}
        seed = int((cfg.get("split") or {}).get("seed", 42))
        min_sep = float((cfg.get("recommendation") or {}).get("min_separation", 0.15))
        k = int(args.top_k)
        rng_master = np.random.default_rng(seed)

        folds = make_folds(work, cfg, args.repeats, seed)
        records, skipped = [], 0
        for fi, (tr, te, fold_seed) in enumerate(folds, start=1):
            train_df = work.iloc[tr]
            pool = work.iloc[te]
            # keep only held-out conditions inside the researcher's bounds
            mask = np.ones(len(pool), dtype=bool)
            for v, (lo, hi) in bounds.items():
                mask &= (pool[v] >= lo).to_numpy() & (pool[v] <= hi).to_numpy()
            pool = pool[mask]
            if len(pool) < max(k, 5) or len(train_df) < 20:
                skipped += 1
                continue

            best_name, model, cv_mae = fit_best_on_train(train_df, feature_names, target, cfg, fold_seed)
            if model is None:
                skipped += 1
                continue

            pred = model.predict(pool[feature_names])
            scored = pool.assign(_pred=pred)
            model_idx = select_diverse(scored, "_pred", k, bounds, min_sep)
            model_mean = float(pool.loc[model_idx, target].mean())

            draws = np.array([
                float(pool.loc[rng_master.choice(pool.index.to_numpy(), size=k, replace=False), target].mean())
                for _ in range(RANDOM_DRAWS)
            ])
            rand_mean = float(draws.mean())
            rand_p95 = float(np.quantile(draws, 0.95))

            mu = train_df[feature_names].mean()
            sd = train_df[feature_names].std(ddof=0).replace(0, 1.0)
            best_row = train_df.loc[train_df[target].idxmax()]
            best_vec = best_row[feature_names].astype(float).to_numpy()
            pool_mat = ((pool[feature_names].astype(float) - mu) / sd).to_numpy(dtype=float)
            best_z = (best_vec - mu.to_numpy(dtype=float)) / sd.to_numpy(dtype=float)
            d = pd.Series(np.sqrt(((pool_mat - best_z) ** 2).sum(axis=1)), index=pool.index)
            nn_idx = d.nsmallest(k).index
            nn_mean = float(pool.loc[nn_idx, target].mean())

            oracle_mean = float(pool[target].nlargest(k).mean())

            records.append({
                "fold": fi,
                "seed": fold_seed,
                "n_train": int(len(train_df)),
                "n_pool": int(len(pool)),
                "model": best_name,
                "cv_mae": round(cv_mae, 6),
                "selected_conditions": [
                    {v: round(float(pool.loc[i, v]), 4) for v in bounds} for i in model_idx
                ],
                "model_selected_measured_mean": round(model_mean, 6),
                "random_mean": round(rand_mean, 6),
                "random_p95": round(rand_p95, 6),
                "nn_best_run_mean": round(nn_mean, 6),
                "oracle_topk_mean": round(oracle_mean, 6),
                "model_percentile_vs_random": round(float((draws < model_mean).mean() * 100), 2),
                "improvement_vs_random_pct": round((model_mean - rand_mean) / rand_mean * 100, 3) if rand_mean else None,
                "beats_random_p95": bool(model_mean > rand_p95),
            })

        if not records:
            status, exit_code = "BLOCKED_DATA", 3
            reason = "没有足够的回放折次（池内样本过少或被边界过滤）"
            return _finish(args, outdir, status, exit_code, reason, {"skipped_folds": skipped}, started)

        imp = [r["improvement_vs_random_pct"] for r in records if r["improvement_vs_random_pct"] is not None]
        mean_imp = float(np.mean(imp)) if imp else float("nan")
        med_imp = float(np.median(imp)) if imp else float("nan")
        win_rate = float(np.mean([r["model_selected_measured_mean"] > r["random_mean"] for r in records]) * 100)
        beat_p95 = float(np.mean([r["beats_random_p95"] for r in records]) * 100)
        med_pct = float(np.median([r["model_percentile_vs_random"] for r in records]))
        nn_imp = float(np.mean([
            (r["nn_best_run_mean"] - r["random_mean"]) / r["random_mean"] * 100
            for r in records if r["random_mean"]
        ]))
        ratios = [r["model_selected_measured_mean"] / r["oracle_topk_mean"]
                  for r in records if r["oracle_topk_mean"]]
        med_ratio = float(np.median(ratios)) if ratios else float("nan")

        if med_imp > 0 and win_rate >= 70 and beat_p95 >= 50:
            verdict = "EVIDENCE_OF_ADVANTAGE"
        elif med_imp > 0 and win_rate >= 50:
            verdict = "WEAK_OR_INCONCLUSIVE"
        else:
            verdict = "NO_EVIDENCE_OVER_RANDOM"

        summary = {
            "skill": {"name": SKILL_NAME, "version": SKILL_VERSION},
            "generated_at_utc": now_utc(),
            "input": {"path": str(input_path), "sha256": sha256_file(input_path),
                      "size_bytes": input_path.stat().st_size},
            "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
            "data_provenance": cfg.get("data_provenance"),
            "setup": {
                "repeats_requested": args.repeats,
                "folds_used": len(records),
                "skipped_folds": skipped,
                "top_k": k,
                "random_draws_per_fold": RANDOM_DRAWS,
                "split_method": (cfg.get("split") or {}).get("method"),
                "group_column": gcol,
                "bounds": {v: list(b) for v, b in bounds.items()},
                "target": target,
                "objective": cfg.get("objective"),
                "seed": seed,
            },
            "aggregate": {
                "mean_improvement_vs_random_pct": round(mean_imp, 3),
                "median_improvement_vs_random_pct": round(med_imp, 3),
                "median_model_percentile_vs_random": round(med_pct, 2),
                "win_rate_vs_random_mean_pct": round(win_rate, 2),
                "beat_random_p95_pct": round(beat_p95, 2),
                "nn_heuristic_mean_improvement_pct": round(nn_imp, 3),
                "model_vs_oracle_median_ratio": round(med_ratio, 3),
            },
            "verdict": verdict,
            "interpretation": _interpret(verdict, mean_imp, med_imp, beat_p95, win_rate, med_ratio),
            "folds": records,
            "caveat": ("回放只检验「在历史条件里挑选」的能力，不等于前瞻性实验价值；"
                       "留出池由历史记录构成，模型未见过的参数组合不在检验范围内。"),
        }
        write_json(summary, outdir / "replay_validation.json")

        _plot(records, outdir / "figures" / "replay_comparison.png", plt)
        (outdir / "replay_report.md").write_text(_report(summary), encoding="utf-8")

        manifest = {
            "skill": {"name": SKILL_NAME, "version": SKILL_VERSION},
            "command": " ".join([Path(sys.executable).name] + sys.argv),
            "argv": sys.argv, "cwd": str(Path.cwd()),
            "duration_seconds": round(time.perf_counter() - started, 3),
            "python": {"executable": sys.executable, "version": sys.version.split()[0]},
            "status": "OK", "exit_code": 0,
            "outputs": [
                {"name": p.relative_to(outdir).as_posix(), "size_bytes": p.stat().st_size,
                 "sha256": sha256_file(p)}
                for p in sorted(outdir.rglob("*")) if p.is_file()
            ],
        }
        write_json(manifest, outdir / "run_manifest.json")

        status, exit_code = "OK", 0
        print(json.dumps({"status": status, "exit_code": 0, "verdict": verdict,
                          "folds": len(records), "summary": summary["aggregate"]},
                         ensure_ascii=False))
        return 0

    except Exception as exc:  # noqa: BLE001
        status, exit_code, reason = "FAILED_RUNTIME", 1, f"{type(exc).__name__}: {exc}"
        tb = traceback.format_exc().strip().splitlines()[-8:]
        print(json.dumps({"status": status, "exit_code": exit_code,
                          "error": reason, "traceback": tb}, ensure_ascii=False))
        return 1


def _interpret(verdict: str, mean_imp: float, med_imp: float, beat_p95: float,
               win_rate: float, med_ratio: float) -> str:
    tail = ("目标分布强右偏，均值提升会被少数极大值放大，因此以中位数为主要判据。")
    if verdict == "EVIDENCE_OF_ADVANTAGE":
        return (f"模型推荐的中位提升为 {med_imp:.2f}%（均值 {mean_imp:.2f}%），"
                f"{win_rate:.0f}% 的折次高于随机均值、{beat_p95:.0f}% 超过随机分布的 95 分位；"
                f"模型捕获到 oracle 上界的中位比例为 {med_ratio:.2f}。{tail}")
    if verdict == "WEAK_OR_INCONCLUSIVE":
        return (f"模型推荐中位提升 {med_imp:.2f}%（均值 {mean_imp:.2f}%），方向为正但"
                f"仅 {beat_p95:.0f}% 折次超过随机 95 分位（{win_rate:.0f}% 高于随机均值），"
                f"证据不足以宣称有效。{tail}")
    return (f"模型推荐中位提升 {med_imp:.2f}%（均值 {mean_imp:.2f}%），"
            f"仅 {beat_p95:.0f}% 折次超过随机 95 分位。没有证据表明推荐优于随机挑选。{tail}")


def _plot(records, out_png, plt) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.3), dpi=160)
    x = np.arange(1, len(records) + 1)

    ax1.plot(x, [r["random_mean"] for r in records], "o", color=C_GREY, ms=5,
             label="random selection (mean of 200 draws)")
    ax1.plot(x, [r["random_p95"] for r in records], "_", color=C_GREY, ms=9,
             label="random 95th percentile")
    ax1.plot(x, [r["model_selected_measured_mean"] for r in records], "o", color=C_PRIMARY,
             ms=6, label="model recommendation")
    ax1.plot(x, [r["nn_best_run_mean"] for r in records], "^", color=C_ACCENT, ms=5,
             alpha=0.85, label="nearest-to-best-run heuristic")
    ax1.set_xlabel("Replay fold")
    ax1.set_ylabel("Measured target of the selected k")
    ax1.set_title("Measured outcome per replay", fontsize=10)
    ax1.legend(fontsize=7.5, frameon=False, loc="best")
    style_axes(ax1)

    deltas = [r["model_selected_measured_mean"] - r["random_mean"] for r in records]
    colors = [C_PRIMARY if d > 0 else C_ACCENT for d in deltas]
    ax2.bar(x, deltas, color=colors, width=0.7)
    ax2.axhline(0, color=C_GREY, linewidth=1.0)
    ax2.set_xlabel("Replay fold")
    ax2.set_ylabel("Model − random (measured)")
    ax2.set_title("Difference from random selection", fontsize=10)
    style_axes(ax2)

    fig.suptitle("Replay validation of candidate recommendations", fontsize=11)
    save_fig(fig, out_png, plt)


def _report(s: dict) -> str:
    agg, setup = s["aggregate"], s["setup"]
    rows = "\n".join(
        f"| {r['fold']} | {r['model']} | {r['n_train']}/{r['n_pool']} | "
        f"{r['model_selected_measured_mean']:.5g} | {r['random_mean']:.5g} | "
        f"{r['random_p95']:.5g} | {r['nn_best_run_mean']:.5g} | "
        f"{r['improvement_vs_random_pct']:+.2f}% | {r['model_percentile_vs_random']:.0f} |"
        for r in s["folds"]
    )
    return f"""# 回放验证报告（推荐有效性，PRD §8.1 第 3 层）

> 数据标签：**{(s.get('data_provenance') or {}).get('type')}**
> 结论仅限「在历史留出条件中选择」的能力，不等于前瞻性实验价值。

## 1. 设置

| 项 | 值 |
| --- | --- |
| 目标列 / 方向 | `{setup['target']}` / {setup['objective']} |
| 每轮预算 k | {setup['top_k']} |
| 回放折次（实际使用/跳过） | {setup['folds_used']} / {setup['skipped_folds']} |
| 划分方式 | {setup['split_method']}（group_column = {setup['group_column']}） |
| 随机基线抽样次数/折 | {setup['random_draws_per_fold']} |
| 种子 | {setup['seed']} |
| 有效边界 | {json.dumps(setup['bounds'], ensure_ascii=False)} |

候选池 = 留出集中**满足约束**的真实实验条件；打分用其**实测**目标值。

## 2. 汇总

| 指标 | 值 |
| --- | --- |
| 模型推荐相对随机的**中位提升** | **{agg['median_improvement_vs_random_pct']:+.2f}%** |
| 模型推荐相对随机的均值提升（受右偏影响） | {agg['mean_improvement_vs_random_pct']:+.2f}% |
| 模型推荐中位百分位（相对随机分布） | {agg['median_model_percentile_vs_random']:.0f} |
| 高于随机均值的折次占比 | {agg['win_rate_vs_random_mean_pct']:.0f}% |
| 超过随机 95 分位的折次占比 | {agg['beat_random_p95_pct']:.0f}% |
| 模型捕获 oracle 上界的中位比例 | {agg['model_vs_oracle_median_ratio']:.2f} |
| 「就近最佳实验」启发式的平均提升 | {agg['nn_heuristic_mean_improvement_pct']:+.2f}% |

**判定：`{s['verdict']}`**

{s['interpretation']}

## 3. 逐折结果

| 折 | 模型 | 训练/池 | 模型选中均值 | 随机均值 | 随机95分位 | 就近最佳 | 相对随机 | 百分位 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
{rows}

## 4. 局限

- 留出池由历史记录构成，模型未见过的参数组合不在检验范围内；这只检验「择优」能力，不检验「探索」能力。
- 随机基线用 200 次有种子抽样估计分布，本身有抽样误差。
- {s['caveat']}
"""


def _finish(args, outdir, status, exit_code, reason, extra, started) -> int:
    payload = {"skill": {"name": SKILL_NAME, "version": SKILL_VERSION},
               "status": status, "exit_code": exit_code, "blocking_reason": reason,
               "generated_at_utc": now_utc(),
               "duration_seconds": round(time.perf_counter() - started, 3), **extra}
    write_json(payload, Path(outdir) / "run_manifest.json")
    print(json.dumps({"status": status, "exit_code": exit_code,
                      "blocking_reason": reason}, ensure_ascii=False))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
