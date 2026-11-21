#!/usr/bin/env python3
"""MixLab unified CLI — P0-FR01 … P0-FR05.

Usage:
    python3 scripts/run_mixlab.py --input data.csv --config config/study.json --outdir outputs/run_001

Exit codes:
    0  OK (candidates produced)
    1  FAILED_RUNTIME
    2  BLOCKED_INPUT        (missing/invalid configuration — never guessed)
    3  BLOCKED_DATA         (too few usable rows, constant target, ...)
    4  BLOCKED_MODEL_QUALITY(cannot beat the mean baseline)
    5  BLOCKED_CONSTRAINTS  (no feasible candidate inside bounds ∩ data range)

The legacy `integration_check.py` is intentionally left untouched.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import audit as audit_mod
import recommend as recommend_mod
import report as report_mod
import train as train_mod
import visualize as visualize_mod
from common import (LABEL_UNVERIFIED, VALID_PROVENANCE, fmt, now_utc, read_table,
                    sha256_file, write_json)

SKILL_NAME = "mixlab-integration-check"
SKILL_VERSION = "0.3.0"

EXIT_OK, EXIT_RUNTIME, EXIT_INPUT, EXIT_DATA, EXIT_MODEL, EXIT_CONSTRAINTS = 0, 1, 2, 3, 4, 5


class Blocked(Exception):
    def __init__(self, status: str, message: str, exit_code: int):
        super().__init__(message)
        self.status = status
        self.message = message
        self.exit_code = exit_code


# --------------------------------------------------------------------------- #
# P0-FR01 — configuration validation (no guessing, ever)
# --------------------------------------------------------------------------- #
def validate_config(cfg: dict, df, pd) -> list:
    errors = []

    if not isinstance(cfg, dict) or not cfg:
        raise Blocked("BLOCKED_INPUT", "配置文件为空或不是 JSON 对象", EXIT_INPUT)

    target = cfg.get("target_column")
    if not target or not isinstance(target, str):
        errors.append("`target_column` 必填，且必须显式给出（禁止自动猜测目标列）")
    elif target not in df.columns:
        errors.append(f"`target_column` = '{target}' 不在数据列中；现有列：{list(df.columns)[:15]}")
    elif not pd.api.types.is_numeric_dtype(pd.to_numeric(df[target], errors="coerce")):
        errors.append(f"`target_column` = '{target}' 不是数值列")
    elif pd.to_numeric(df[target], errors="coerce").notna().sum() == 0:
        errors.append(f"`target_column` = '{target}' 没有任何数值")

    prov = cfg.get("data_provenance") or {}
    if not isinstance(prov, dict) or not prov.get("type"):
        errors.append("`data_provenance.type` 必填（PUBLIC_EXPERIMENTAL / SIMULATED / …）")
    elif str(prov.get("type")) not in VALID_PROVENANCE:
        errors.append(f"`data_provenance.type` = '{prov.get('type')}' 不在允许集合 {sorted(VALID_PROVENANCE)}")
    if isinstance(prov, dict) and not prov.get("source"):
        errors.append("`data_provenance.source` 必填（URL / DOI / 说明）")

    obj = cfg.get("objective") or {}
    direction = str(obj.get("direction", "")).lower()
    if direction not in {"maximize", "minimize", "target"}:
        errors.append("`objective.direction` 必填，取值 maximize / minimize / target")
    elif direction == "target":
        try:
            float(obj.get("target_value"))
        except (TypeError, ValueError):
            errors.append("`objective.direction=target` 时必须给出数值 `objective.target_value`")

    variables = cfg.get("variables") or {}
    if not isinstance(variables, dict) or not variables:
        errors.append("`variables` 必填：至少一个变量并注明 role / unit")
        variables = {}

    controllable = 0
    for name, spec in variables.items():
        if not isinstance(spec, dict):
            errors.append(f"变量 `{name}` 的配置格式应为对象")
            continue
        if name not in df.columns:
            errors.append(f"变量 `{name}` 不在数据列中")
        role = str(spec.get("role", "")).lower()
        if role not in {"controllable", "context"}:
            errors.append(f"变量 `{name}` 的 role 必须为 controllable 或 context")
        if not str(spec.get("unit", "")).strip():
            errors.append(f"变量 `{name}` 缺少 unit")
        if role == "controllable":
            controllable += 1
            bounds = spec.get("bounds")
            if not (isinstance(bounds, (list, tuple)) and len(bounds) == 2):
                errors.append(f"变量 `{name}` 缺少 `bounds: [lower, upper]`")
            else:
                try:
                    lo, hi = float(bounds[0]), float(bounds[1])
                    if not (math.isfinite(lo) and math.isfinite(hi)):
                        errors.append(f"变量 `{name}` 的 bounds 含非有限值")
                    elif lo >= hi:
                        errors.append(f"变量 `{name}` 的 bounds 冲突：lower={lo} 不小于 upper={hi}")
                except (TypeError, ValueError):
                    errors.append(f"变量 `{name}` 的 bounds 必须是两个数值")
        elif role == "context":
            fv = spec.get("fixed_value")
            try:
                float(fv)
            except (TypeError, ValueError):
                errors.append(f"固定变量 `{name}` 必须给出数值 `fixed_value`")
    if variables and controllable == 0:
        errors.append("`variables` 中至少需要一个 role=controllable 的变量")

    split = cfg.get("split") or {}
    method = str(split.get("method", "random")).lower()
    if method not in {"random", "group"}:
        errors.append(f"`split.method` = '{method}' 不支持（random / group）")
    if method == "group":
        gcol = split.get("group_column")
        if not gcol:
            errors.append("`split.method=group` 时必须给出 `split.group_column`")
        elif gcol not in df.columns:
            errors.append(f"`split.group_column` = '{gcol}' 不在数据列中")

    rec = cfg.get("recommendation") or {}
    try:
        top_k = int(rec.get("top_k", 3))
        if not 1 <= top_k <= 10:
            errors.append(f"`recommendation.top_k` = {top_k} 超范围（1–10）")
    except (TypeError, ValueError):
        errors.append("`recommendation.top_k` 必须是整数")
    try:
        budget = int(rec.get("candidate_budget", 200))
        if not 1 <= budget <= 5000:
            errors.append(f"`recommendation.candidate_budget` = {budget} 超范围（1–5000）")
    except (TypeError, ValueError):
        errors.append("`recommendation.candidate_budget` 必须是整数")

    transform = str((cfg.get("model") or {}).get("target_transform", "none")).lower()
    if transform not in {"none", "log", "log10"}:
        errors.append(f"`model.target_transform` = '{transform}' 不支持（none / log / log10）")
    elif transform in {"log", "log10"} and target in df.columns:
        y = pd.to_numeric(df[target], errors="coerce").dropna()
        if len(y) and (y <= 0).any():
            errors.append(f"`model.target_transform={transform}` 要求目标列全部 > 0")

    return errors


# --------------------------------------------------------------------------- #
# pipeline
# --------------------------------------------------------------------------- #
def run_pipeline(args, cfg: dict, df, io_meta: dict, ctx: dict, outdir: Path,
                 pd, np, plt) -> None:
    target = cfg["target_column"]
    provenance = cfg["data_provenance"]["type"]
    quality = cfg.get("quality") or {}
    min_rows = int(quality.get("min_rows", 30))
    gate_pct = float(quality.get("min_mae_improvement_pct", 5.0))

    # ---- FR02 audit (report only, never delete) ----
    ctx["audit"] = audit_mod.audit_frame(df, target, cfg["variables"], provenance, pd, np)
    (outdir / "figures").mkdir(exist_ok=True)
    audit_mod.plot_overview(df, ctx["audit"], target, provenance,
                            outdir / "figures" / "data_overview.png", pd, np, plt)
    write_json(ctx["audit"], outdir / "data_audit.json")

    # ---- working frame ----
    feature_names = list(cfg["variables"].keys())
    work = df[feature_names + [target]].copy()
    work[target] = pd.to_numeric(work[target], errors="coerce")
    for name in feature_names:
        work[name] = pd.to_numeric(work[name], errors="coerce")
    n_before = len(work)
    work = work.dropna()
    ctx["resolved"]["rows_dropped_missing"] = int(n_before - len(work))
    ctx["resolved"]["rows_used"] = int(len(work))
    if len(work) < min_rows:
        raise Blocked("BLOCKED_DATA",
                      f"去掉缺失值后仅 {len(work)} 行，低于最小样本量 {min_rows}", EXIT_DATA)
    y_all = work[target].to_numpy(dtype=float)
    if float(np.ptp(y_all)) == 0.0:
        raise Blocked("BLOCKED_DATA", f"目标列 `{target}` 无变化（方差为 0）", EXIT_DATA)

    # ---- FR03 split / models / gate ----
    group_col = (cfg.get("split") or {}).get("group_column")
    if group_col and group_col not in work.columns:
        work = work.assign(**{group_col: df.loc[work.index, group_col]})
    try:
        tr, te, split_info = train_mod.split_indices(work, cfg, np)
    except ValueError as exc:
        raise Blocked("BLOCKED_INPUT", f"数据划分失败: {exc}", EXIT_INPUT)
    split_info["n_train"], split_info["n_test"] = int(len(tr)), int(len(te))
    ctx["split_info"] = split_info
    if split_info.get("warning"):
        ctx["warnings"].append(split_info["warning"])
    if len(te) < int(quality.get("min_test_rows_for_r2", 20)):
        ctx["warnings"].append(f"测试集仅 {len(te)} 行，R² 不稳定，请谨慎解读")

    Xtr = work.iloc[tr][feature_names]
    Xte = work.iloc[te][feature_names]
    ytr, yte = work.iloc[tr][target].to_numpy(dtype=float), work.iloc[te][target].to_numpy(dtype=float)

    seed = int((cfg.get("split") or {}).get("seed", 42))
    estimators = train_mod.build_estimators(cfg, len(tr), seed, np)
    results, preds = train_mod.evaluate(estimators, Xtr, ytr, Xte, yte, seed, np)
    best = train_mod.choose_best(results)
    passed, gate_reason = train_mod.quality_gate(best, results, gate_pct)
    ctx["model_results"], ctx["best_model"] = results, best
    ctx["gate"] = {"passed": passed, "reason": gate_reason, "min_improvement_pct": gate_pct}

    write_json({"best_model": best, "gate": ctx["gate"], "models": results,
                "split": split_info, "target_column": target,
                "target_transform": (cfg.get("model") or {}).get("target_transform", "none"),
                "n_train": len(tr), "n_test": len(te)},
               outdir / "model_metrics.json")

    # predictions.csv (best model, test split only)
    import pandas as pd_any  # same pandas instance
    pred_df = pd_any.DataFrame({
        "original_row_index": work.index.to_numpy()[te],
        "y_true": np.round(yte, 6),
        "y_pred": np.round(preds[best]["test"], 6),
        "residual": np.round(yte - preds[best]["test"], 6),
    })
    pred_df["abs_error"] = np.round(pred_df["residual"].abs(), 6)
    pred_df.to_csv(outdir / "predictions.csv", index=False)

    visualize_mod.plot_pred_vs_actual(yte, preds[best]["test"], results[best]["metrics_test"],
                                      best, provenance, outdir / "figures" / "pred_vs_actual.png",
                                      np, plt)
    visualize_mod.plot_target_distribution(y_all, target, provenance, len(work),
                                            outdir / "figures" / "target_distribution.png",
                                            np, plt)

    if not passed:
        raise Blocked("BLOCKED_MODEL_QUALITY", gate_reason, EXIT_MODEL)

    # ---- FR04 candidates ----
    bounds, problems = recommend_mod.support_bounds(work, cfg)
    if problems:
        raise Blocked("BLOCKED_CONSTRAINTS", "; ".join(problems), EXIT_CONSTRAINTS)
    ctx["resolved"]["effective_bounds"] = bounds

    cands, mode = recommend_mod.generate_candidates(cfg, bounds, seed, np)
    if cands.empty:
        raise Blocked("BLOCKED_CONSTRAINTS", "没有生成任何候选参数组合", EXIT_CONSTRAINTS)
    ctx["resolved"]["candidate_mode"] = mode

    ranked = recommend_mod.score_and_rank(cands, estimators[best], work, cfg,
                                          feature_names, bounds, np)
    if ranked.empty:
        raise Blocked("BLOCKED_CONSTRAINTS", "候选全部被约束或多样性规则过滤", EXIT_CONSTRAINTS)
    ranked.insert(0, "candidate_id", [f"C{i:02d}" for i in range(1, len(ranked) + 1)])
    ctx["candidates"] = ranked
    ranked.to_csv(outdir / "candidate_experiments.csv", index=False)

    visualize_mod.plot_candidate_comparison(ranked, work, bounds,
                                            outdir / "figures" / "candidate_comparison.png",
                                            np, plt)
    visualize_mod.plot_candidate_ranking(ranked, work[target], cfg["objective"]["direction"],
                                         outdir / "figures" / "candidate_ranking.png", plt)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description="MixLab unified pipeline (P0-FR01..FR05)")
    parser.add_argument("--input", required=True, help="experiment table (.csv/.xlsx)")
    parser.add_argument("--config", required=True, help="study configuration JSON")
    parser.add_argument("--outdir", default="outputs/run", help="output directory")
    parser.add_argument("--tag", default=None, help="free-text label for the manifest")
    args = parser.parse_args()

    started = time.perf_counter()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    figures_dir = outdir / "figures"
    figures_dir.mkdir(exist_ok=True)

    ctx = {
        "skill_name": SKILL_NAME, "skill_version": SKILL_VERSION,
        "command": " ".join([Path(sys.executable).name] + sys.argv),
        "cwd": str(Path.cwd()),
        "started_at_utc": now_utc(),
        "input_meta": None, "config_path": str(Path(args.config).resolve()),
        "config": {}, "resolved": {}, "audit": None, "split_info": {},
        "model_results": {}, "best_model": None, "gate": {}, "candidates": None,
        "warnings": [], "provenance": None,
        "status": "FAILED_RUNTIME", "exit_code": EXIT_RUNTIME, "blocking_reason": None,
        "error": None,
    }

    status, exit_code, reason = "FAILED_RUNTIME", EXIT_RUNTIME, None

    try:
        import json

        input_path = Path(args.input)
        if not input_path.is_absolute():
            input_path = (Path.cwd() / input_path).resolve()
        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = (Path.cwd() / config_path).resolve()

        if not input_path.exists():
            raise Blocked("BLOCKED_INPUT", f"输入文件不存在: {input_path}", EXIT_INPUT)
        if not config_path.exists():
            raise Blocked("BLOCKED_INPUT", f"配置文件不存在: {config_path}", EXIT_INPUT)

        ctx["input_meta"] = {"path": str(input_path), "name": input_path.name,
                             "size_bytes": input_path.stat().st_size,
                             "sha256": sha256_file(input_path)}

        # dependency check before touching data
        missing = []
        for mod in ("pandas", "numpy", "sklearn", "matplotlib"):
            try:
                __import__(mod)
            except Exception:  # noqa: BLE001
                missing.append(mod)
        if missing:
            raise Blocked("FAILED_RUNTIME", f"缺少依赖: {', '.join(missing)}", EXIT_RUNTIME)

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd

        try:
            cfg = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            raise Blocked("BLOCKED_INPUT", f"配置文件不是合法 JSON: {type(exc).__name__}: {exc}",
                          EXIT_INPUT)

        ctx["config"] = cfg
        ctx["config_path"] = str(config_path)
        ctx["config_sha256"] = sha256_file(config_path)
        ctx["provenance"] = ((cfg.get("data_provenance") or {}).get("type"))

        try:
            df, io_meta = read_table(input_path, pd)
        except Exception as exc:  # noqa: BLE001
            raise Blocked("BLOCKED_INPUT", f"数据文件读取失败: {exc}", EXIT_INPUT)
        ctx["input_meta"].update({"encoding": io_meta.get("encoding"),
                                  "separator": io_meta.get("separator"),
                                  "sheet": io_meta.get("sheet")})

        if df.shape[1] < 2:
            raise Blocked("BLOCKED_INPUT", "数据表列数不足 2，无法建模", EXIT_INPUT)
        if not df.columns.is_unique:
            dup = df.columns[df.columns.duplicated()].tolist()
            raise Blocked("BLOCKED_INPUT", f"列名重复: {dup}", EXIT_INPUT)

        errors = validate_config(cfg, df, pd)
        if errors:
            raise Blocked("BLOCKED_INPUT", "配置校验失败：" + "; ".join(errors), EXIT_INPUT)

        run_pipeline(args, cfg, df, io_meta, ctx, outdir, pd, np, plt)
        status, exit_code, reason = "OK", EXIT_OK, None

    except Blocked as blocked:
        status, exit_code, reason = blocked.status, blocked.exit_code, blocked.message
        ctx["error"] = {"type": "Blocked", "message": blocked.message, "traceback": None}
    except Exception as exc:  # noqa: BLE001
        status, exit_code = "FAILED_RUNTIME", EXIT_RUNTIME
        reason = f"{type(exc).__name__}: {exc}"
        ctx["error"] = {"type": type(exc).__name__, "message": str(exc),
                        "traceback": traceback.format_exc().strip().splitlines()[-10:]}

    # ---------------- finalize (always writes evidence) ----------------
    artifacts = []
    try:
        write_json({
            "skill": {"name": SKILL_NAME, "version": SKILL_VERSION},
            "input": ctx["input_meta"], "config_path": ctx["config_path"],
            "config_sha256": ctx.get("config_sha256"),
            "config": ctx["config"], "resolved": ctx["resolved"],
            "status": status, "blocking_reason": reason,
        }, outdir / "run_config.json")
        artifacts.append(outdir / "run_config.json")

        for extra in ("data_audit.json", "model_metrics.json", "predictions.csv",
                      "candidate_experiments.csv"):
            p = outdir / extra
            if p.exists():
                artifacts.append(p)
        for p in sorted(figures_dir.glob("*.png")):
            artifacts.append(p)

        ctx["status"], ctx["exit_code"], ctx["blocking_reason"] = status, exit_code, reason
        ctx["data_label_candidate"] = LABEL_UNVERIFIED
        ctx["outputs"] = [
            {"name": str(p.relative_to(outdir)), "size_bytes": p.stat().st_size,
             "sha256": sha256_file(p)} for p in artifacts
        ]

        report_path = outdir / "research_report.md"
        report_path.write_text(report_mod.build_report(ctx), encoding="utf-8")
        artifacts.append(report_path)

        manifest = {
            "skill": {"name": SKILL_NAME, "version": SKILL_VERSION},
            "status": status, "exit_code": exit_code, "blocking_reason": reason,
            "command": ctx["command"], "argv": sys.argv, "cwd": ctx["cwd"],
            "started_at_utc": ctx["started_at_utc"], "finished_at_utc": now_utc(),
            "duration_seconds": round(time.perf_counter() - started, 3),
            "python": {"executable": sys.executable, "version": sys.version.split()[0]},
            "input": ctx["input_meta"], "config_path": ctx["config_path"],
            "config_sha256": ctx.get("config_sha256"),
            "seed": (ctx["config"].get("split") or {}).get("seed"),
            "data_provenance": ctx["config"].get("data_provenance"),
            "split": ctx.get("split_info"), "best_model": ctx.get("best_model"),
            "gate": ctx.get("gate"), "warnings": ctx["warnings"],
            "error": ctx.get("error"),
            "outputs": [
                {"name": str(p.relative_to(outdir)), "size_bytes": p.stat().st_size,
                 "sha256": sha256_file(p)} for p in artifacts
            ] + [{"name": "run_manifest.json", "size_bytes": None, "sha256": None,
                  "note": "self"}],
        }
        write_json(manifest, outdir / "run_manifest.json")
    except Exception as exc:  # noqa: BLE001 - never hide the original failure
        print(json.dumps({"status": status, "exit_code": exit_code,
                          "error": f"{type(exc).__name__}: {exc}",
                          "note": "finalize failed"}, ensure_ascii=False))

    print(json.dumps({
        "skill": SKILL_NAME, "version": SKILL_VERSION,
        "status": status, "exit_code": exit_code, "blocking_reason": reason,
        "best_model": ctx.get("best_model"),
        "metrics_test": (ctx.get("model_results") or {}).get(ctx.get("best_model") or "", {})
                        .get("metrics_test"),
        "n_candidates": 0 if ctx.get("candidates") is None else int(len(ctx["candidates"])),
        "outdir": str(outdir),
    }, ensure_ascii=False))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
