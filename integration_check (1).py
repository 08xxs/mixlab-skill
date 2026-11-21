#!/usr/bin/env python3
"""mixlab-integration-check — runnable CSV integration test for MixLab.

Reads a user-supplied CSV, runs data checks, fits a linear regression on
numeric columns, evaluates MAE / RMSE / R2, and writes a PNG figure plus a
predictions CSV, together with machine-readable evidence (run manifest,
data checks, metrics, Markdown report).

Exit codes
----------
0  PASS (artifacts generated; warnings may exist)
2  INPUT error (missing/unreadable/incompatible CSV)
3  RUNTIME error (unexpected failure during processing)

The script never invents values: every number in the outputs comes from the
actual computation on the given file.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import re
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

SKILL_NAME = "mixlab-integration-check"
SKILL_VERSION = "1.0.0"

REQUIRED_LIBS = ["pandas", "numpy", "sklearn", "matplotlib"]
PROBE_LIBS = REQUIRED_LIBS + ["scipy"]

TARGET_HINTS = (
    "yield", "target", "response", "label", "output", "quality",
    "score", "value", "result", "measurement",
)
ID_PATTERN = re.compile(r"(^id$|_id$|^index$|^no$|^序号$|^编号$|^sample$)", re.IGNORECASE)


class InputError(Exception):
    """The supplied CSV cannot be used for this analysis."""


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def probe_libraries() -> dict:
    out = {}
    for name in PROBE_LIBS:
        try:
            mod = importlib.import_module(name)
            out[name] = {
                "version": getattr(mod, "__version__", None),
                "import_ok": True,
                "error": None,
            }
        except Exception as exc:  # noqa: BLE001
            out[name] = {
                "version": None,
                "import_ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
    return out


def read_table(path: Path, pd) -> tuple[object, dict]:
    """Read a CSV with encoding and separator fallback."""
    meta = {"encoding": None, "separator": ","}
    last_decode_error = None
    df = None
    for enc in ("utf-8-sig", "gb18030", "latin-1"):
        try:
            df = pd.read_csv(path, encoding=enc)
            meta["encoding"] = enc
            break
        except UnicodeDecodeError as exc:
            last_decode_error = exc
        except pd.errors.EmptyDataError as exc:
            raise InputError(f"CSV 为空或无表头: {exc}")
        except Exception as exc:  # noqa: BLE001
            raise InputError(f"CSV 解析失败: {type(exc).__name__}: {exc}")
    if df is None:
        raise InputError(f"无法解码 CSV 文件: {last_decode_error}")

    if df.shape[1] == 1:
        head = path.read_text(encoding=meta["encoding"], errors="replace").splitlines()[0]
        for sep, label in ((";", "semicolon"), ("\t", "tab"), ("|", "pipe")):
            if head.count(sep) >= 2:
                df = pd.read_csv(path, encoding=meta["encoding"], sep=sep)
                meta["separator"] = label
                break
    return df, meta


def numeric_columns(df, pd) -> list[str]:
    return [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])
    ]


def pick_target(df, numeric: list[str], pd, explicit: str | None):
    if explicit:
        if explicit not in df.columns:
            raise InputError(f"--target '{explicit}' 不在 CSV 列中；现有列: {list(df.columns)}")
        if explicit not in numeric:
            raise InputError(f"--target '{explicit}' 不是数值列（dtype={df[explicit].dtype}）")
        return explicit, "explicit: --target"

    for hint in TARGET_HINTS:
        for col in numeric:
            low = str(col).lower()
            if low == hint or low.endswith(hint) or hint in low:
                return col, f"auto: 列名命中关键字 '{hint}'"

    if not numeric:
        raise InputError("CSV 中没有任何数值列，无法进行回归分析")
    return numeric[-1], "auto: 回退到最后一个数值列"


def pick_features(df, numeric: list[str], target: str, pd) -> tuple[list[str], list[dict]]:
    feats, dropped = [], []
    n = len(df)
    for col in numeric:
        if col == target:
            continue
        if ID_PATTERN.search(str(col)):
            dropped.append({"column": col, "reason": "identifier-like name"})
            continue
        if pd.api.types.is_integer_dtype(df[col]) and n > 3 and df[col].nunique(dropna=True) == n:
            dropped.append({"column": col, "reason": "integer column with all-unique values"})
            continue
        if df[col].nunique(dropna=True) <= 1:
            dropped.append({"column": col, "reason": "constant column"})
            continue
        feats.append(col)
    return feats, dropped


def run_checks(df, numeric: list[str], pd, np) -> dict:
    checks = {
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "columns": [
            {
                "name": str(c),
                "dtype": str(df[c].dtype),
                "n_missing": int(df[c].isna().sum()),
                "n_unique": int(df[c].nunique(dropna=True)),
            }
            for c in df.columns
        ],
        "numeric_columns": list(numeric),
        "non_numeric_columns": [str(c) for c in df.columns if c not in numeric],
        "total_missing_cells": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "constant_columns": [str(c) for c in df.columns if df[c].nunique(dropna=True) <= 1],
        "iqr_outliers_1p5": {},
    }
    for col in numeric:
        series = df[col].dropna()
        if len(series) >= 8:
            q1, q3 = float(series.quantile(0.25)), float(series.quantile(0.75))
            iqr = q3 - q1
            if iqr > 0:
                mask = (series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)
                checks["iqr_outliers_1p5"][str(col)] = int(mask.sum())
    return checks


def plot_results(y_true, y_pred, metrics: dict, title_suffix: str, png_path: Path, np, plt) -> None:
    residuals = y_true - y_pred
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.4), dpi=160)

    ax1.scatter(y_true, y_pred, s=46, color="#2f6ea8", edgecolor="white",
                linewidth=0.8, zorder=3, label=f"test samples (n={len(y_true)})")
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    pad = 0.06 * (hi - lo if hi > lo else 1.0)
    ax1.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "--",
             color="#b0413e", linewidth=1.4, label="ideal y = x", zorder=2)
    ax1.set_xlabel("Measured target")
    ax1.set_ylabel("Predicted target")
    ax1.set_title("Measured vs predicted", fontsize=10)
    ax1.text(
        0.03, 0.97,
        f"MAE = {metrics['MAE']:.4g}\nRMSE = {metrics['RMSE']:.4g}\nR2 = {metrics['R2']:.4g}",
        transform=ax1.transAxes, va="top", ha="left", fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.38", facecolor="#f4f6f8", edgecolor="#c9d2da"),
    )
    ax1.legend(loc="lower right", fontsize=8, frameon=False)
    ax1.grid(alpha=0.25, linewidth=0.6)

    ax2.scatter(y_pred, residuals, s=46, color="#3f8f6b", edgecolor="white",
                linewidth=0.8, zorder=3)
    ax2.axhline(0.0, color="#b0413e", linestyle="--", linewidth=1.2)
    ax2.set_xlabel("Predicted target")
    ax2.set_ylabel("Residual (measured - predicted)")
    ax2.set_title("Residuals", fontsize=10)
    ax2.grid(alpha=0.25, linewidth=0.6)

    fig.suptitle(f"MixLab integration check — {title_suffix}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(png_path)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(
        description="MixLab integration check: CSV -> data checks -> linear regression -> metrics -> PNG/CSV"
    )
    parser.add_argument("--input", required=True, help="path to the CSV to analyse")
    parser.add_argument("--outdir", default="mixlab_integration_output", help="output directory")
    parser.add_argument("--target", default=None, help="target column (default: auto-detect)")
    parser.add_argument("--test-size", type=float, default=0.25, help="test fraction (default 0.25)")
    parser.add_argument("--seed", type=int, default=42, help="random seed (default 42)")
    parser.add_argument("--tag", default="user_csv", help="label used in the figure title")
    parser.add_argument(
        "--drop-outliers",
        action="store_true",
        help="drop rows flagged by the 1.5*IQR rule on the used columns (default: keep them)",
    )
    args = parser.parse_args()

    started = time.perf_counter()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    record: dict = {
        "skill": {"name": SKILL_NAME, "version": SKILL_VERSION},
        "script": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve())},
        "command": " ".join([Path(sys.executable).name] + sys.argv),
        "argv": sys.argv,
        "cwd": str(Path.cwd()),
        "started_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": {
            "executable": sys.executable,
            "version": sys.version.split()[0],
            "implementation": __import__("platform").python_implementation(),
        },
        "libraries": probe_libraries(),
        "params": {
            "input": args.input,
            "outdir": str(outdir),
            "target": args.target,
            "test_size": args.test_size,
            "seed": args.seed,
            "tag": args.tag,
        },
        "input": None,
        "data_checks": None,
        "model": None,
        "outputs": [],
        "status": "ERROR",
        "exit_code": None,
        "error": None,
        "warnings": [],
    }

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = (Path.cwd() / input_path).resolve()

    artifacts: list[Path] = []
    checks: dict = {}
    metrics: dict = {}
    status = "ERROR"
    exit_code = 3
    error_info = None

    missing_libs = [n for n, v in record["libraries"].items() if not v["import_ok"]]
    try:
        if not input_path.exists():
            raise InputError(f"输入文件不存在: {input_path}")
        if not input_path.is_file():
            raise InputError(f"输入路径不是文件: {input_path}")
        if input_path.suffix.lower() not in {".csv", ".txt", ".tsv"}:
            record["warnings"].append(f"输入扩展名为 {input_path.suffix}，仍按 CSV 尝试解析")

        if missing_libs:
            raise RuntimeError(f"缺少必要的 Python 库: {', '.join(missing_libs)}")

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        from sklearn.model_selection import train_test_split

        t0 = time.perf_counter()
        df, io_meta = read_table(input_path, pd)
        record["input"] = {
            "path": str(input_path),
            "name": input_path.name,
            "size_bytes": input_path.stat().st_size,
            "sha256": sha256_file(input_path),
            "encoding": io_meta["encoding"],
            "separator": io_meta["separator"],
            "read_seconds": round(time.perf_counter() - t0, 4),
        }

        if df.shape[1] == 1 and df.shape[0] <= 1:
            raise InputError("CSV 仅解析出单列单行，可能分隔符或表头不合法")

        numeric = numeric_columns(df, pd)
        if not numeric:
            raise InputError("CSV 中没有数值列，无法进行回归分析")

        checks = run_checks(df, numeric, pd, np)
        target, target_reason = pick_target(df, numeric, pd, args.target)
        features, dropped = pick_features(df, numeric, target, pd)

        work = df[features + [target]].dropna()
        rows_dropped = int(len(df) - len(work))
        if not features:
            raise InputError("除目标列外没有可用数值特征")
        if len(work) < 8:
            raise InputError(f"去掉缺失值后仅剩 {len(work)} 行（需要 >= 8）")

        outliers_dropped = 0
        if args.drop_outliers and len(work) >= 8:
            flag = pd.Series(False, index=work.index)
            for col in features + [target]:
                s = work[col]
                q1, q3 = float(s.quantile(0.25)), float(s.quantile(0.75))
                iqr = q3 - q1
                if iqr > 0:
                    flag |= (s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)
            outliers_dropped = int(flag.sum())
            work = work[~flag]
            if len(work) < 8:
                raise InputError(
                    f"按 1.5*IQR 剔除 {outliers_dropped} 行后仅剩 {len(work)} 行（需要 >= 8）"
                )

        if not (0.05 <= args.test_size <= 0.5):
            raise InputError(f"--test-size={args.test_size} 超出合理范围 [0.05, 0.5]")

        X_raw = work[features].to_numpy(dtype=float)
        y = work[target].to_numpy(dtype=float)
        if float(np.ptp(y)) == 0.0:
            raise InputError(f"目标列 '{target}' 无变化（方差为 0）")

        idx = np.arange(len(work))
        idx_train, idx_test = train_test_split(
            idx, test_size=args.test_size, random_state=args.seed, shuffle=True
        )

        mu = X_raw[idx_train].mean(axis=0)
        sd = X_raw[idx_train].std(axis=0, ddof=0)
        sd[sd == 0] = 1.0
        X = (X_raw - mu) / sd

        model = LinearRegression().fit(X[idx_train], y[idx_train])
        pred_train = model.predict(X[idx_train])
        pred_test = model.predict(X[idx_test])

        mae = float(mean_absolute_error(y[idx_test], pred_test))
        rmse = float(np.sqrt(mean_squared_error(y[idx_test], pred_test)))
        r2 = float(r2_score(y[idx_test], pred_test))
        mae_tr = float(mean_absolute_error(y[idx_train], pred_train))
        rmse_tr = float(np.sqrt(mean_squared_error(y[idx_train], pred_train)))
        r2_tr = float(r2_score(y[idx_train], pred_train))
        baseline_mae = float(mean_absolute_error(y[idx_test], np.full(len(idx_test), y[idx_train].mean())))

        metrics = {
            "status": "PASS",
            "target": target,
            "target_selection": target_reason,
            "features": features,
            "dropped_columns": dropped,
            "rows_total": int(len(df)),
            "rows_used": int(len(work)),
            "rows_dropped_missing": rows_dropped,
            "outliers_dropped": outliers_dropped,
            "drop_outliers_enabled": bool(args.drop_outliers),
            "n_train": int(len(idx_train)),
            "n_test": int(len(idx_test)),
            "test_size": args.test_size,
            "random_seed": args.seed,
            "drop_outliers": bool(args.drop_outliers),
            "model": "sklearn.linear_model.LinearRegression (z-score features, fit on train only)",
            "metrics_test": {"MAE": round(mae, 6), "RMSE": round(rmse, 6), "R2": round(r2, 6)},
            "metrics_train_reference": {
                "MAE": round(mae_tr, 6), "RMSE": round(rmse_tr, 6), "R2": round(r2_tr, 6),
            },
            "baseline_mean_predictor": {"test_MAE": round(baseline_mae, 6)},
            "mae_improvement_vs_baseline_pct": round(
                (baseline_mae - mae) / baseline_mae * 100.0, 3
            ) if baseline_mae > 1e-12 else None,
            "coefficients_standardized": {
                f: round(float(c), 6) for f, c in zip(features, model.coef_)
            },
            "intercept": round(float(model.intercept_), 6),
            "target_stats": {
                "mean": round(float(y.mean()), 6),
                "std": round(float(y.std(ddof=0)), 6),
                "min": round(float(y.min()), 6),
                "max": round(float(y.max()), 6),
            },
        }
        checks["target"] = target
        checks["rows_dropped_for_missing"] = rows_dropped

        # ---- predictions.csv ----
        predicted = pd.DataFrame({
            "original_row_index": work.index.to_numpy()[idx_test],
            "y_true": np.round(y[idx_test], 6),
            "y_pred": np.round(pred_test, 6),
            "residual": np.round(y[idx_test] - pred_test, 6),
            "abs_error": np.round(np.abs(y[idx_test] - pred_test), 6),
        })
        predictions_csv = outdir / "predictions.csv"
        predicted.to_csv(predictions_csv, index=False)
        artifacts.append(predictions_csv)

        # ---- metrics.json / data_checks.json ----
        metrics_json = outdir / "metrics.json"
        metrics_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        artifacts.append(metrics_json)

        checks_json = outdir / "data_checks.json"
        checks_json.write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8")
        artifacts.append(checks_json)

        # ---- PNG ----
        png_path = outdir / "integration_plot.png"
        plot_results(y[idx_test], pred_test, metrics["metrics_test"], args.tag, png_path, np, plt)
        artifacts.append(png_path)

        # ---- warnings from data quality ----
        if checks["total_missing_cells"] > 0:
            record["warnings"].append(f"输入含 {checks['total_missing_cells']} 个缺失单元格")
        if checks["duplicate_rows"] > 0:
            record["warnings"].append(f"输入含 {checks['duplicate_rows']} 行重复记录")
        if rows_dropped > 0:
            record["warnings"].append(f"因缺失值剔除 {rows_dropped} 行，实际使用 {len(work)} 行")
        outlier_total = sum(checks["iqr_outliers_1p5"].values())
        if outlier_total:
            record["warnings"].append(f"IQR 1.5 倍规则检出 {outlier_total} 个离群点（未剔除）")
        if len(idx_test) < 10:
            record["warnings"].append(f"测试集仅 {len(idx_test)} 行，指标置信度有限")

        status = "PASS"
        exit_code = 0

    except InputError as exc:
        error_info = {"type": "InputError", "message": str(exc), "traceback": None}
        status, exit_code = "FAIL_INPUT", 2
    except Exception as exc:  # noqa: BLE001
        error_info = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc().strip().splitlines()[-12:],
        }
        status, exit_code = "FAIL_RUNTIME", 3

    record["status"] = status
    record["exit_code"] = exit_code
    record["error"] = error_info
    record["finished_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    record["duration_seconds"] = round(time.perf_counter() - started, 4)
    record["data_checks"] = checks or None
    record["model"] = metrics or None

    # ---- report.md (hashes of artifacts written before it) ----
    def artifact_rows() -> str:
        if not artifacts:
            return "| — | — | — | 未生成 |"
        return "\n".join(
            f"| `{p.name}` | {p.stat().st_size} | `{sha256_file(p)}` |" for p in artifacts
        )

    if metrics:
        metric_lines = "\n".join([
            f"- 目标列：`{metrics['target']}`（{metrics['target_selection']}）",
            f"- 特征列：{', '.join('`' + f + '`' for f in metrics['features'])}",
            f"- 划分：训练 {metrics['n_train']} / 测试 {metrics['n_test']}（test_size={metrics['test_size']}, seed={metrics['random_seed']}）",
            f"- 缺失剔除 {metrics['rows_dropped_missing']} 行；IQR 离群剔除 {metrics['outliers_dropped']} 行"
            f"（--drop-outliers {'开启' if metrics['drop_outliers'] else '关闭'}）",
            f"- **测试集 MAE = {metrics['metrics_test']['MAE']}**",
            f"- **测试集 RMSE = {metrics['metrics_test']['RMSE']}**",
            f"- **测试集 R² = {metrics['metrics_test']['R2']}**",
            f"- 训练集（参考）MAE / RMSE / R² = {metrics['metrics_train_reference']['MAE']} / "
            f"{metrics['metrics_train_reference']['RMSE']} / {metrics['metrics_train_reference']['R2']}",
            f"- 均值基线测试集 MAE = {metrics['baseline_mean_predictor']['test_MAE']}，"
            f"模型相对改进 {metrics['mae_improvement_vs_baseline_pct']}%",
        ])
    else:
        metric_lines = "- 未计算（见错误信息）"

    check_lines = ""
    if checks:
        check_lines = "\n".join([
            f"- 行数 / 列数：{checks['n_rows']} / {checks['n_columns']}",
            f"- 数值列：{', '.join(checks['numeric_columns']) or '无'}",
            f"- 非数值列：{', '.join(checks['non_numeric_columns']) or '无'}",
            f"- 缺失单元格总数：{checks['total_missing_cells']}",
            f"- 重复行：{checks['duplicate_rows']}",
            f"- 常量列：{', '.join(checks['constant_columns']) or '无'}",
            f"- IQR 离群点：{json.dumps(checks['iqr_outliers_1p5'], ensure_ascii=False)}",
        ])

    lib_table = "\n".join(
        f"| {n} | {v['version'] or 'MISSING'} | {'ok' if v['import_ok'] else v['error']} |"
        for n, v in record["libraries"].items()
    )
    warn_block = "\n".join(f"- {w}" for w in record["warnings"]) or "- 无"

    report = f"""# MixLab Integration Check — 执行报告

> 由 Skill `{SKILL_NAME}` v{SKILL_VERSION} 的脚本实际运行生成。
> 所有数值均来自对输入文件的真实计算；本报告不包含任何占位或虚构指标。

## 1. 结论

| 项 | 值 |
| --- | --- |
| 运行状态 | **{status}** |
| 退出码 | {exit_code} |
| 开始（UTC） | {record['started_at_utc']} |
| 耗时 | {record['duration_seconds']} s |
| 错误 | {json.dumps(error_info, ensure_ascii=False) if error_info else '无'} |

## 2. 实际执行的命令与环境

```bash
{record['command']}
```

- 工作目录：`{record['cwd']}`
- 脚本：`{record['script']['path']}`
- 脚本 SHA-256：`{record['script']['sha256']}`
- Python：{record['python']['version']}（{record['python']['implementation']}，`{record['python']['executable']}`）

| 库 | 版本 | 导入 |
| --- | --- | --- |
{lib_table}

## 3. 输入文件

| 项 | 值 |
| --- | --- |
| 路径 | `{record['input']['path'] if record['input'] else args.input}` |
| 大小 | {record['input']['size_bytes'] if record['input'] else '—'} B |
| SHA-256 | `{record['input']['sha256'] if record['input'] else '—'}` |
| 编码 / 分隔符 | {record['input']['encoding'] if record['input'] else '—'} / {record['input']['separator'] if record['input'] else '—'} |

## 4. 数据检查

{check_lines or '- 未执行（输入在检查前已失败）'}

## 5. 回归与指标

{metric_lines}

## 6. 输出文件（字节数 / SHA-256）

| 文件 | 字节 | SHA-256 |
| --- | --- | --- |
{artifact_rows()}

> `run_manifest.json` 与 `integration_report.md` 自身不计入上表（写入顺序导致，避免自引用哈希）；
> `run_manifest.json` 记录了包含本报告在内的全部产物哈希。

## 7. 警告

{warn_block}

## 8. 复现方式

```bash
python3 {record['script']['path']} --input {record['input']['path'] if record['input'] else args.input} --outdir <输出目录> --seed {args.seed}
```
"""

    report_path = outdir / "integration_report.md"
    report_path.write_text(report, encoding="utf-8")
    artifacts.append(report_path)

    record["outputs"] = [
        {"name": str(p.relative_to(outdir)), "size_bytes": p.stat().st_size, "sha256": sha256_file(p)}
        for p in artifacts
    ]
    record["outputs"].append({
        "name": "run_manifest.json",
        "size_bytes": None,
        "sha256": None,
        "note": "self; see file on disk",
    })
    manifest_path = outdir / "run_manifest.json"
    manifest_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "skill": SKILL_NAME,
        "version": SKILL_VERSION,
        "status": status,
        "exit_code": exit_code,
        "outdir": str(outdir),
        "target": metrics.get("target") if metrics else None,
        "metrics_test": metrics.get("metrics_test") if metrics else None,
        "outputs": [p.name for p in artifacts] + ["run_manifest.json"],
        "error": error_info,
    }
    print(json.dumps(summary, ensure_ascii=False))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
