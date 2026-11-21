"""P0-FR02 — data audit and overview figure.

Audit only *reports*; it never deletes rows. Cleaning policy is recorded by the
caller in run_config.json, as required by PRD V2.0 §P0-FR02.
"""

from __future__ import annotations

from pathlib import Path

from common import C_GREY, C_PRIMARY, LABEL_PREDICTED, save_fig, style_axes


def audit_frame(df, target: str, variables: dict, provenance: str, pd, np) -> dict:
    checks = {
        "data_label": provenance,
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "columns": [
            {
                "name": str(c),
                "dtype": str(df[c].dtype),
                "n_missing": int(df[c].isna().sum()),
                "missing_pct": round(float(df[c].isna().mean() * 100.0), 2),
                "n_unique": int(df[c].nunique(dropna=True)),
            }
            for c in df.columns
        ],
        "total_missing_cells": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "constant_columns": [str(c) for c in df.columns if df[c].nunique(dropna=True) <= 1],
        "numeric_columns": [str(c) for c in df.columns if pd.api.types.is_numeric_dtype(df[c])],
        "role_columns": {
            role: [v for v, spec in variables.items() if spec.get("role") == role]
            for role in ("controllable", "context")
        },
        "iqr_suspicious": {},
        "missing_in_used_columns": {},
        "warnings": [],
    }

    used = [target] + [v for v in variables]
    for col in used:
        if col not in df.columns:
            continue
        checks["missing_in_used_columns"][col] = int(df[col].isna().sum())
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(series) >= 8:
            q1, q3 = float(series.quantile(0.25)), float(series.quantile(0.75))
            iqr = q3 - q1
            if iqr > 0:
                mask = (series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)
                checks["iqr_suspicious"][col] = int(mask.sum())

    y = pd.to_numeric(df[target], errors="coerce")
    y_clean = y.dropna()
    checks["target_stats"] = {
        "column": target,
        "n": int(y_clean.shape[0]),
        "n_missing": int(y.isna().sum()),
        "mean": round(float(y_clean.mean()), 6) if len(y_clean) else None,
        "std": round(float(y_clean.std(ddof=0)), 6) if len(y_clean) else None,
        "min": round(float(y_clean.min()), 6) if len(y_clean) else None,
        "p05": round(float(y_clean.quantile(0.05)), 6) if len(y_clean) else None,
        "median": round(float(y_clean.median()), 6) if len(y_clean) else None,
        "p95": round(float(y_clean.quantile(0.95)), 6) if len(y_clean) else None,
        "max": round(float(y_clean.max()), 6) if len(y_clean) else None,
    }

    if checks["duplicate_rows"]:
        checks["warnings"].append(f"{checks['duplicate_rows']} 行完全重复（仅标记，未删除）")
    for col, n in checks["missing_in_used_columns"].items():
        if n:
            checks["warnings"].append(f"建模列 `{col}` 缺 {n} 行")
    for col, n in checks["iqr_suspicious"].items():
        if n:
            checks["warnings"].append(f"`{col}` 有 {n} 个 IQR 可疑值（仅标记，未删除）")
    return checks


def plot_overview(df, audit: dict, target: str, provenance: str, out_png: Path, pd, np, plt) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.4), dpi=160)

    miss = (
        df.isna().mean().mul(100.0).sort_values(ascending=False)
    )
    top = miss.head(12)[::-1]
    colors = [C_PRIMARY if v > 0 else C_GREY for v in top.values]
    ax1.barh([str(c)[:26] for c in top.index], top.values, color=colors, height=0.65)
    ax1.set_xlabel("Missing values (%)")
    ax1.set_title(f"Missingness (n = {len(df)} rows)", fontsize=10)
    ax1.set_xlim(0, max(5.0, float(top.max()) * 1.15 if len(top) else 5.0))
    style_axes(ax1, grid=False)

    y = pd.to_numeric(df[target], errors="coerce").dropna()
    ax2.hist(y, bins=30, color=C_PRIMARY, alpha=0.85, edgecolor="white", linewidth=0.6)
    ax2.axvline(float(y.median()), color=C_GREY, linestyle="--", linewidth=1.2,
                label=f"median = {y.median():.4g}")
    ax2.set_xlabel(f"{target}  (MEASURED)")
    ax2.set_ylabel("Count")
    ax2.set_title(f"Target distribution (n = {len(y)})", fontsize=10)
    ax2.legend(fontsize=8, frameon=False)
    style_axes(ax2)

    fig.suptitle(f"Data audit — label: {provenance}", fontsize=11)
    save_fig(fig, out_png, plt)
