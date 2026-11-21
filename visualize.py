"""P0-FR05 — scientific figures for predictions and candidate proposals."""

from __future__ import annotations

import math

import numpy as np

from common import C_ACCENT, C_GREY, C_GOOD, C_PRIMARY, C_WARN, save_fig, style_axes


def plot_target_distribution(y, target: str, provenance: str, n_rows: int, out_png, np, plt) -> None:
    y = np.asarray(y, dtype=float)
    fig, ax = plt.subplots(figsize=(6.2, 4.2), dpi=160)
    ax.hist(y, bins=32, color=C_PRIMARY, alpha=0.85, edgecolor="white", linewidth=0.6)
    ax.axvline(float(np.median(y)), color=C_GREY, linestyle="--", linewidth=1.3,
               label=f"median = {float(np.median(y)):.4g}")
    ax.set_xlabel(f"{target}  (MEASURED)")
    ax.set_ylabel("Count")
    ax.set_title(f"Target distribution — n = {n_rows}, label: {provenance}", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    style_axes(ax)
    save_fig(fig, out_png, plt)


def plot_pred_vs_actual(y_true, y_pred, metrics: dict, best_name: str,
                        provenance: str, out_png, np, plt) -> None:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    fig, ax = plt.subplots(figsize=(5.6, 5.0), dpi=160)
    ax.scatter(y_true, y_pred, s=42, color=C_PRIMARY, edgecolor="white",
               linewidth=0.7, zorder=3, label=f"test samples (n={len(y_true)})")
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    pad = 0.06 * (hi - lo if hi > lo else 1.0)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "--", color=C_ACCENT,
            linewidth=1.4, label="ideal y = x", zorder=2)
    ax.set_xlabel("Observed target")
    ax.set_ylabel("Predicted target (PREDICTED)")
    ax.set_title(f"Test-set agreement — {best_name}", fontsize=10)
    ax.text(0.03, 0.97,
            f"MAE = {metrics['MAE']:.4g}\nRMSE = {metrics['RMSE']:.4g}\nR2 = {metrics['R2']:.4g}",
            transform=ax.transAxes, va="top", ha="left", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.38", facecolor="#f4f6f8", edgecolor="#c9d2da"))
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    style_axes(ax)
    fig.suptitle(f"Data label: {provenance}", fontsize=8.5, color=C_GREY)
    save_fig(fig, out_png, plt)


def plot_candidate_comparison(cand, df, bounds: dict, out_png, np, plt) -> None:
    variables = list(bounds)
    if not variables:
        return
    ncols = 2 if len(variables) > 1 else 1
    nrows = int(math.ceil(len(variables) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.3 * ncols, 3.2 * nrows), dpi=160,
                             squeeze=False)

    for i, var in enumerate(variables):
        ax = axes[i // ncols][i % ncols]
        train_vals = df[var].astype(float).to_numpy()
        rng = np.random.default_rng(0)
        ax.scatter(np.zeros(len(train_vals)) + rng.normal(0, 0.045, len(train_vals)),
                   train_vals, s=10, color=C_GREY, alpha=0.45, linewidth=0,
                   label=f"training data (n={len(train_vals)})")
        cand_vals = cand[var].astype(float).to_numpy()
        colors = [C_WARN if "OUTSIDE" in str(f) or "SPARSE" in str(f) else C_GOOD
                  for f in cand["risk_flags"]]
        ax.scatter(np.ones(len(cand_vals)), cand_vals, s=90, color=colors,
                   edgecolor="white", linewidth=1.0, zorder=3, label="proposed candidates")
        for x, y, rank in zip(np.ones(len(cand_vals)), cand_vals, cand["rank"]):
            ax.annotate(f"#{int(rank)}", (x, y), textcoords="offset points",
                        xytext=(8, 4), fontsize=8)
        ax.axhspan(bounds[var]["effective_bounds"][0], bounds[var]["effective_bounds"][1],
                   color=C_PRIMARY, alpha=0.06, zorder=0)
        ax.set_xlim(-0.45, 1.55)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["observed", "proposed"], fontsize=8.5)
        ax.set_ylabel(f"{var} ({bounds[var]['unit']})", fontsize=8.5)
        ax.set_title(var, fontsize=9)
        style_axes(ax)
        if i == 0:
            ax.legend(fontsize=7.5, frameon=False, loc="best")

    for j in range(len(variables), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.suptitle("Candidate parameters vs observed data (shaded = effective bounds)", fontsize=10)
    save_fig(fig, out_png, plt)


def plot_candidate_ranking(cand, y_train, direction: str, out_png, plt) -> None:
    fig, ax = plt.subplots(figsize=(6.6, max(2.6, 0.62 * len(cand) + 1.2)), dpi=160)
    labels = [f"#{int(r['rank'])}  {r['candidate_id']}" for _, r in cand.iterrows()]
    values = cand["predicted_target"].astype(float).to_numpy()
    colors = [C_WARN if "OUTSIDE" in str(f) or "SPARSE" in str(f) or
              "PREDICTED_OUTSIDE" in str(f) else C_PRIMARY for f in cand["risk_flags"]]
    ypos = np.arange(len(cand))[::-1]
    ax.barh(ypos, values, color=colors, height=0.6)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.axvline(float(y_train.max()), color=C_GREY, linestyle=":", linewidth=1.2,
               label=f"observed max = {float(y_train.max()):.4g}")
    ax.axvline(float(y_train.min()), color=C_GREY, linestyle=":", linewidth=1.2,
               label=f"observed min = {float(y_train.min()):.4g}")
    for y, v in zip(ypos, values):
        ax.text(v, y, f"  {v:.4g}", va="center", fontsize=8)
    ax.set_xlabel("Predicted target (PREDICTED, unverified)")
    ax.set_title(f"Candidate ranking — objective: {direction}", fontsize=10)
    ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    style_axes(ax)
    save_fig(fig, out_png, plt)
