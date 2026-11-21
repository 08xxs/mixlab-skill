"""P0-FR05/FR06 — research report assembled only from computed values."""

from __future__ import annotations

import json

from common import fmt


def build_report(ctx: dict) -> str:
    cfg = ctx["config"]
    obj = cfg.get("objective") or {}
    direction = obj.get("direction")
    gate = ctx.get("gate") or {}
    results = ctx.get("model_results") or {}
    best = ctx.get("best_model")
    audit = ctx.get("audit") or {}

    def model_table() -> str:
        if not results:
            return "| — | — | — | — | — |\n"
        lines = []
        for name, r in results.items():
            mt = r["metrics_test"]
            lines.append(
                f"| {name} | {fmt(r['metrics_train']['MAE'])} | {fmt(r['metrics_test']['MAE'])} "
                f"| {fmt(mt['RMSE'])} | {fmt(mt['R2'])} "
                f"| {fmt(r['cv_mae_mean']) if r['cv_mae_mean'] is not None else '—'} |"
            )
        return "\n".join(lines)

    def candidate_block() -> str:
        cand = ctx.get("candidates")
        if cand is None or len(cand) == 0:
            return f"未生成候选清单。阻断状态：`{ctx.get('status')}`\n\n原因：{ctx.get('blocking_reason')}"
        cols = [c for c in cand.columns if c not in {"reason", "units", "data_label"}]
        head = "| " + " | ".join(cols) + " |\n|" + "|".join([" --- "] * len(cols)) + "|\n"
        rows = []
        for _, r in cand.iterrows():
            vals = []
            for c in cols:
                v = r[c]
                vals.append(f"{v:.6g}" if isinstance(v, float) else str(v))
            rows.append("| " + " | ".join(vals) + " |")
        return head + "\n".join(rows) + "\n"

    outputs = ctx.get("outputs") or []
    out_table = "\n".join(
        f"| `{o['name']}` | {o['size_bytes']} | `{o['sha256'][:16]}…` |" for o in outputs
    ) or "| — | — | — |"

    warnings = ctx.get("warnings") or []
    warn_block = "\n".join(f"- {w}" for w in warnings) or "- 无"

    return f"""# MixLab 运行报告

> 由 `run_mixlab.py`（Skill `{ctx['skill_name']}` v{ctx['skill_version']}）实际运行生成。
> 所有数值来自对输入文件的真实计算；候选方案均为 `{ctx.get('data_label_candidate', 'UNVERIFIED_RECOMMENDATION')}`。

## 1. 研究目标

- 目标列：`{cfg.get('target_column')}`，方向：**{direction}**
{f"- 目标值：{obj.get('target_value')}" if direction == "target" else ""}
- 可调变量：{", ".join(f"`{v}`" for v, s in cfg.get("variables", {}).items() if s.get("role") == "controllable") or "无"}
- 固定条件：{", ".join(f"`{v}` = {s.get('fixed_value')}" for v, s in cfg.get("variables", {}).items() if s.get("role") == "context") or "无"}

## 2. 数据来源

| 项 | 值 |
| --- | --- |
| 文件 | `{ctx['input_meta']['path']}` |
| 大小 | {ctx['input_meta']['size_bytes']} B |
| SHA-256 | `{ctx['input_meta']['sha256']}` |
| 数据类型标签 | **{ctx['provenance']}** |
| 来源说明 | {cfg.get('data_provenance', {}).get('source')} |
| 样本量 | {audit.get('n_rows')} 行 × {audit.get('n_columns')} 列 |

原始文件未被修改；全部分析基于内存中的副本。

## 3. 验证方法

- 划分：`{(cfg.get('split') or {}).get('method')}`，seed={(cfg.get('split') or {}).get('seed')}，
  测试比例 {(cfg.get('split') or {}).get('test_size')}；
  训练 {ctx['split_info'].get('n_train')} / 测试 {ctx['split_info'].get('n_test')}。
- 预处理仅在训练折内拟合（sklearn Pipeline + StandardScaler）。
- 目标变换：`{(cfg.get('model') or {}).get('target_transform', 'none')}`。
- 基线：`dummy_mean`（训练集均值）；候选模型：linear，必要时 random_forest。
- 质量门槛：最佳模型测试 MAE 需比均值基线低至少
  {ctx.get('gate', {}).get('min_improvement_pct', 0)}%。

| 模型 | 训练 MAE | 测试 MAE | 测试 RMSE | 测试 R² | 训练内 CV MAE |
| --- | --- | --- | --- | --- | --- |
{model_table()}

**门槛结论：** {'通过' if gate.get('passed') else '未通过'} — {gate.get('reason')}

## 4. 观察结果

- 目标列分布：均值 {fmt(audit.get('target_stats', {}).get('mean'))}，中位数
  {fmt(audit.get('target_stats', {}).get('median'))}，范围
  [{fmt(audit.get('target_stats', {}).get('min'))}, {fmt(audit.get('target_stats', {}).get('max'))}]。
- 缺失单元格 {audit.get('total_missing_cells')} 个；重复行 {audit.get('duplicate_rows')} 行。
- 数据审计仅标记异常，未删除任何记录（策略：`flag_only`）。
- 最佳模型：`{best}`。

## 5. 候选建议

{candidate_block()}

所有候选均为模型预测，**尚未经过任何真实实验验证**，不得作为实验结果引用。
约束生效范围：见 `run_config.json` 中的 `variables.*.bounds` 与 `effective_bounds`（候选只在与训练数据重叠的区间内生成）。

## 6. 局限与下一步

- 单次划分，未做前瞻性验证；候选的排序只反映当前模型。
- 风险标记（SPARSE_NEIGHBORHOOD / OUTSIDE_TRAIN_P05_P95 / PREDICTED_OUTSIDE_OBSERVED_RANGE）
  只提示不确定性，不代表已经校准的置信区间。
- 下一步：由研究人员核对物理/安全约束后，挑选 1–3 组做真实实验，并用新数据回灌复跑。

## 7. 复现

```bash
python3 scripts/run_mixlab.py --input {ctx['input_meta']['path']} --config {ctx['config_path']} --outdir <输出目录>
```

## 8. 运行产物

| 文件 | 字节 | SHA-256 |
| --- | --- | --- |
{out_table}

## 9. 警告

{warn_block}
"""
