---
name: torch-eval-tune
description: This skill should be used when the user asks to evaluate a PyTorch model, compute validation metrics, analyze model errors, compare experiments, diagnose underperformance, suggest tuning directions, or mentions 评估、验证、metrics、error analysis、调优、超参、experiment report、tuning plan in a deep learning workflow. Prefer this skill whenever the main need is understanding model quality and deciding what to improve next rather than building the trainer or exporting the model.
version: 0.1.0
---

# Torch Eval Tune Skill

## Purpose / 目的

Provide a focused workflow for validation, analysis, and tuning guidance in PyTorch projects.
为 PyTorch 项目提供验证、结果分析与调优建议工作流，帮助 Claude 从训练结果中提炼问题、比较实验、形成可执行改进方向。

## When to use / 何时使用

Use this skill when the main question is “how well does the model work, why, and what should improve next?”
当核心问题是“模型效果怎样、问题出在哪、下一步该怎么调”时，使用这个 skill。

Typical triggers include:
常见触发场景包括：

- Generate evaluation code or metric computation flow
- Compare checkpoints or experiments
- Analyze confusion matrix, ROC, PR curve, per-class metrics, IoU, BLEU, or task-specific metrics
- Investigate failure cases or error clusters
- Produce `experiment_report` or `tuning_plan`
- “帮我评估模型效果” / “分析错误样本” / “比较两个实验结果”
- “给出调优建议” / “做 validation 和 metrics 统计”

Do not use this skill when the real task is building the first training loop or exporting a serving endpoint.
如果真正任务是搭训练循环或导出服务接口，不要优先使用这个 skill。

## Core workflow / 核心工作流

1. Read the training context and the target metric definition.
   先读取训练背景与目标指标，不脱离任务目标做空泛评估。
2. Separate evaluation from tuning.
   先给出观察与度量，再给调优建议，不要把两者混成同一层。
3. Prefer task-appropriate metrics over generic defaults.
   优先选与任务贴合的指标，而不是机械套用通用默认值。
4. Identify failure modes, not just aggregate scores.
   除了总体分数，更要定位失败模式、类别偏差、样本簇问题。
5. Turn observations into a short prioritized tuning plan.
   把观察结果收敛成短而明确的优先级调优计划。
6. Produce artifacts that can guide the next experiment cycle.
   输出能直接指导下一轮实验的报告与调优产物。

## Inputs to gather / 需要收集的输入

Gather the minimum evaluation-defining inputs:
收集评估所需的最小输入：

- Checkpoints or experiment outputs / checkpoint 或实验产物
- Validation or test split definition / 验证集或测试集定义
- Task-specific metrics / 任务相关指标
- Baseline or target thresholds / 基线或目标阈值
- Whether qualitative error analysis is needed / 是否需要定性误差分析
- Any prior hypotheses about failure modes / 对失败模式的已有怀疑

If no target metric is defined, surface that gap instead of pretending evaluation is objective.
如果目标指标尚未定义，应明确指出这一缺口，而不是假装评估结论是客观完备的。

## Expected outputs / 期望产出

Typical outputs include:
常见产出包括：

- Evaluation scaffolding
- Metric computation targets
- Structured experiment comparison notes
- Error analysis outputs or guidance
- `tuning_plan.yaml` or equivalent tuning summary
- `experiment_report.md` / JSON-style result summary

Prefer outputs that distinguish facts, interpretations, and suggested next actions.
产出应明确区分事实、解释和下一步建议。

## Boundaries / 边界

This skill owns evaluation and improvement guidance, not raw training execution or deployment packaging.
这个 skill 负责评估与改进建议，不负责底层训练执行或部署打包。

Do:
可做：

- Metric strategy
- Validation/test evaluation structure
- Error analysis
- Experiment comparison
- Prioritized tuning suggestions

Do not do:
不要做：

- Redefine the full trainer stack
- Replace model architecture ownership wholesale
- Build serving APIs or export paths
- Turn every tuning question into blind large-scale search

If model or training contracts are missing, state the dependency first before inventing evaluation assumptions.
如果模型或训练契约缺失，应先指出依赖缺口，再继续评估设计。

## Collaboration with other skills / 与其他 skill 的协作

This skill usually follows `torch-train` and informs the next iteration.
这个 skill 通常接在 `torch-train` 之后，用来驱动下一轮迭代。

Recommended handoff:
推荐交接方式：

- Consume training artifacts and assumptions from `torch-train`
- Use model semantics aligned with `../../../../shared/contracts/model_contract.example.yaml`
- Produce findings that may send work back to `torch-data`, `torch-model`, or `torch-train`
- Pass only deployment-relevant validation conclusions to `torch-infer-deploy`

Shared references:
共享参考：

- `../../../../shared/contracts/project_spec.example.yaml`
- `../../../../shared/contracts/model_contract.example.yaml`
- `../../../../docs/workflow.md`
- `../../../../docs/architecture.md`

## Working rules / 工作规则

- Choose metrics that match the task and business objective.
- 指标选择必须贴合任务目标与业务目标。
- Distinguish score reporting from root-cause analysis.
- 区分“报分数”和“找原因”。
- Prefer short, ranked tuning suggestions over long generic advice lists.
- 优先给出短而有优先级的调优建议，而不是泛泛长清单。
- Avoid overclaiming causality from limited evidence.
- 在证据有限时，不要过度宣称因果关系。
- Make experiment comparisons reproducible and explicit.
- 实验对比应尽量可复现、可追溯、口径一致。

## Additional resources / 附加资源

Shared references:
共享参考：

- `../../../../shared/contracts/project_spec.example.yaml`
- `../../../../shared/contracts/model_contract.example.yaml`
- `../../../../shared/contracts/data_contract.example.yaml`

As this skill evolves:
后续可在当前目录补充：

- `references/` for metric patterns, report templates, and failure analysis checklists
- `examples/` for evaluation reports and experiment comparisons
- `scripts/` for metric aggregation or benchmark helpers
