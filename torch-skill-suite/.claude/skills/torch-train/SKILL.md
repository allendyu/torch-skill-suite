---
name: torch-train
description: This skill should be used when the user asks to build or automate a PyTorch training loop, define optimizer or scheduler setup, add checkpointing and logging, support AMP or resume flow, create trainer scaffolding, or mentions 训练循环、trainer、optimizer、scheduler、checkpoint、resume、混合精度、train config in a deep learning workflow. Prefer this skill whenever the main need is making training run reliably rather than choosing model architecture or evaluating results.
version: 0.1.0
---

# Torch Train Skill

## Purpose / 目的

Provide a focused workflow for building a reliable PyTorch training system.
为 PyTorch 训练系统生成提供专用工作流，帮助 Claude 把模型与数据契约落成最小可运行、可恢复、可观测的训练闭环。

## When to use / 何时使用

Use this skill when the main task is to make training run correctly and repeatedly.
当主要任务是“把训练跑起来，并且能持续跑、恢复跑、记录跑”时，使用这个 skill。

Typical triggers include:
常见触发场景包括：

- Generate a training loop or trainer module
- Wire optimizer, scheduler, and loss
- Add checkpointing, logging, resume, AMP, or gradient accumulation
- Produce `train_config.yaml`
- “给我补训练循环” / “帮我写 trainer” / “加 checkpoint 和 resume”
- “基于 data/model contract 生成 train.py” / “配置 optimizer 和 scheduler”

Do not use this skill when the main request is architecture design, metric analysis, or deployment export.
如果主要请求是模型结构设计、评估分析或部署导出，不要优先使用这个 skill。

## Core workflow / 核心工作流

1. Read the data and model contracts first.
   先读取数据契约与模型契约，避免训练环与上下游脱节。
2. Build the smallest training path that can complete a real step or epoch.
   先构建能真实跑通 step 或 epoch 的最小训练路径。
3. Add optimizer, scheduler, and loss only after interfaces are clear.
   在接口明确后再接 optimizer、scheduler、loss。
4. Add observability and recovery features next.
   再加入日志、checkpoint、resume、best model 保存等可观测与恢复能力。
5. Treat AMP, accumulation, and distributed support as layered enhancements.
   把 AMP、梯度累积、多卡支持视为分层增强，而不是最初就做成大一统系统。
6. Produce or update `train_config.yaml` so experiments remain reproducible.
   生成或更新 `train_config.yaml`，保证实验配置可复现。

## Inputs to gather / 需要收集的输入

Gather the minimum training-defining inputs:
收集定义训练流程的最小输入：

- Data contract / 数据契约
- Model contract / 模型契约
- Loss target / loss 定义
- Optimizer preference / optimizer 偏好
- Scheduler preference / scheduler 偏好
- Epochs, batch size hints, device constraints / 训练轮数、批大小、设备约束
- Need for AMP, accumulation, resume, early stopping, or multi-GPU / 是否需要混合精度、梯度累积、恢复训练、早停、多卡

If the user gives too many optional features at once, prioritize a working baseline before advanced features.
如果用户一次要求很多高级功能，先保证基线训练可运行，再逐步加增强项。

## Expected outputs / 期望产出

Typical outputs include:
常见产出包括：

- Training loop or trainer scaffolding
- Optimizer / scheduler / loss wiring plan
- Checkpoint and resume flow
- Logging hooks or experiment record structure
- `train_config.yaml`
- Clear run assumptions and entrypoints

Prefer outputs that can complete a single training step or a full epoch without ambiguity.
产出应尽量能明确支撑单步或整轮训练，不留关键歧义。

## Boundaries / 边界

This skill owns training execution flow, not upstream data modeling or downstream deployment.
这个 skill 负责训练执行流，不负责上游数据建模或下游部署。

Do:
可做：

- Training loop design
- Loss / optimizer / scheduler wiring
- Checkpoint and logging flow
- Resume and reproducibility structure

Do not do:
不要做：

- Redefine dataset semantics already owned by `torch-data`
- Redesign backbone/head architecture already owned by `torch-model`
- Own full evaluation analysis and tuning reporting
- Build inference API or export pipeline

If model or data interfaces are unclear, stop and surface the contract gap instead of guessing through the training loop.
如果模型或数据接口不清楚，应先暴露契约缺口，而不是在训练循环里硬猜。

## Collaboration with other skills / 与其他 skill 的协作

This skill consumes upstream contracts and unblocks downstream evaluation.
这个 skill 消费上游契约，并为下游评估与部署提供训练产物。

Recommended handoff:
推荐交接方式：

- Read project scope from `../../../../shared/contracts/project_spec.example.yaml`
- Consume `../../../../shared/contracts/data_contract.example.yaml`
- Consume `../../../../shared/contracts/model_contract.example.yaml`
- Produce training-facing outputs that can feed `torch-eval-tune`
- Pass checkpoints and training assumptions to `torch-infer-deploy` when inference work begins

Global references:
全局参考：

- `../../../../docs/workflow.md`
- `../../../../docs/architecture.md`

## Working rules / 工作规则

- Prefer a runnable baseline before feature richness.
- 先确保能跑，再追求功能完整。
- Make training state explicit: config, seeds, checkpoints, log paths.
- 明确训练状态：配置、随机种子、checkpoint、日志路径。
- Avoid mixing evaluation policy or deployment details into trainer code unless required.
- 除非必要，不把评估策略或部署细节混进训练主逻辑。
- Keep the trainer compatible with later tuning and resume workflows.
- 让训练结构天然兼容后续调参与断点恢复。
- Surface unresolved assumptions clearly.
- 对未决假设要显式说明，不要静默硬编码。

## Additional resources / 附加资源

Shared contract references:
共享契约参考：

- `../../../../shared/contracts/project_spec.example.yaml`
- `../../../../shared/contracts/data_contract.example.yaml`
- `../../../../shared/contracts/model_contract.example.yaml`

As this skill evolves:
后续可在当前目录补充：

- `references/` for training patterns, AMP/DDP notes, and checkpoint conventions
- `examples/` for trainer layouts and run examples
- `scripts/` for smoke-train or resume validation helpers
