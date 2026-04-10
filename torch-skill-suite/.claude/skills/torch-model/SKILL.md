---
name: torch-model
description: This skill should be used when the user asks to design or generate a PyTorch model, choose a backbone, define a prediction head, build model scaffolding from a data contract, reason about architecture tradeoffs, or mentions 模型搭建、模型结构、backbone、head、architecture、forward、model contract in a deep learning workflow. Prefer this skill whenever the main need is model structure rather than data preparation, optimizer setup, or deployment.
version: 0.1.0
---

# Torch Model Skill

## Purpose / 目的

Provide a focused workflow for designing and generating PyTorch model structure.
为 PyTorch 模型结构设计与生成提供专用工作流，把任务目标与数据契约转换成清晰、可实现、可训练的模型骨架。

## When to use / 何时使用

Use this skill when the main question is “what model should exist and how should it be structured?”
当核心问题是“模型该长什么样、怎么组织、如何和数据对接”时，使用这个 skill。

Typical triggers include:
常见触发场景包括：

- Build a model from `data_contract.yaml`
- Choose backbone / neck / head layout
- Create modular `nn.Module` scaffolding
- Check shape compatibility and output semantics
- Produce `model_contract.yaml`
- “帮我搭一个 torch 模型骨架” / “给这个任务选 backbone”
- “基于 data contract 生成模型代码” / “设计分类头或分割头”

Do not use this skill when the main request is dataset engineering, optimizer scheduling, experiment evaluation, or model serving.
如果主要问题是数据工程、训练超参、实验评估或部署服务化，不要优先使用这个 skill。

## Core workflow / 核心工作流

1. Read the project objective and the data-facing contract first.
   先读取项目目标与数据契约，而不是直接开始发明模型。
2. Identify the minimal architecture family that fits the task.
   先确定最适合任务的最小架构族，例如 CNN、Transformer、UNet、LSTM，而不是过早复杂化。
3. Separate reusable structure from task-specific head logic.
   把可复用 backbone 与任务相关 head 分开组织。
4. Make input/output semantics explicit.
   明确输入 shape、输出维度、类别数、mask 语义、序列长度等关键接口。
5. Prefer standard, readable PyTorch modules unless there is a clear reason not to.
   除非有明确理由，否则优先标准、清晰、易训练的 PyTorch 模块结构。
6. Produce or update `model_contract.yaml` so training and deployment can proceed consistently.
   生成或更新 `model_contract.yaml`，让 `torch-train` 与 `torch-infer-deploy` 共享同一模型定义。

## Inputs to gather / 需要收集的输入

Gather the minimum architecture-defining inputs:
收集定义模型所需的最小输入：

- Task type / 任务类型
- Data contract / 数据契约
- Output target semantics / 输出目标语义
- Backbone preference or constraints / backbone 偏好与资源约束
- Pretrained preference / 是否预训练
- Deployment or latency constraints / 部署、延迟、体积约束

If architecture choice is still ambiguous, narrow the option set and explain the tradeoff briefly.
如果模型选择仍有歧义，应缩小候选范围，并用简洁方式说明取舍。

## Expected outputs / 期望产出

Typical outputs include:
常见产出包括：

- PyTorch model scaffolding
- Backbone / head breakdown
- Forward path definition
- Shape expectations and compatibility notes
- `model_contract.yaml`
- Brief rationale for the chosen architecture level

Prefer outputs that are easy for `torch-train` to wire into a training loop.
产出应尽量便于 `torch-train` 直接接入训练流程。

## Boundaries / 边界

This skill owns model structure, not training policy or deployment implementation.
这个 skill 负责模型结构，不负责训练策略或部署实现。

Do:
可做：

- Model family selection
- Module layout
- Backbone/head decomposition
- Forward interface definition
- Model contract definition

Do not do:
不要做：

- Dataset split or preprocessing ownership
- Optimizer / scheduler tuning
- Full experiment analysis
- Export / service API implementation

If the user asks for architecture and training at once, complete the architecture contract first, then hand off to `torch-train`.
如果用户同时要求模型与训练，先完成模型契约，再交给 `torch-train`。

## Collaboration with other skills / 与其他 skill 的协作

This skill usually sits between data and training.
这个 skill 通常位于数据与训练之间。

Recommended handoff:
推荐交接方式：

- Read project scope from `../../../../shared/contracts/project_spec.example.yaml`
- Consume data assumptions aligned with `../../../../shared/contracts/data_contract.example.yaml`
- Produce outputs aligned with `../../../../shared/contracts/model_contract.example.yaml`
- Hand off to `torch-train` once model inputs, outputs, and loss-facing semantics are stable
- Hand off to `torch-infer-deploy` later when export or serving becomes relevant

Global references:
全局参考：

- `../../../../docs/workflow.md`
- `../../../../docs/architecture.md`

## Working rules / 工作规则

- Prefer established model families over unnecessary novelty.
- 优先成熟模型家族，不凭空发明复杂结构。
- Make interfaces explicit: input shape, output shape, head semantics, loss compatibility.
- 明确接口：输入 shape、输出 shape、head 语义、与 loss 的兼容关系。
- Keep module boundaries readable and editable.
- 保持模块边界清晰，便于后续训练与部署修改。
- Avoid mixing training concerns into model definitions unless truly required.
- 除非确有必要，不把 optimizer、scheduler 等训练侧逻辑混入模型定义。
- Optimize for clarity and downstream compatibility first.
- 优先保证清晰性与下游兼容性。

## Additional resources / 附加资源

Shared contract references:
共享契约参考：

- `../../../../shared/contracts/project_spec.example.yaml`
- `../../../../shared/contracts/data_contract.example.yaml`
- `../../../../shared/contracts/model_contract.example.yaml`

As this skill evolves:
后续可在当前目录补充：

- `references/` for architecture patterns and task-specific modeling notes
- `examples/` for sample model layouts by task
- `scripts/` for repeated shape checks or model inspection utilities
