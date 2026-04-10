---
name: torch-infer-deploy
description: This skill should be used when the user asks to build PyTorch inference code, export a model to TorchScript or ONNX, package batch or online inference, add a FastAPI or service wrapper, benchmark inference latency, prepare deployment artifacts, or mentions 推理、部署、导出、TorchScript、ONNX、serving、API、benchmark、deployment contract in a deep learning workflow. Prefer this skill whenever the main need is taking a trained model into inference or serving rather than training it or analyzing its validation metrics.
version: 0.1.0
---

# Torch Infer Deploy Skill

## Purpose / 目的

Provide a focused workflow for PyTorch inference packaging and deployment preparation.
为 PyTorch 模型提供推理封装与部署准备工作流，帮助 Claude 把训练完成的模型转化为可调用、可服务化、可评估性能的推理系统。

## When to use / 何时使用

Use this skill when the main question is “how do we run or serve this trained model reliably?”
当核心问题是“训练好的模型如何稳定推理、导出、服务化”时，使用这个 skill。

Typical triggers include:
常见触发场景包括：

- Build `infer.py` or batch inference entrypoints
- Export to TorchScript or ONNX
- Wrap a model with FastAPI, TorchServe-style, or service scaffolding
- Add preprocessing / postprocessing around inference
- Benchmark latency or throughput
- Produce `deploy_contract.yaml`
- “帮我做推理代码” / “导出 ONNX” / “做 API 服务化”
- “把模型包装成部署产物” / “测一下推理延迟与吞吐”

Do not use this skill if the main problem is data cleaning, model family selection, or validation metric interpretation.
如果主要问题是数据清洗、模型选型或验证指标解释，不要优先使用这个 skill。

## Core workflow / 核心工作流

1. Start from an already-defined trained model interface.
   从已经明确的训练后模型接口出发，不跳过模型输入输出定义。
2. Align preprocessing and postprocessing with the training side.
   推理前处理与后处理要与训练侧保持对齐。
3. Deliver local inference before full serviceization.
   先交付本地推理路径，再扩展到服务化。
4. Add export formats only after correctness is clear.
   先确认推理结果正确，再做 TorchScript 或 ONNX 导出。
5. Treat benchmarking as part of the deployment path.
   把性能测试视为部署的一部分，而不是事后补充。
6. Produce or update `deploy_contract.yaml` to stabilize runtime assumptions.
   生成或更新 `deploy_contract.yaml`，把运行时约定固定下来。

## Inputs to gather / 需要收集的输入

Gather the minimum deployment-defining inputs:
收集定义部署路径所需的最小输入：

- Trained checkpoint or exported artifact / 训练好的 checkpoint 或导出模型
- Input and output semantics / 输入输出语义
- Preprocess and postprocess expectations / 前后处理要求
- Target runtime form: local CLI, batch, API, service / 目标运行形态
- Platform or latency constraints / 平台与延迟约束
- Required export format / 所需导出格式

If inference assumptions are not aligned with training, highlight the mismatch first.
如果推理假设与训练侧不一致，应先明确指出不一致点。

## Expected outputs / 期望产出

Typical outputs include:
常见产出包括：

- Local inference scaffolding
- Batch or online inference targets
- Export path for TorchScript / ONNX
- Service wrapper targets
- Benchmark guidance or scripts
- `deploy_contract.yaml`

Prefer outputs that preserve correctness first, then optimize portability or performance.
产出优先保证正确性，再追求可移植性与性能。

## Boundaries / 边界

This skill owns inference packaging and deployment preparation, not upstream training or project-wide engineering policy.
这个 skill 负责推理封装与部署准备，不负责上游训练流程或整体工程治理策略。

Do:
可做：

- Inference path design
- Export format planning
- Service wrapper structure
- Runtime benchmarking targets
- Deployment contract definition

Do not do:
不要做：

- Redesign dataset ownership
- Replace the core training loop
- Own experiment analysis and hyperparameter tuning
- Absorb all long-term repo standardization work

If the real issue is model quality rather than inference packaging, hand back to `torch-eval-tune` first.
如果真正问题是模型效果，而不是推理包装，应先交还给 `torch-eval-tune`。

## Collaboration with other skills / 与其他 skill 的协作

This skill usually follows successful training and validation.
这个 skill 通常位于训练与验证之后。

Recommended handoff:
推荐交接方式：

- Consume model semantics aligned with `../../../../shared/contracts/model_contract.example.yaml`
- Consume deployment assumptions aligned with `../../../../shared/contracts/deploy_contract.example.yaml`
- Read training-side expectations from upstream outputs
- Hand engineering standardization concerns to `torch-engineering` if packaging grows beyond a simple deploy path

Shared references:
共享参考：

- `../../../../shared/contracts/project_spec.example.yaml`
- `../../../../shared/contracts/model_contract.example.yaml`
- `../../../../shared/contracts/deploy_contract.example.yaml`
- `../../../../docs/workflow.md`
- `../../../../docs/architecture.md`

## Working rules / 工作规则

- Keep preprocessing and postprocessing explicit.
- 明确记录前处理与后处理，不要默认它们“显然存在”。
- Favor a correct local inference path before scaling to services.
- 先保证本地推理正确，再扩展到服务化。
- Treat export as a compatibility step, not a substitute for verification.
- 导出是兼容步骤，不是正确性验证的替代品。
- Measure latency and throughput against a clear runtime assumption.
- 性能测试必须绑定明确运行假设。
- Keep deployment artifacts understandable by downstream maintainers.
- 让部署产物与运行约定对后续维护者保持可读。

## Additional resources / 附加资源

Shared references:
共享参考：

- `../../../../shared/contracts/project_spec.example.yaml`
- `../../../../shared/contracts/model_contract.example.yaml`
- `../../../../shared/contracts/deploy_contract.example.yaml`

As this skill evolves:
后续可在当前目录补充：

- `references/` for export notes, serving patterns, and runtime checklists
- `examples/` for CLI, batch, and API inference examples
- `scripts/` for export validation and benchmark helpers
