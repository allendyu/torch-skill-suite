---
name: torch-data
description: This skill should be used when the user asks to build or automate PyTorch data engineering, create Dataset or DataLoader code, define preprocessing or augmentation, split datasets, inspect training data quality, generate a data contract, or mentions 数据工程、数据处理、Dataset、DataLoader、预处理、数据切分、样本检查、data contract for a deep learning workflow. Prefer this skill whenever the main need is data preparation rather than model design or training logic.
version: 0.1.0
---

# Torch Data Skill

## Purpose / 目的

Provide a focused workflow for PyTorch data engineering in deep learning projects.
为深度学习项目提供面向 PyTorch 的数据工程工作流，帮助 Claude 把原始数据整理为可训练、可验证、可复用的数据输入层。

## When to use / 何时使用

Use this skill when the request is primarily about data preparation.
当请求的核心是“把数据准备好、组织好、检查好”时，优先使用这个 skill。

Typical triggers include:
常见触发场景包括：

- Build a `Dataset` or `DataLoader`
- Create preprocessing or augmentation pipelines
- Split train / val / test data
- Inspect label mapping, class imbalance, missing values, bad samples, or shape issues
- Generate `data_contract.yaml`
- “帮我做数据工程” / “给这个项目补一个 Dataset” / “把图像目录做成 DataLoader”
- “为当前项目生成数据预处理流程” / “检查样本和标签是否有问题”

Do not use this skill if the main task is choosing model architecture, writing the training loop, or exporting a model for deployment.
如果主要任务是模型结构设计、训练循环生成或部署导出，不要优先使用这个 skill。

## Core workflow / 核心工作流

1. Identify task type and data modality.
   识别任务类型与数据模态，如 classification、segmentation、detection、NLP、timeseries。
2. Locate the actual data source and inspect the current project structure.
   找到真实数据来源与项目现有结构，而不是凭空假设数据格式。
3. Gather only the missing information required to define the data interface.
   只补齐决定数据接口所必需的信息，例如标签字段、输入路径、切分策略、变换需求。
4. Design the minimal reliable Dataset / DataLoader path.
   优先给出最小可运行、可检查、可复用的数据加载结构。
5. Add validation checks before expanding complexity.
   先补数据检查与 sanity check，再考虑高级增强与复杂抽象。
6. Produce or update `data_contract.yaml` so later skills can consume a stable interface.
   生成或更新 `data_contract.yaml`，为后续 `torch-model` 和 `torch-train` 提供稳定输入。

## Inputs to gather / 需要收集的输入

Collect the minimum set of data-facing requirements:
收集最小必要的数据侧输入：

- Task type / 任务类型
- Input modality / 输入模态
- Data source path or storage layout / 数据源路径或存储结构
- Label field or target definition / 标签字段或目标定义
- Split policy / 切分策略
- Transform, normalization, tokenization, or augmentation requirements / 预处理与增强需求
- Batch constraints such as image size, sequence length, batch size hints / 批大小、尺寸、序列长度等约束

If key information is missing, summarize the missing fields explicitly before generating code.
如果关键信息缺失，先明确列出缺的字段，再继续生成代码或结构建议。

## Expected outputs / 期望产出

Typical outputs include:
常见产出包括：

- Dataset / DataLoader scaffolding
- Transform or preprocessing targets
- Split logic or dataset indexing structure
- Sanity-check guidance for samples, labels, shapes, and imbalance
- `data_contract.yaml`
- `inferred_format_spec` plus inspection metadata such as `confidence`, `warnings`, `observed_fields`, and `missing_information`
- Clear assumptions about data layout and mapping

Prefer outputs that can be used directly by `torch-model` and `torch-train`.
产出应尽量能直接被 `torch-model` 与 `torch-train` 消费。

## Current inspection coverage / 当前检查覆盖范围

Current implemented inspection coverage includes:
当前已实现的数据检查覆盖包括：

- image: classification / detection / segmentation
- text: classification
- time_series: classification / regression
- tabular: classification / regression
- audio: classification
- video: classification
- multimodal: manifest-driven classification

Current recognizable patterns include:
当前可识别的常见模式包括：

- `ImageFolder`, flat image directories
- YOLO-style image/label directories, COCO-style JSON hints
- image/mask directory pairs for segmentation
- CSV / TSV / JSONL / JSON tabular or manifest files
- NPZ-by-suffix time-series inputs
- audio/video metadata manifests
- frame directories for video
- multimodal manifests such as image+text or audio+text rows

Inspection is heuristic rather than guaranteed truth.
数据检查是启发式推断，不应被视为绝对真值。

## Boundaries / 边界

This skill owns data preparation, not the whole project.
这个 skill 负责数据准备，不负责整个项目的其他模块。

Do:
可做：

- Dataset design
- Dataloader structure
- Transform pipeline design
- Data validation and contract definition

Do not do:
不要做：

- Choose backbone or architecture family
- Design optimizer / scheduler strategy
- Own evaluation metrics strategy
- Export or serve trained models

If the user starts from data but the real blocker is model or training design, finish the data-facing contract first, then hand off to the correct downstream skill.
如果请求从数据切入，但真正阻塞点是模型或训练设计，先把数据契约整理清楚，再交给对应下游 skill。

## Collaboration with other skills / 与其他 skill 的协作

This skill usually works first.
这个 skill 通常是工作流起点。

Recommended handoff:
推荐交接方式：

- Start from `../../../../shared/contracts/project_spec.example.yaml` if the project goal is still broad.
- Produce a stable data interface aligned with `../../../../shared/contracts/data_contract.example.yaml`.
- Hand off to `torch-model` once input shape, target definition, and dataset semantics are clear.
- Hand off to `torch-train` after the model contract exists and the data loading path is stable.

Use shared workflow references when available:
如需查看全局工作流约定，可参考：

- `../../../../docs/workflow.md`
- `../../../../docs/architecture.md`

## Working rules / 工作规则

- Prefer minimal reliable data pipelines over speculative abstractions.
- 优先生成最小可靠的数据管线，不为假设中的未来需求做复杂抽象。
- Reuse existing project structure if present instead of replacing it wholesale.
- 如果项目里已经有部分数据代码，优先复用与补全，而不是整套推翻。
- Validate at the data boundary: paths, labels, shapes, missing values, corrupted samples.
- 在数据边界做验证：路径、标签、shape、缺失值、坏样本。
- Distinguish observation from assumption.
- 明确区分“已观察到的信息”和“暂时假设的信息”。
- Keep later skills unblocked by making the contract explicit.
- 通过明确契约，降低后续 skill 的歧义成本。

## Additional resources / 附加资源

Use the shared contracts as the canonical cross-skill interface:
跨 skill 协作时，优先使用共享契约文件作为统一接口：

- `../../../../shared/contracts/project_spec.example.yaml`
- `../../../../shared/contracts/data_contract.example.yaml`

As this skill evolves:
随着后续迭代，可在当前目录补充：

- `references/` for detailed dataset patterns and modality-specific guidance
- `examples/` for sample data layouts and contract examples
- `scripts/` for reusable validation or conversion helpers
