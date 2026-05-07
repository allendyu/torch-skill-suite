# CLAUDE.md

本文件为 Claude Code（claude.ai/code）在此仓库中工作时提供指导。

## 项目概览

**Torch Skill Suite** 是一个用于自动化 PyTorch 深度学习工程工作流的 Claude Code skill 套件。它由六个专用 skill 组成，遵循串行的、契约驱动的工作流：

1. `torch-data` —— 数据工程（Dataset/DataLoader、预处理、数据校验）
2. `torch-model` —— 模型构建（backbone 选择、head 设计、模型骨架）
3. `torch-train` —— 训练循环生成（optimizer、scheduler、checkpoint、日志）
4. `torch-eval-tune` —— 验证与调优（指标、错误分析、超参数建议）
5. `torch-infer-deploy` —— 推理与部署（TorchScript/ONNX 导出、FastAPI 服务）
6. `torch-engineering` —— 工程化扩展与标准化（项目结构、测试、CI）

## 架构

### 基于 Skill 的设计
每个 skill 都是一个自包含的 Claude Code skill，位于 `torch-skill-suite/.claude/skills/` 中。各个 skill 通过共享的 **contract 文件**（YAML）通信，这些文件定义了各阶段之间的接口。推荐流程是串行执行（1→6），但如果相关 contract 已存在，skills 也可以单独使用。

### 契约驱动工作流
该套件使用四个核心 contract schema（JSON Schema）来保证各个 skill 之间的一致性：
- `shared/schemas/project_spec.schema.json` —— 全局任务定义
- `shared/schemas/data_contract.schema.json` —— 数据规格（输入 shape、数据切分、预处理）
- `shared/schemas/model_contract.schema.json` —— 模型规格（backbone、head、loss 兼容性）
- `shared/schemas/deploy_contract.schema.json` —— 部署规格（导出格式、服务类型）

规范示例 contract 位于 `shared/contracts/`，文件名使用 `*.example.yaml` 后缀；每个文件都是可直接作为脚手架输入的 schema-valid contract。场景示例和 recipe 位于 `shared/examples/contracts/`。skill 运行时产物通常使用无后缀名称，例如 `data_contract.yaml`、`model_contract.yaml` 和 `deploy_contract.yaml`。每个 skill 会消费上一个阶段的 contract，并产出供下游阶段使用的 contract。

### Skill 边界
每个 skill 的职责和边界都记录在对应的 `SKILL.md` 文件中（例如 `torch-skill-suite/.claude/skills/torch-data/SKILL.md`）。关键原则：
- `torch-data` 负责数据准备，不负责模型设计。
- `torch-model` 负责模型结构，不负责训练策略。
- `torch-train` 负责训练循环，不负责部署。
- skill 之间通过显式 contract 交接，避免歧义。

## 常用开发任务

### 校验 Contract
从包目录（`torch-skill-suite/`）运行校验命令。规范示例文件使用 `*.example.yaml` 后缀：
```bash
cd torch-skill-suite
python .claude/skills/torch-data/scripts/validate_contract.py --contract shared/contracts/data_contract.example.yaml
python .claude/skills/torch-model/scripts/validate_contract.py --contract shared/contracts/model_contract.example.yaml
python .claude/skills/torch-infer-deploy/scripts/validate_contract.py --contract shared/contracts/deploy_contract.example.yaml
```

### 运行测试
由于 `pytest.ini` 位于包目录，请从 `torch-skill-suite/` 运行测试：
```bash
cd torch-skill-suite
python -m pytest
```

当前健康检查：2026-05-07 自检结果为 `199 passed`。测试目前会出现与 `torch.jit.trace`、`torch.jit.save`、`torch.jit.load` 相关的 PyTorch deprecation warning；这些 warning 不会导致测试失败，但部署导出代码后续应跟踪 `torch.export` 迁移路线。

### 检查数据集
检查脚本会尝试推断数据集格式：
```bash
python torch-skill-suite/.claude/skills/torch-data/scripts/inspect_dataset.py --path /path/to/dataset --data_type image --task_type classification
```

### 运行 Skills
当用户请求匹配 skill 描述时，skill 会被自动触发（见 system-reminder）。如需手动调用，可使用 `/skill` 命令并指定 skill 名称（例如 `/skill torch-data`）。

## 重要目录

- `torch-skill-suite/.claude/skills/` —— skill 定义（SKILL.md 文件）
- `torch-skill-suite/shared/schemas/` —— contract 的 JSON Schema 文件
- `torch-skill-suite/shared/contracts/` —— contract YAML 示例文件
- `torch-skill-suite/examples/` —— 示例工作区占位目录（MVP 进行中）
- `torch-skill-suite/docs/` —— 架构与工作流文档（当前仍为占位内容）

## 当前状态

- **MVP 阶段**：六个 skill 均已有初始实现，图像分类路径是当前主要 MVP 路线。
- **验证状态**：从 `torch-skill-suite/` 运行测试时，当前测试套件通过（2026-05-07 自检为 `199 passed`）；规范 contract 示例均可通过校验。
- **文档**：`docs/` 已补充架构、工作流和 MVP 状态说明；完整设计仍可参考仓库根目录中的 `torch_skill_suite_plan.md`。
- **示例**：`shared/contracts/` 包含 schema-valid 的规范 `*.example.yaml` 示例；`shared/examples/contracts/` 和各 skill 本地 `examples/` 目录包含场景化示例。
- **已知 warning**：TorchScript 导出测试目前会触发 PyTorch 对 `torch.jit.*` 的 deprecation warning；短期应保持 TorchScript 可用，同时评估后续迁移到 `torch.export`。

## 关键设计原则

1. 优先 **最小但可靠的 pipeline**，而不是推测性的抽象。
2. 通过 contract 文件提供 **显式接口**。
3. **基于观察而非假设** —— skill 在生成代码前应先检查现有项目结构。
4. **下游兼容性** —— 每个 skill 的输出都应能被下一个 skill 直接使用。
5. 在可行时优先 **基于模板生成**，由 Claude 负责填充参数并适配结构。

## 给未来 Claude Code 实例的说明

- 在生成代码前，始终先检查是否已有 contract 文件（如 `data_contract.yaml`、`model_contract.yaml` 等）。
- 优先扩展现有项目结构，而不是整体替换。
- 当某个 skill 被触发时，先读取它的 `SKILL.md`，理解其职责边界和协作方式。
- 共享 schema 是 contract 校验的权威来源，应基于它们保证互操作性。
- 最终目标是实现一个全自动的 PyTorch 工作流：用户从原始数据出发，通过顺序调用各个 skill，最终得到已部署、已工程化的项目。
