---
name: torch-engineering
description: This skill should be used when the user asks to standardize or extend a PyTorch project, improve repository structure, organize configs, add testing, linting, CI guidance, introduce registries or extension points, refactor an ML codebase for maintainability, or mentions 工程化、项目结构、配置系统、测试、lint、CI、registry、可维护性、标准化 in a deep learning workflow. Prefer this skill whenever the main need is making the codebase easier to evolve and maintain rather than building one specific data, model, training, evaluation, or deployment module.
version: 0.1.0
---

# Torch Engineering Skill

## Purpose / 目的

Provide a focused workflow for turning generated ML code into maintainable engineering structure.
为深度学习项目提供工程化治理工作流，帮助 Claude 把“能跑”的代码演进成“可维护、可测试、可扩展、可协作”的项目结构。

## When to use / 何时使用

Use this skill when the main need is repository quality, consistency, and long-term maintainability.
当主要需求是仓库质量、一致性与长期可维护性时，使用这个 skill。

Typical triggers include:
常见触发场景包括：

- Normalize project structure
- Add config system conventions
- Introduce tests, linting, formatting, CI guidance
- Create plugin, registry, or extension-friendly layout
- Refactor generated ML code into reusable modules
- “帮我做工程化改造” / “整理项目结构” / “补测试和 lint”
- “引入配置系统” / “让这个训练项目更好维护” / “做 registry 化扩展”

Do not use this skill when the request is narrowly about a single Dataset, model head, training loop, or export script.
如果请求只聚焦某一个 Dataset、模型 head、训练循环或导出脚本，不要优先使用这个 skill。

## Core workflow / 核心工作流

1. Inspect the current repository shape before proposing structure changes.
   先检查当前仓库结构，再提重构方案，不做脱离上下文的“理想化重写”。
2. Separate must-fix engineering gaps from nice-to-have polish.
   区分必须补齐的工程缺口与可选优化项。
3. Standardize configuration, testing, and module boundaries incrementally.
   用渐进方式统一配置、测试和模块边界，而不是一次性大拆大改。
4. Preserve the existing working path while improving extensibility.
   在提升可扩展性的同时，尽量保留现有可运行路径。
5. Introduce conventions only when they reduce maintenance cost.
   只有在能降低维护成本时，才引入新约定和新抽象。
6. Provide a staged path if the requested refactor is broad.
   如果改造范围很大，应先给分阶段路径，再逐步实施。

## Inputs to gather / 需要收集的输入

Gather the minimum engineering-defining inputs:
收集定义工程化改造所需的最小输入：

- Current repository layout / 当前仓库结构
- Existing data/model/train/eval/deploy modules / 已有模块分布
- Pain points: duplication, coupling, config sprawl, test gaps / 主要痛点
- Team or usage constraints / 团队或使用约束
- Preferred config or tooling direction / 配置与工具偏好
- Required quality gates / 所需质量门禁

If the repo already has conventions, align with them unless there is a strong reason not to.
如果仓库已经存在稳定约定，除非有强理由，否则优先对齐而不是推翻。

## Expected outputs / 期望产出

Typical outputs include:
常见产出包括：

- Normalized project structure plan
- Config system targets
- Testing and linting targets
- CI or quality-gate guidance
- Registry / extension-point design targets
- Refactor sequencing suggestions

Prefer outputs that reduce future maintenance cost without blocking the current working pipeline.
产出应尽量降低后续维护成本，同时不阻断现有可运行路径。

## Boundaries / 边界

This skill owns engineering quality and extensibility, not the domain-specific logic of the other five skills.
这个 skill 负责工程质量与可扩展性，不取代其他五个 skill 的领域职责。

Do:
可做：

- Repository normalization
- Config and module organization
- Testing / lint / CI guidance
- Registry and extensibility patterns
- Refactor staging

Do not do:
不要做：

- Re-own data semantics
- Redesign model architecture from scratch without need
- Rebuild the trainer when only a config cleanup is needed
- Replace deployment design unless engineering concerns truly demand it

If the request is actually about one pipeline stage, finish the stage-specific work with the corresponding skill first, then apply engineering cleanup.
如果请求本质上属于某个具体流程阶段，应先由对应 skill 完成业务侧工作，再做工程化整理。

## Collaboration with other skills / 与其他 skill 的协作

This skill can run last as a stabilizer, or earlier when structure problems block progress.
这个 skill 可以在工作流末尾做稳定化，也可以在结构问题阻塞开发时提前介入。

Recommended handoff:
推荐交接方式：

- Read the overall intent from `../../../../shared/contracts/project_spec.example.yaml`
- Align structure with the outputs expected by `torch-data`, `torch-model`, `torch-train`, `torch-eval-tune`, and `torch-infer-deploy`
- Use this skill to normalize shared configs, test harnesses, and extension points across all pipeline stages

Shared references:
共享参考：

- `../../../../docs/workflow.md`
- `../../../../docs/architecture.md`
- `../../../../shared/contracts/project_spec.example.yaml`
- `../../../../shared/contracts/data_contract.example.yaml`
- `../../../../shared/contracts/model_contract.example.yaml`
- `../../../../shared/contracts/deploy_contract.example.yaml`

## Working rules / 工作规则

- Prefer staged refactors over broad rewrites.
- 优先分阶段重构，而不是大范围一次性推翻。
- Standardize only where inconsistency creates real cost.
- 只在不一致确实带来成本时才做标准化。
- Keep generated code understandable by humans.
- 保持生成代码对人类维护者可理解。
- Avoid introducing framework-heavy machinery without clear payoff.
- 没有明确收益时，不要引入过重的工程框架。
- Preserve working behavior while improving structure.
- 在改进结构时尽量保留现有可运行行为。

## Additional resources / 附加资源

Shared references:
共享参考：

- `../../../../docs/workflow.md`
- `../../../../docs/architecture.md`
- `../../../../shared/contracts/project_spec.example.yaml`

As this skill evolves:
后续可在当前目录补充：

- `references/` for config patterns, test strategies, and refactor checklists
- `examples/` for normalized repository layouts
- `scripts/` for project scaffolding or quality checks
