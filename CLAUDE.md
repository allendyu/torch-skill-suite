# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Torch Skill Suite** is a Claude Code skill suite for automating PyTorch deep learning engineering workflows. It consists of six specialized skills that follow a sequential, contract-driven workflow:

1. `torch-data` — Data engineering (Dataset/DataLoader, preprocessing, data validation)
2. `torch-model` — Model construction (backbone selection, head design, model scaffolding)
3. `torch-train` — Training loop generation (optimizer, scheduler, checkpointing, logging)
4. `torch-eval-tune` — Validation and tuning (metrics, error analysis, hyperparameter suggestions)
5. `torch-infer-deploy` — Inference and deployment (TorchScript/ONNX export, FastAPI service)
6. `torch-engineering` — Engineering extension and standardization (project structure, testing, CI)

## Architecture

### Skill-Based Design
Each skill is a self-contained Claude Code skill located in `torch-skill-suite/.claude/skills/`. Skills communicate via shared **contract files** (YAML) that define interfaces between stages. The recommended flow is sequential (1→6), but skills can be used independently if contracts are available.

### Contract-Driven Workflow
The suite uses four core contract schemas (JSON Schema) to ensure consistency across skills:
- `shared/schemas/project_spec.schema.json` — Global task definition
- `shared/schemas/data_contract.schema.json` — Data specification (input shape, splits, preprocessing)
- `shared/schemas/model_contract.schema.json` — Model specification (backbone, head, loss compatibility)
- `shared/schemas/deploy_contract.schema.json` — Deployment specification (export format, service type)

Canonical example contracts are in `shared/contracts/`; each file is a single schema-valid contract that can be used directly as a scaffold input. Scenario examples and recipes are in `shared/examples/contracts/`. Each skill consumes contracts from previous stages and produces contracts for downstream stages.

### Skill Boundaries
Each skill's responsibilities and boundaries are documented in its `SKILL.md` file (e.g., `torch-skill-suite/.claude/skills/torch-data/SKILL.md`). Key principles:
- `torch-data` owns data preparation, not model design.
- `torch-model` owns model structure, not training policy.
- `torch-train` owns training loops, not deployment.
- Skills hand off via explicit contracts to avoid ambiguity.

## Common Development Tasks

### Validating Contracts
Use the validation script to check `data_contract.yaml` against the JSON schema:
```bash
python torch-skill-suite/.claude/skills/torch-data/scripts/validate_contract.py --contract path/to/data_contract.yaml
```

### Inspecting Datasets
The inspection script attempts to infer dataset format:
```bash
python torch-skill-suite/.claude/skills/torch-data/scripts/inspect_dataset.py --path /path/to/dataset --data_type image --task_type classification
```

### Running Skills
Skills are automatically triggered when user requests match their descriptions (see system-reminder). To manually invoke a skill, use the `/skill` command with the skill name (e.g., `/skill torch-data`).

## Important Directories

- `torch-skill-suite/.claude/skills/` — Skill definitions (SKILL.md files)
- `torch-skill-suite/shared/schemas/` — JSON Schema files for contracts
- `torch-skill-suite/shared/contracts/` — Canonical schema-valid contract example YAML files
- `torch-skill-suite/shared/examples/contracts/` — Scenario contract examples and recipes
- `torch-skill-suite/examples/` — Placeholder example workspaces (MVP in progress)
- `torch-skill-suite/docs/` — Architecture and workflow documentation (currently placeholders)

## Current State

- **MVP phase**: All six skills now have initial implementations, with the image-classification path serving as the primary MVP route.
- **Documentation**: `docs/` contains placeholder files; refer to `torch_skill_suite_plan.md` in the repository root for the comprehensive design.
- **Examples**: `shared/contracts/` contains canonical schema-valid examples; `shared/examples/contracts/` and skill-local `examples/` directories contain scenario-specific examples.

## Key Design Principles

1. **Minimal reliable pipelines** over speculative abstractions.
2. **Explicit interfaces** via contract files.
3. **Observations over assumptions** — skills should inspect existing project structure before generating code.
4. **Downstream compatibility** — each skill’s output should be directly usable by the next skill.
5. **Template-based generation** where possible, with Claude filling parameters and adapting structure.

## Notes for Future Claude Code Instances

- Always check for existing contract files (`data_contract.yaml`, `model_contract.yaml`, etc.) before generating code.
- Prefer extending existing project structure over wholesale replacement.
- When a skill is triggered, read its SKILL.md to understand boundaries and collaboration patterns.
- The shared schemas are authoritative for contract validation; use them to ensure interoperability.
- The ultimate goal is a fully automated PyTorch workflow where users can start with raw data and end with a deployed, engineered project via sequential skill invocations.