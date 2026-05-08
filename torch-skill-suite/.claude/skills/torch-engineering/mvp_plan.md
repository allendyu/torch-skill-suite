# torch-engineering Development Roadmap

## Strategic Positioning

`torch-engineering` is the sixth skill in the suite. Its job is to lift the project from "working pipeline" to "maintainable engineering codebase": project structure, configs, tests, lint, CI, registries, and extension points. It runs **last as a stabilizer** in the canonical workflow, but can be invoked earlier when structure problems block progress in the other five skills.

## Current Status

This skill is **documentation-only**. `SKILL.md` defines its scope, boundaries, inputs, outputs, and collaboration patterns, but no scripts, templates, or tests exist yet. The other five production skills already ship runnable scripts and tests.

The repository today already has:

- A package-level `pytest.ini` with explicit `testpaths` per skill
- A `conftest.py` that bootstraps `shared/python` onto `sys.path`
- A GitHub Actions CI workflow that runs `flake8`, contract validators, and per-skill pytest suites on Python 3.12
- Shared utilities under `shared/python/torch_skill_shared/` consumed by all five production skills
- A `shared/route_map.yaml` that already acts as a lightweight route registry

`torch-engineering` should build on these primitives rather than replace them.

## Core Goals

When activated, `torch-engineering` should be able to:

1. Inspect repository layout and surface concrete gaps (missing tests, inconsistent configs, scripts not under `pytest.ini` testpaths, etc.)
2. Standardize project structure for generated workspaces (downstream of the other five skills)
3. Introduce or reinforce a config system convention (start with YAML + Pydantic; consider Hydra later)
4. Add or extend test, lint, and type-check gates
5. Provide a registry / extension-point pattern that's compatible with the existing `route_map.yaml`
6. Stage refactors so the working pipeline never breaks during cleanup

## Architecture (planned)

```
existing repo ──→ inspect_project.py ──→ engineering_report.yaml
                        │
                        └──→ apply_normalization.py (optional, staged) ──→ refactored repo
```

No directory under `.claude/skills/torch-engineering/` exists for `scripts/`, `templates/`, or `tests/` yet — the first concrete deliverable should create them.

## Development Phases

### Phase 1 — P0: Inspection skill (planned)
- [ ] `scripts/inspect_project.py` — read repo, emit a structured `engineering_report.yaml` listing gaps and proposed remediations (no writes)
- [ ] `tests/` smoke test on a sample repo fixture
- [ ] Reuse `torch_skill_shared.yaml_utils` for IO

### Phase 2 — P1: Config normalization
- [ ] Templated `configs/` layout with YAML + Pydantic schemas
- [ ] Migration helper that lifts inline constants into config files without breaking existing entrypoints
- [ ] Documentation pattern: one config schema per stage (data / model / train / eval / deploy)

### Phase 3 — P2: Quality gates
- [ ] `flake8` / `ruff` / `mypy` recommended config templates
- [ ] Pre-commit hook scaffold
- [ ] CI workflow snippets for downstream user repos (not just this skill suite)

### Phase 4 — P3: Registry / extension points
- [ ] Registry pattern that aligns with `shared/route_map.yaml`
- [ ] Plugin discovery convention for adding new routes without forking the suite

## Boundaries

- Does **not** redesign data, model, training, evaluation, or deployment behavior — those belong to the other five skills
- Does **not** prematurely introduce framework-heavy machinery (Hydra, Lightning, etc.) without clear payoff
- Prefers staged refactors over broad rewrites

## Verification Checklist

- [x] `SKILL.md` defining scope, inputs, outputs, and boundaries
- [ ] `scripts/inspect_project.py` for read-only repository diagnosis
- [ ] First test fixture and smoke test
- [ ] Config-system convention with at least one runnable example
- [ ] CI / lint snippets that downstream user projects can adopt
- [ ] Registry pattern compatible with `route_map.yaml`

## Notes

- Treat this skill as the place to land repository-wide concerns that don't naturally fit any single pipeline stage
- Reuse `shared/python/torch_skill_shared/` rather than duplicating utility code
- The CI workflow at `.github/workflows/ci.yml` is the current ground truth for what "passing" means; engineering changes should keep CI green, not bypass it
