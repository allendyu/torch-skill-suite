# torch-data Development Roadmap

## Strategic Positioning

`torch-data` should be developed as a **standalone, high-coverage data foundation skill** before full suite integration.

At this stage, its goal is not merely to hand off a minimal `data_contract.yaml` to downstream skills, but to become a reusable data-interface layer for PyTorch projects that can:

1. Describe many data modalities in a unified way
2. Support many downstream task types without coupling to one training stack
3. Inspect and validate real-world datasets
4. Generate structured artifacts that later skills can consume after integration

This direction matches the repository strategy of **developing each skill independently first, then integrating and optimizing later**.

---

## Core Goals

Build `torch-data` into a skill that can:

1. Cover broad data modalities:
   - image
   - text
   - time_series
   - tabular
   - audio
   - video
   - multimodal
2. Support broad downstream task families:
   - classification
   - detection
   - segmentation
   - regression
   - generation
   - translation
   - clustering
   - reinforcement_learning
   - and future expansion such as forecasting, sequence labeling, ranking, anomaly detection
3. Support both:
   - `user_provided` dataset format definitions
   - `auto_inferred` dataset format inspection
4. Produce more than one useful artifact:
   - `data_contract.yaml`
   - dataset profile / observed statistics
   - validation report / risk summary
5. Remain independently useful even before `torch-model`, `torch-train`, and the rest of the suite are fully mature

---

## Product Definition

At independent-development stage, `torch-data` should be treated as:

- a **data contract generator**
- a **dataset inspection tool**
- a **data validation tool**
- a **data compatibility description layer**
- later, a **Dataset/DataLoader scaffold generator**

This means the skill should prioritize:

1. describing data correctly
2. checking data quality reliably
3. exposing assumptions explicitly
4. supporting many modality/task combinations

before it prioritizes full downstream integration.

---

## Development Principles

### 1. Prefer broad, explicit data modeling over narrow pipeline coupling
The contract should describe the data world completely enough that many future skills can consume it.

### 2. Prefer inspection and validation before code generation
A correct contract and strong data checks are more valuable than prematurely generating loaders for unsupported cases.

### 3. Distinguish observation from assumption
Inspection results should separate:
- observed facts
- inferred guesses
- missing information requiring user confirmation

### 4. Use staged support levels
Not every modality-task combination needs the same maturity level at first.

### 5. Keep downstream compatibility explicit
Compatibility notes should be written into structured artifacts instead of living only in prose docs.

---

## Capability Layers

## Layer 1 — Contract Support

The first responsibility of `torch-data` is to provide a strong and extensible data contract.

### Current baseline
The current schema already includes:
- `data_type`
- `task_type`
- `input_spec`
- `output_spec`
- `splits`
- `preprocessing`
- `data_format_option`
- `user_format_spec`
- `inferred_format_spec`
- `metadata`

### Next schema direction
The schema should evolve to better support broad modality/task coverage by adding or strengthening areas such as:
- `task_context`
- `sample_spec`
- `validation_report`
- `compatibility`
- stronger conditional validation via `if/then/else`, `oneOf`, and task-specific requirements

### Contract priorities
1. Support multi-modality cleanly
2. Support different supervision types
3. Support paired / unpaired / sequence / dense labels
4. Support downstream hints without locking into one model/trainer design

---

## Layer 2 — Inspection Support

`torch-data` should inspect real datasets and infer structure where possible.

### Current baseline
The current inspector now supports:
- image classification
- image detection
- image segmentation
- text classification
- time-series classification/regression
- tabular classification/regression
- audio classification
- video classification
- multimodal classification (manifest-driven)

The current implementation uses a registry-based single-file inspector with a unified output shape. This is sufficient for the current MVP phase and keeps the CLI stable while support breadth expands.

### Next inspection direction
Keep the current CLI stable while continuing to improve per-modality/task heuristics. If the script becomes hard to maintain, split the internal inspectors into dedicated modules later rather than prematurely introducing more files.

Possible later structure:

```text
inspectors/
├── image_classification.py
├── image_detection.py
├── image_segmentation.py
├── text_classification.py
├── text_seq2seq.py
├── tabular.py
├── time_series.py
├── audio.py
├── video.py
└── multimodal.py
```

### Inspector output should include
- detected format
- confidence
- observed fields
- warnings
- missing information
- suggested contract patch

### Recognized auto-inferred patterns today
- image classification: ImageFolder, flat image directories
- image detection: YOLO-style image/label folders, COCO-style JSON
- image segmentation: image/mask directory pairs, COCO-style JSON hints
- text classification: CSV, JSONL, text-file directories
- tabular: CSV, TSV, JSONL, JSON, Parquet-by-suffix, tabular split directories
- time series: CSV/TSV, NPZ-by-suffix
- audio classification: folder-per-class audio data, manifest-driven audio metadata
- video classification: folder-per-class clip data, frame directories, manifest-driven video metadata
- multimodal: CSV/TSV/JSON/JSONL manifests with multiple detected modalities

### Current limitations
- inspection remains heuristic and should separate observed facts from assumptions
- audio/video support is structure-based only and does not decode media yet
- Parquet and NPZ support is currently suffix-based rather than full schema introspection
- multimodal inspection currently focuses on manifest-driven alignment rather than implicit filename pairing

---

## Layer 3 — Validation Support

`torch-data` should validate not only contract structure but also dataset quality.

### Validation targets by modality

#### image
- corrupted files
- size/channel anomalies
- class imbalance
- bbox validity
- mask label validity

#### text
- empty samples
- encoding issues
- label cleanliness
- sequence length distribution
- source-target alignment

#### tabular
- missing values
- duplicate rows
- target leakage candidates
- extreme values
- categorical cardinality issues

#### time_series
- missing timestamps
- out-of-order records
- irregular sampling frequency
- broken group identities

#### audio/video
- unreadable files
- duration anomalies
- sample-rate / fps inconsistency

### Validation outputs
In addition to schema validation, `torch-data` should eventually produce:
- `data_profile.yaml` or `data_profile.json`
- `data_validation_report.md` or structured YAML

---

## Layer 4 — Scaffold Generation

Only after contract, inspection, and validation are stable should `torch-data` expand Dataset/DataLoader generation.

### Initial scaffold targets
Prioritize scaffold generation for:
- image classification
- image detection
- image segmentation
- text classification
- tabular classification/regression
- time-series windowed datasets

### Later scaffold targets
Add later:
- audio classification
- video classification
- multimodal paired datasets
- more custom collate patterns

---

## Support Matrix Strategy

Support should be tracked across three dimensions:

1. **Contract support** — can the modality/task be expressed cleanly?
2. **Inspect support** — can the skill infer or inspect raw data automatically?
3. **Scaffold support** — can the skill generate PyTorch dataset/loader code?

### Priority support matrix

| Modality | Task | Contract | Inspect | Scaffold |
|---|---|---:|---:|---:|
| image | classification | yes | yes | yes |
| image | detection | yes | yes | partial |
| image | segmentation | yes | yes | partial |
| text | classification | yes | yes | yes |
| text | translation/generation | yes | partial | partial |
| tabular | classification/regression | yes | yes | yes |
| time_series | forecasting/regression/classification | yes | yes | partial |
| audio | classification | yes | yes | no |
| video | classification | yes | yes | no |
| multimodal | paired classification/retrieval | yes | yes | no |

This matrix should be updated as implementation progresses.

---

## Recommended Development Phases

## Phase 1 — Strengthen Contract Coverage

### Objective
Make `torch-data` a strong standalone data description and validation skill.

### Work items
1. Expand task taxonomy where needed
2. Tighten `data_contract.schema.json`
3. Add task-specific conditional validation
4. Add more example contracts across modalities and tasks
5. Clarify field semantics in docs

### Deliverables
- improved schema
- expanded examples
- updated guides
- validator coverage for examples

### Exit criteria
- at least 15–20 representative example contracts
- examples cover major modality/task combinations
- validator passes all valid examples and rejects known invalid ones

---

## Phase 2 — Expand Inspection Coverage

### Objective
Make `torch-data` useful on real datasets even before downstream integration.

### Work items
1. Modularize inspectors
2. Add confidence / warning / missing-info output
3. Support major raw formats:
   - ImageFolder / flat image dirs / COCO / YOLO / image-mask pairs
   - CSV / JSONL / chat-style text datasets
   - CSV / Parquet tabular data
   - CSV / NPZ time-series data
   - WAV-dir / metadata-CSV audio data
   - MP4-dir / frame-dir video data
4. Convert inspection results into contract patches

### Deliverables
- inspector modules
- unified inspection output format
- test fixtures / dummy datasets

### Exit criteria
- each primary modality has at least 2–3 recognizable real-world storage formats
- failed inspection cases return actionable feedback
- inspection output can be translated into contract fields consistently

---

## Phase 3 — Add Data Quality Validation

### Objective
Make `torch-data` valuable even when no code generation is requested.

### Work items
1. Add modality-aware quality checks
2. Generate dataset profile artifacts
3. Generate validation reports and warnings
4. Surface downstream-risk signals such as imbalance, leakage, and invalid annotation structures

### Deliverables
- dataset profiling output
- validation report format
- validation scripts or reusable modules

### Exit criteria
- profile/report generation works on representative datasets
- validation catches common quality failures across major modalities

---

## Phase 4 — Add Scaffold Generation

### Objective
Generate reusable PyTorch Dataset/DataLoader scaffolds once the data layer is stable.

### Work items
1. Generate dataset code for priority modality/task combinations
2. Add transform pipeline wiring
3. Add collate guidance for variable-length or structured labels
4. Keep generated code readable and minimally opinionated

### Deliverables
- scaffold generator(s)
- scaffold templates
- runnable examples for core cases

### Exit criteria
- generated scaffolds work for priority scenarios
- outputs remain aligned with the contract and validation assumptions

---

## Immediate Next Steps

1. Tighten `data_contract.schema.json` with conditional requirements
2. Add examples for:
   - image segmentation
   - tabular regression
   - audio classification
   - video classification
   - multimodal paired data
3. Refactor `inspect_dataset.py` toward modular inspectors
4. Define a structured inspection output format
5. Add automated validation tests for all example contracts
6. Add a support matrix section to README or docs for visibility

---

## Verification Checklist

- [x] Updated `data_contract.schema.json`
- [x] Updated `data_contract.example.yaml`
- [x] Created `references/` with core guides
- [x] Created first-batch examples
- [x] Created validation and inspection scripts
- [x] First scaffold generators for priority scenarios (`generate_dataset.py` covers image classification, text classification, tabular classification/regression, and image segmentation)
- [ ] Tighten schema with task-aware constraints
- [ ] Expand examples across major modalities/tasks
- [ ] Modularize inspectors
- [ ] Define structured inspection outputs
- [ ] Add validation tests for all examples
- [ ] Add dataset profile and validation-report artifacts
- [ ] Extend scaffold generation to audio, video, multimodal, time-series windowing

---

## Notes

- The current schema is intentionally flexible; future iterations should increase strictness where it improves interoperability and validation quality.
- Independent skill development is the current strategy, so `torch-data` should optimize first for standalone usefulness and broad coverage.
- Later suite integration should consume the artifacts `torch-data` already produces rather than forcing `torch-data` into a narrowly coupled downstream shape.
