# Snapshot Generation Plan

This document tracks the requirements for generating API snapshots in the `ml-framework-snapshots` repository to support Ghost Mode in `ml-switcheroo`. Snapshots are critical to prevent LLM hallucinations by grounding the agent in accurate API schemas.

## High Priority (Extractors Exist)

The following frameworks already have extractor modules implemented in `src/ml_framework_snapshots/frameworks/`, but their `.json` snapshots have not been generated or added to the active build list.

- [x] **Flax NNX (`flax_nnx`)**
  - [x] Add `flax_nnx` to the `frameworks` list in `generate_all_snapshots.py`.
  - [x] Generate `flax_nnx_v*.json` via the generator script.
- [ ] **PaxML / Praxis (`pax`)**
  - [ ] Add `pax` (or `paxml`) to the `frameworks` list in `generate_all_snapshots.py`.
  - [ ] Generate `pax_v*.json` (and ensure it maps to `paxml` as expected by Switcheroo) via the generator script.

## Medium Priority (New Extractors Required)

These frameworks are explicitly registered in `ml-switcheroo` but lack extraction logic in the snapshots repository. We need to write new Python extractors that dump their APIs into JSON schema format.

- [ ] **MaxText (`maxtext`)**
  - [ ] Create `src/ml_framework_snapshots/frameworks/maxtext.py`.
  - [ ] Implement `extract()` to scrape or load MaxText layer/model configurations.
  - [ ] Hook into `generate_all_snapshots.py`.
  - [ ] Generate `maxtext_v*.json`.
- [x] **DeepSpeed (`deepspeed`)** *(Extractor exists but not generated)*
  - [x] Add `deepspeed` to the `frameworks` list in `generate_all_snapshots.py`.
  - [x] Generate `deepspeed_v*.json`.
- [x] **Optax Shim (`optax_shim`)** *(Extractor exists but not generated)*
  - [x] Add `optax_shim` to the `frameworks` list in `generate_all_snapshots.py`.
  - [x] Generate `optax_shim_v*.json`.
- [x] **Orbax Checkpoint (`orbax_checkpoint`)** *(Extractor exists but not generated)*
  - [x] Add `orbax_checkpoint` to the `frameworks` list in `generate_all_snapshots.py`.
  - [x] Generate `orbax_checkpoint_v*.json`.
- [x] **HuggingFace (`huggingface`)** *(Extractor exists but not generated)*
  - [x] Add `huggingface` to the `frameworks` list in `generate_all_snapshots.py`.
  - [x] Generate `huggingface_v*.json`.
- [x] **Triton (`triton`)** *(Extractor exists but not generated)*
  - [x] Add `triton` to the `frameworks` list in `generate_all_snapshots.py`.
  - [x] Generate `triton_v*.json`.
- [x] **Scikit-Learn (`sklearn`)** *(Extractor exists but not generated)*
  - [x] Add `sklearn` to the `frameworks` list in `generate_all_snapshots.py`.
  - [x] Generate `sklearn_v*.json`.
- [x] **ONNX Runtime (`onnxruntime`)** *(Extractor exists but not generated)*
  - [x] Add `onnxruntime` to the `frameworks` list in `generate_all_snapshots.py`.
  - [x] Generate `onnxruntime_v*.json`.

## High Complexity (Custom DSLs and Compiled Backends)

These adapters represent structural representations, declarative languages, or compiled backends. Extracting them requires more complex parsing (e.g., regex, AST analysis, or TableGen parsing).

- [ ] **MLIR (`mlir`)**
  - [ ] Investigate how to extract MLIR dialects (e.g., `linalg`, `arith`, `tensor`).
  - [ ] Create `src/ml_framework_snapshots/frameworks/mlir.py` or a dedicated shell script.
  - [ ] Generate `mlir_v*.json`.
- [ ] **HTML DSL (`html_dsl`)**
  - [ ] Determine API surface (tags, properties).
  - [ ] Create extractor and generate `html_dsl_v*.json`.
- [ ] **LaTeX DSL (`latex_dsl`)**
  - [ ] Determine API surface (math environments, formatting macros).
  - [ ] Create extractor and generate `latex_dsl_v*.json`.
- [ ] **TikZ (`tikz`)**
  - [ ] Determine API surface (drawing commands, node shapes).
  - [ ] Create extractor and generate `tikz_v*.json`.
- [ ] **SASS (`sass`)**
  - [ ] Determine API surface (mixins, properties).
  - [ ] Create extractor and generate `sass_v*.json`.
