ml-framework-snapshots
======================

[![License](https://img.shields.io/badge/license-Apache--2.0%20OR%20MIT-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![CI](https://github.com/SamuelMarks/ml-frameworks-snapshots/actions/workflows/ci.yml/badge.svg)](https://github.com/SamuelMarks/ml-frameworks-snapshots/actions)
[![Tests](https://img.shields.io/badge/tests-unknown%25-lightgrey.svg)]()
[![Docs](https://img.shields.io/badge/docs-100%25-brightgreen.svg)]()

**ML Framework Snapshots** is a toolset designed to statically extract and formalize API signatures from major machine learning frameworks into stable, serializable schemas.

By deeply introspecting libraries like PyTorch, JAX, TensorFlow, Keras, MLX, and Flax without requiring them to be imported natively into your final application, this project acts as the foundational "Ghost Mode" layer for ML synthesis tools, API emulation layers, and cross-framework translation compilers.

---

## 📖 Why Does This Exist?

Machine Learning frameworks frequently utilize heavy GPU-bound libraries, complex C/C++ extensions, and dynamic metaprogramming. If you are building tools to analyze, compile, or transpile ML code, installing every ML framework into your runtime environment is prohibitive—especially for lightweight environments like WebAssembly (WASM), CI/CD pipelines, or edge devices.

`ml-framework-snapshots` decouples **API discovery** from **API execution**. It allows you to:
1. Extract robust metadata (signatures, docstrings, type hints, overloads, return types) from installed ML frameworks into standard JSON snapshots.
2. Ship those lightweight JSON snapshots to your lightweight downstream tools.
3. Use those snapshots to generate Type Stubs (`.pyi`), Pydantic models, JSON Schemas, or OpenAPI definitions, and perform structural compliance checking or diffs across framework versions.

## ✨ Core Features

- **Multi-Framework Introspection**: Natively supports PyTorch, JAX, TensorFlow, Keras, MLX, Flax, Scikit-Learn, HuggingFace (Transformers, Diffusers, Tokenizers), Triton, ONNXRuntime, and DeepSpeed.
- **Deep Static & Runtime Analysis**: Achieves maximum fidelity by cascading through AST parsers (`cdd-python`), static typing analyzers (`griffe`), and standard runtime reflection (`inspect`), before falling back to custom C-Extension docstring parsers.
- **Rich Context Extraction**: Beyond standard arguments, it captures docstrings, parameter descriptions, function overloads, `raises` exceptions, return types, and environment execution tags (e.g., CUDA vs. CPU).
- **Format Agnostic Exports**: Instantly convert captured API snapshots into OpenAPI specifications, JSON Schema, Pydantic V2 models, and Protobuf (`.proto`) definitions.
- **Structural Diffing & Semantic Versioning**: Compare two API snapshots to generate markdown changelogs, detecting not just added/removed functions but highlighting potentially breaking signature changes.
- **Python Stub Generation**: Export snapshots back into python via `.pyi` type stubs so that IDEs and language servers can understand the API without the framework installed.
- **Compliance Checking**: Automatically test a new API implementation (like a transpiled module or a wrapper wrapper) against a canonical snapshot to measure coverage and highlight signature mismatches.

---

## 🚀 Installation

Requires Python >= 3.9.

```bash
pip install ml-framework-snapshots
```

If you intend to generate new snapshots from your environment, you must install the frameworks you wish to snapshot:

```bash
# E.g., to snapshot PyTorch and JAX:
pip install ml-framework-snapshots[torch,jax]
```

*(Note: If you only want to use the CLI to diff, export, or check existing JSON snapshots, you do not need to install the heavy ML framework dependencies.)*

---

## 💻 CLI Usage

The tool operates primarily via the `ml_framework_snapshots` command line interface.

### 1. Capture Snapshots

Extract API structures from the current environment and save them as JSON. You can specify exact frameworks or use `"all"`.

```bash
ml_framework_snapshots capture torch jax keras --out-dir ./snapshots
```
*Use `--include-nonpublic` if you want to include internal/private APIs (methods starting with `_`).*

### 2. Check Compliance

Test a target module's API compliance against a reference snapshot. Excellent for verifying custom shims or wrappers.

```bash
ml_framework_snapshots check ./snapshots/torch_v2.0.0.json ./my_project/src/my_torch_shim --reference-prefix torch --target-prefix my_project.my_torch_shim
```
Outputs a percentage score, missing APIs, and a list of mismatched signatures.

### 3. Diff & Changelogs

Find API drift between two versions of the same framework.

```bash
ml_framework_snapshots diff ./snapshots/jax_v0.4.0.json ./snapshots/jax_v0.4.1.json --changelog
```

### 4. Generate Type Stubs

Generate standard Python `.pyi` stub files that can be distributed to enable auto-completion without full installations.

```bash
ml_framework_snapshots generate-stubs --input ./snapshots/torch_v2.0.0.json --out-dir ./stubs/
```

### 5. Export Definitions

Export the framework definitions to standard schemas.

```bash
# Export all Torch API definitions to Pydantic models
ml_framework_snapshots export --input ./snapshots/torch_v2.0.0.json --format pydantic --out-dir ./pydantic_models/

# Export to OpenAPI schema
ml_framework_snapshots export --input ./snapshots/torch_v2.0.0.json --format openapi --out-dir ./openapi/
```

---

## 🛠️ SDK Usage

You can also integrate the snapshot engine programmatically into your own python applications.

```python
from ml_framework_snapshots.api import extract_snapshot, write_snapshot
from ml_framework_snapshots.diff import diff_snapshots, generate_changelog

# Extract snapshot for PyTorch (if available in the local env)
snapshot = extract_snapshot("torch", include_nonpublic=False)

if snapshot:
    print(f"Captured Torch v{snapshot['version']}")

    # Save the snapshot to JSON
    write_snapshot("torch", snapshot, output_dir="./snapshots")

# Diffing programmatically
# diff_result = diff_snapshots(snap_old, snap_new)
# print(generate_changelog(diff_result))
```

---

## 🤝 Contribution

We welcome contributions.

**Development Setup**:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r test-requirements.txt
```

Run tests with coverage:
```bash
pytest --cov=src/ml_framework_snapshots --cov-branch
```

---

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <https://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <https://opensource.org/licenses/MIT>)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions
