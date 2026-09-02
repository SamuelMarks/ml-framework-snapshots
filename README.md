ml-framework-snapshots
======================

[![License](https://img.shields.io/badge/license-Apache--2.0%20OR%20MIT-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![CI](https://github.com/SamuelMarks/ml-framework-snapshots/actions/workflows/ci.yml/badge.svg)](https://github.com/SamuelMarks/ml-framework-snapshots/actions/workflows/ci.yml)
[![Tests](https://img.shields.io/badge/tests-100%25-brightgreen.svg)]()
[![Docs](https://img.shields.io/badge/docs-100%25-brightgreen.svg)]()

**ML Framework Snapshots** is a core component of the **ml-switcheroo** ecosystem. It is a toolset designed to statically extract and formalize API signatures from major machine learning frameworks into stable, serializable `GhostRef` schemas (as defined in `ml_switcheroo_ir`).

By deeply introspecting libraries like PyTorch, JAX, TensorFlow, Keras, MLX, and Flax without requiring them to be imported natively into your final application, this project acts as the foundational "Ghost Mode" layer for ML synthesis tools, API emulation layers, and cross-framework translation compilers.

---

## 📖 Why Does This Exist?

Machine Learning frameworks frequently utilize heavy GPU-bound libraries, complex C/C++ extensions, and dynamic metaprogramming. If you are building tools to analyze, compile, or transpile ML code, installing every ML framework into your runtime environment is prohibitive—especially for lightweight environments like WebAssembly (WASM), CI/CD pipelines, or edge devices.

`ml-framework-snapshots` decouples **API discovery** from **API execution**. It allows you to:
1. Extract robust metadata (signatures, docstrings, type hints, overloads, return types) from installed ML frameworks into standard JSON snapshots.
2. Ship those lightweight JSON snapshots to your lightweight downstream tools.
3. Use those snapshots to generate Type Stubs (`.pyi`), Pydantic models, JSON Schemas, or OpenAPI definitions, and perform structural compliance checking or diffs across framework versions.

## ✨ Core Features

- **Multi-Framework Introspection**: Natively supports a wide array of machine learning libraries and toolkits (see [Supported Frameworks](#-supported-frameworks) below), including non-Python domains like NVIDIA SASS and AMD RDNA instruction sets via static JSON extraction.
- **Deep Static & Runtime Analysis**: Achieves maximum fidelity by cascading through AST parsers (`cdd-python`), static typing analyzers (`griffe`), and standard runtime reflection (`inspect`), before falling back to custom C-Extension docstring parsers.
- **Rich Context Extraction**: Beyond standard arguments, it captures docstrings, parameter descriptions, function overloads, `raises` exceptions, return types, and environment execution tags (e.g., CUDA vs. CPU).
- **Format Agnostic Exports**: Instantly convert captured API snapshots into OpenAPI specifications, JSON Schema, Pydantic V2 models, and Protobuf (`.proto`) definitions.
- **Structural Diffing & Semantic Versioning**: Compare two API snapshots to generate markdown changelogs, detecting not just added/removed functions but highlighting potentially breaking signature changes.
- **Python Stub Generation**: Export snapshots back into python via `.pyi` type stubs so that IDEs and language servers can understand the API without the framework installed.
- **Compliance Checking**: Automatically test a new API implementation (like a transpiled module or a custom wrapper) against a canonical snapshot to measure coverage and highlight signature mismatches.

---

## 📦 Supported Frameworks

| Framework | Link | Description |
|-----------|------|-------------|
| **CuPy** | [cupy.dev](https://cupy.dev/) | NumPy/SciPy-compatible Array Library for GPU-accelerated computing |
| **Dask** | [dask.org](https://dask.org/) | Library for parallel computing in Python |
| **DeepSpeed** | [deepspeed.ai](https://www.deepspeed.ai/) | Deep learning optimization library |
| **Flax (NNX)** | [flax.readthedocs.io](https://flax.readthedocs.io/) | Neural network library and ecosystem for JAX |
| **HTML DSL** | *Built-in* | HTML DSL API Snapshot Extractor |
| **HuggingFace** | [huggingface.co](https://huggingface.co/) | Tools and models for NLP and more |
| **JAX** | [jax.readthedocs.io](https://jax.readthedocs.io/) | Composable transformations of Python+NumPy programs |
| **Keras** | [keras.io](https://keras.io/) | Deep learning API |
| **LaTeX DSL** | *Built-in* | LaTeX DSL API Snapshot Extractor |
| **MaxText** | [github.com/google/maxtext](https://github.com/google/maxtext) | A simple, performant and highly scalable Jax LLM |
| **MLIR** | [mlir.llvm.org](https://mlir.llvm.org/) | Multi-Level Intermediate Representation |
| **MLX** | [ml-explore.github.io/mlx](https://ml-explore.github.io/mlx/) | An array framework for Apple silicon |
| **NumPy** | [numpy.org](https://numpy.org/) | The fundamental package for scientific computing with Python |
| **NVIDIA SASS** | *Built-in* | NVIDIA SASS Assembly Snapshot Extractor |
| **AMD RDNA** | *Built-in* | AMD RDNA Assembly Snapshot Extractor |
| **ONNXRuntime** | [onnxruntime.ai](https://onnxruntime.ai/) | Cross-platform, high performance ML inferencing and training accelerator |
| **Optax** | [optax.readthedocs.io](https://optax.readthedocs.io/) | Gradient processing and optimization library for JAX |
| **Orbax** | [orbax.readthedocs.io](https://orbax.readthedocs.io/) | Checkpointing library for JAX |
| **Pax** | [github.com/google/paxml](https://github.com/google/paxml) | Jax-based machine learning framework |
| **PyTorch** | [pytorch.org](https://pytorch.org/) | Tensors and Dynamic neural networks in Python |
| **Scikit-Learn** | [scikit-learn.org](https://scikit-learn.org/) | Machine learning in Python |
| **TensorFlow** | [tensorflow.org](https://www.tensorflow.org/) | An end-to-end open source machine learning platform |
| **TikZ DSL** | *Built-in* | TikZ DSL API Snapshot Extractor |
| **Triton** | [triton-lang.org](https://triton-lang.org/) | An open-source Python-like programming language |

---

## 🚀 Installation

Requires Python >= 3.9.

```bash
pip install ml-framework-snapshots
```

If you intend to generate new snapshots from your environment, you must install the target frameworks (or install the meta-package that brings them in):

```bash
# Install the library along with the heavy framework dependencies
pip install "ml-framework-snapshots[frameworks]"
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

## 🛡️ The Ecosystem & Preventing LLM Hallucinations

`ml-framework-snapshots` is the foundational API layer for a broader ecosystem of cross-framework translation and compilation tools. By providing deterministic, versioned JSON schemas of ML APIs, it serves as the **ground-truth source** that prevents AI-driven transpilers and compilers from hallucinating incorrect arguments, shapes, or structural hierarchies.

- **[ml-switcheroo](https://github.com/SamuelMarks/ml-switcheroo)**: A universal compiler and transpiler that solves the $O(N^2)$ interoperability problem. It maps major dialects (PyTorch, JAX, TensorFlow) to a central "Hub" abstract standard. To generate accurate transpiled code, `ml-switcheroo` relies on these JSON snapshots to validate function signatures, ensuring the output is semantically exact rather than just structurally plausible.
- **[ml-switcheroo-compiler](https://github.com/SamuelMarks/ml-switcheroo-compiler)**: The core execution engine that enforces a "No Math in Frontends" rule. It lowers Unified IR directly into highly optimized WebGPU or WASM SIMD executables for in-browser execution. The compiler uses these snapshots to statically resolve API routing without needing heavy framework dependencies.
- **[zero-zoo](https://github.com/SamuelMarks/zero-zoo) (and the `zero-*` wrappers)**: The central proving grounds for the ecosystem. It maintains a zoo of canonical model implementations across all dialects and utilizes matrix testing to ensure that the lightweight API shells (like `zero-pytorch`) produce float-for-float identical results compared to native frameworks.

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
