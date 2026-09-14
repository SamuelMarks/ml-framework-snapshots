ml-framework-snapshots
======================

[![License](https://img.shields.io/badge/license-Apache--2.0%20OR%20MIT-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![CI](https://github.com/SamuelMarks/ml-framework-snapshots/actions/workflows/ci.yml/badge.svg)](https://github.com/SamuelMarks/ml-framework-snapshots/actions/workflows/ci.yml)
[![Tests](https://img.shields.io/badge/tests-100%25-brightgreen.svg)]()
[![Docs](https://img.shields.io/badge/docs-100%25-brightgreen.svg)]()

**ML Framework Snapshots** is a core component of the **ml-switcheroo** ecosystem. It is a toolset designed to statically extract and formalize API signatures, compiler intermediate representations (IR), and hardware instruction set architectures (ISAs) into stable, serializable `GhostRef` schemas (as defined in `ml_switcheroo_ir`).

By deeply introspecting libraries like PyTorch, JAX, TensorFlow, Keras, MLX, Triton, Core MLIR, StableHLO, NVIDIA SASS/PTX, and AMD RDNA without requiring heavy GPU drivers or native dependencies in downstream tools, this project acts as the foundational **"Ghost Mode"** layer for ML synthesis tools, compiler backends, and agentic anti-hallucination engines.

---

## 📖 Why Does This Exist?

Machine Learning frameworks frequently utilize heavy GPU-bound libraries, complex C/C++ extensions, and dynamic metaprogramming. If you are building tools to analyze, compile, or transpile ML code, installing every ML framework into your runtime environment is prohibitive—especially for lightweight environments like WebAssembly (WASM), CI/CD pipelines, or edge devices.

`ml-framework-snapshots` decouples **API discovery** from **API execution**. It allows you to:
1. Extract robust metadata (signatures, docstrings, type hints, overloads, return types, instruction scheduling) from installed ML frameworks and compiler ODS definitions into standard JSON snapshots.
2. Ship those lightweight JSON snapshots to zero-dependency downstream tools.
3. Ground Large Language Models (LLMs), transpilers, and IDEs against hallucinations via an in-memory Grounding SDK and a live Model Context Protocol (MCP) server.
4. Export schemas to Type Stubs (`.pyi`), Pydantic V2 models, JSON Schemas, OpenAPI, Protobuf v3, and optimized LLM prompt contexts.

---

## ✨ Core Features

- **Multi-Domain Introspection**: Natively introspects Python ML frameworks, compiled C-extensions, LLVM TableGen ODS dialects (Core MLIR, StableHLO), GPU assembly ISAs (NVIDIA SASS, NVIDIA PTX, AMD RDNA/CDNA), and declarative domain DSLs (HTML, LaTeX, TikZ).
- **Deep Static & Runtime Analysis**: Cascades through AST parsers (`cdd-python`), static typing analyzers (`griffe`), and runtime reflection (`inspect`), with fallback C-extension docstring regex parsers and AST keyword access analysis (`KwargAccessVisitor`).
- **Subprocess Isolation Engine**: Isolates framework extractions in clean child subprocesses, avoiding CUDA/Metal driver initializations, C++ symbol collisions, and memory leaks.
- **Anti-Hallucination Grounding SDK**: In-memory `GroundingEngine` with Levenshtein fuzzy distance matching, cross-framework argument translation (e.g. PyTorch `dim` vs. NumPy `axis`), and SSA operand/attribute verification.
- **Model Context Protocol (MCP) Server**: Full JSON-RPC 2.0 tool server exposing real-time signature lookups, keyword hallucination validation, anti-pattern diagnostics, and assembly checking directly to AI agents.
- **High-Throughput SQLite FTS5 Index**: Ephemeral on-demand full-text search index for sub-millisecond symbol lookups without memory-heavy JSON loading.
- **Format-Agnostic Exports**: Convert snapshots into OpenAPI specifications, JSON Schema, Pydantic V2 models, Protobuf (`.proto`) messages/services, and scoped LLM prompt context templates with built-in hallucination guards.
- **Structural Diffing & Semantic Changelogs**: Compare snapshots to generate Markdown changelogs, classifying changes into breaking vs. non-breaking updates across signatures, modifiers, and microarchitecture capabilities.
- **Python Type Stub Generation**: Reconstruct `.pyi` type stubs so that IDEs and language servers provide accurate autocompletion without needing frameworks installed.
- **Static Compliance Verification**: Benchmark candidate framework shims or transpiled modules against canonical reference snapshots, scoring API parity and reporting mismatches.

---

## 📦 Supported Frameworks & Hardware ISAs

| Framework / Target | Link / Origin | Description |
|:---|:---|:---|
| **AMD RDNA** | *Built-in* | AMD RDNA1-4 / CDNA GFX assembly instruction set and VOPD dual-issue profiles |
| **CuPy** | [cupy.dev](https://cupy.dev/) | NumPy/SciPy-compatible array library for GPU-accelerated computing |
| **Dask** | [dask.org](https://dask.org/) | Library for parallel computing in Python |
| **DeepSpeed** | [deepspeed.ai](https://www.deepspeed.ai/) | Extreme-scale deep learning optimization library |
| **Flax (NNX)** | [flax.readthedocs.io](https://flax.readthedocs.io/) | Neural network library and ecosystem for JAX |
| **HTML DSL** | *Built-in* | Declarative HTML tags and attribute snapshot schema |
| **HuggingFace** | [huggingface.co](https://huggingface.co/) | Transformers, Diffusers, and Tokenizers API signatures |
| **JAX** | [jax.readthedocs.io](https://jax.readthedocs.io/) | Composable transformations of Python+NumPy programs |
| **Keras** | [keras.io](https://keras.io/) | Multi-backend deep learning API |
| **LaTeX DSL** | *Built-in* | Standard LaTeX mathematical environments and formatting macros |
| **MaxText** | [github.com/google/maxtext](https://github.com/google/maxtext) | Performant and highly scalable JAX LLM implementation |
| **MLIR** | [mlir.llvm.org](https://mlir.llvm.org/) | Core MLIR dialects (`arith`, `math`, `tensor`, `linalg`, `scf`, `gpu`, etc.) via TableGen |
| **MLX** | [ml-explore.github.io/mlx](https://ml-explore.github.io/mlx/) | Array framework optimized for Apple Silicon |
| **NumPy** | [numpy.org](https://numpy.org/) | The fundamental package for scientific computing with Python |
| **NVIDIA PTX** | *Built-in* | NVIDIA Parallel Thread Execution (PTX) ISA, state spaces, and type qualifiers |
| **NVIDIA SASS** | *Built-in* | NVIDIA SASS GPU assembly instruction set and cycle-accurate control codes |
| **ONNXRuntime** | [onnxruntime.ai](https://onnxruntime.ai/) | Cross-platform, high-performance ML inferencing accelerator |
| **Optax** | [optax.readthedocs.io](https://optax.readthedocs.io/) | Gradient processing and optimization library for JAX |
| **Orbax** | [orbax.readthedocs.io](https://orbax.readthedocs.io/) | Checkpointing and persistence library for JAX |
| **Pax** | [github.com/google/paxml](https://github.com/google/paxml) | JAX-based high-performance machine learning framework |
| **PyTorch** | [pytorch.org](https://pytorch.org/) | Dynamic neural networks, ATen operators, and native C-extensions |
| **Scikit-Learn** | [scikit-learn.org](https://scikit-learn.org/) | Classical machine learning algorithms in Python |
| **StableHLO** | [github.com/openxla/stablehlo](https://github.com/openxla/stablehlo) | Backward-compatible ML compiler operations, attributes, and regions |
| **TensorFlow** | [tensorflow.org](https://www.tensorflow.org/) | End-to-end open source platform for machine learning |
| **TikZ DSL** | *Built-in* | Declarative diagramming and vector geometry DSL schema |
| **Triton** | [triton-lang.org](https://triton-lang.org/) | Python-like programming language for high-throughput GPU kernels |

---

## 🚀 Installation

Requires **Python >= 3.10**.

```bash
pip install ml-framework-snapshots
```

### Installation Extras

```bash
# Install with heavy framework dependencies for live extraction
pip install "ml-framework-snapshots[frameworks]"

# Install with development and code generation tools
pip install "ml-framework-snapshots[generate]"

# Install with full test dependencies
pip install "ml-framework-snapshots[test]"
```

*(Note: Running the CLI to diff, export, ground, or check existing pre-bundled JSON snapshots requires **zero** heavy ML dependencies.)*

---

## 💻 CLI Usage

The tool provides an extensive command line interface via `ml_framework_snapshots`:

### 1. Capture Snapshots

Extract API structures from your local environment and save them as JSON:

```bash
ml_framework_snapshots capture torch jax keras --out-dir ./snapshots
```
*Flags:*
- `--isolated`: Runs each extraction in an isolated child subprocess to prevent driver crashes.
- `--include-nonpublic`: Captures internal and private APIs (`_` prefix).

### 2. Check Implementation Compliance

Test a local module's API compliance against a canonical reference snapshot:

```bash
ml_framework_snapshots check ./snapshots/torch_v2.4.0.json ./my_project/src/my_torch_shim
    --reference-prefix torch --target-prefix my_project.my_torch_shim
```

### 3. Diff & Semantic Changelogs

Compare two snapshots to detect additions, deletions, and breaking/non-breaking signature shifts:

```bash
ml_framework_snapshots diff ./snapshots/jax_v0.4.30.json ./snapshots/jax_v0.4.31.json --changelog
```

### 4. Generate Python Type Stubs (`.pyi`)

Export snapshots back into PEP-484 `.pyi` type stubs with sanitized signatures and overload definitions:

```bash
ml_framework_snapshots generate-stubs --input ./snapshots/torch_v2.4.0.json --out-dir ./stubs/
```

### 5. Multi-Target Schema & Prompt Export

Export framework definitions to standard schemas or compact LLM prompt context:

```bash
# Export to Pydantic V2 models
ml_framework_snapshots export --input ./snapshots/torch_v2.4.0.json --format pydantic --out-dir ./models/

# Export to OpenAPI 3.0 specification
ml_framework_snapshots export --input ./snapshots/torch_v2.4.0.json --format openapi --out-dir ./openapi/

# Export to Protobuf v3 message and gRPC definitions
ml_framework_snapshots export --input ./snapshots/torch_v2.4.0.json --format protobuf --out-dir ./proto/

# Export compact, typed LLM prompt context with hallucination guards
ml_framework_snapshots export --input ./snapshots/torch_v2.4.0.json --format prompt --out-dir ./prompts/
```

### 6. Hardware ISA & Compiler Dialect Verification

Directly validate assembly snippets and compiler IR from the terminal:

```bash
# Validate NVIDIA SASS instruction against Hopper architecture
ml_framework_snapshots check-sass WGMMA --sm-arch sm_90 --modifiers .F16

# Validate AMD RDNA instruction against RDNA3 architecture
ml_framework_snapshots check-rdna v_dual_fmac_f32 --gfx-arch GFX11/RDNA3

# Validate StableHLO operation attributes and operand counts
ml_framework_snapshots check-stablehlo stablehlo.dot_general --operands-count 2 --attributes dot_dimension_numbers
```

### 7. Snapshot Management & Local Cache

```bash
# List all pre-bundled and locally cached snapshots
ml_framework_snapshots list-snapshots

# Download official pre-compiled snapshot releases into cache
ml_framework_snapshots pull torch 2.4.0
ml_framework_snapshots download-all

# Cryptographically verify the integrity of cached snapshot files
ml_framework_snapshots verify-local-cache

# Rebuild or query the local SQLite FTS5 search index
ml_framework_snapshots index --rebuild
```

---

## 🤖 Model Context Protocol (MCP) Server

`ml-framework-snapshots` includes a built-in Model Context Protocol (MCP) server that connects LLM coding assistants and transpiler agents to ground-truth framework signatures in real time via JSON-RPC 2.0.

### Starting the Server

```bash
ml_framework_snapshots mcp
```

### Available MCP Tools

| Tool Name | Description |
|:---|:---|
| `get_api_signature` | Look up exact parameter types, kinds, defaults, and return specifications. |
| `search_apis` | Search available framework operations and mnemonics by keyword or prefix. |
| `check_hallucination` | Verify whether an API path or keyword arguments represent LLM hallucinations. |
| `explain_anti_pattern` | Provide canonical migration advice for common cross-framework anti-patterns. |
| `check_sass_instruction` | Validate NVIDIA SASS assembly against target compute capabilities (`sm_70`–`sm_100`). |
| `check_rdna_instruction` | Validate AMD RDNA/CDNA assembly against GFX generations (`GFX9`–`GFX12`). |
| `check_ptx_instruction` | Validate NVIDIA PTX instructions, type qualifiers, state spaces, and scopes. |
| `check_mlir_op` | Verify MLIR dialect operations against TableGen ODS traits and type constraints. |
| `check_stablehlo_op` | Verify StableHLO operation attributes, dimensions, and region signatures. |
| `check_code_block` | Batch verify an entire code block for nonexistent APIs and invalid kwargs. |
| `translate_concept_arguments` | Translate arguments across frameworks (e.g. PyTorch `dim` to NumPy `axis`). |

### Client Configuration Examples

#### Cursor (`.cursor/mcp.json`)
```json
{
  "mcpServers": {
    "ml-framework-snapshots": {
      "command": "python",
      "args": ["-m", "ml_framework_snapshots.mcp_server"]
    }
  }
}
```

#### Claude Desktop (`claude_desktop_config.json`)
```json
{
  "mcpServers": {
    "ml-framework-snapshots": {
      "command": "python",
      "args": ["-m", "ml_framework_snapshots.mcp_server"]
    }
  }
}
```

#### Gemini CLI
```json
{
  "mcpServers": {
    "ml-framework-snapshots": {
      "command": "python",
      "args": ["-m", "ml_framework_snapshots.mcp_server"]
    }
  }
}
```

---

## 🛠️ Python SDK Usage

### 1. Snapshot Extraction & Diffing

```python
from ml_framework_snapshots.api import extract_snapshot, write_snapshot
from ml_framework_snapshots.diff import diff_snapshots, generate_changelog

# Extract snapshot for PyTorch (uses isolated subprocess by default)
snapshot = extract_snapshot("torch")

if snapshot:
    print(f"Captured Torch v{snapshot['version']}")
    write_snapshot("torch", snapshot, output_dir="./snapshots")

# Diff two snapshot dictionaries
# result = diff_snapshots(snap_v1, snap_v2)
# print(generate_changelog(result))
```

### 2. Anti-Hallucination Grounding SDK

```python
from ml_framework_snapshots.grounding import (
    GroundingEngine,
    validate_python_call,
    validate_sass_instruction,
    validate_stablehlo_op,
)

engine = GroundingEngine()

# 1. Verify a Python API call and catch hallucinated kwargs
report = validate_python_call(
    framework="torch",
    api_path="torch.sum",
    args=[],
    kwargs={"input": None, "axis": 0},  # Hallucination: 'axis' instead of 'dim'
    engine=engine,
)

if not report.is_grounded:
    for diag in report.diagnostics:
        print(f"[{diag.severity}] {diag.message} -> Suggested: {diag.suggested_fix}")

# 2. Verify GPU assembly against target architecture
sass_report = validate_sass_instruction(
    mnemonic="WGMMA",
    architecture="sm_80",  # WGMMA is sm_90+ only
    operands=["R0", "R1"],
    engine=engine,
)
print("SASS Grounded:", sass_report.is_grounded)

# 3. Verify StableHLO compiler operations
hlo_report = validate_stablehlo_op(
    op_name="stablehlo.dot_general",
    operand_types=["tensor<128x64xf32>", "tensor<64x256xf32>"],
    attributes={},  # Missing required dot_dimension_numbers
    engine=engine,
)
print("StableHLO Grounded:", hlo_report.is_grounded)
```

---

## 🛡️ The Ecosystem & Preventing LLM Hallucinations

`ml-framework-snapshots` serves as the ground-truth contract for a broader suite of cross-framework translation and compilation tools:

- **[ml-switcheroo](https://github.com/SamuelMarks/ml-switcheroo)**: Universal compiler and transpiler solving the $O(N^2)$ ML interoperability problem by translating dialects (PyTorch, JAX, TensorFlow) through a canonical intermediate representation.
- **[ml-switcheroo-compiler](https://github.com/SamuelMarks/ml-switcheroo-compiler)**: Core execution backend lowering Unified IR into WebGPU and WASM SIMD binaries for zero-dependency execution.
- **[zero-zoo](https://github.com/SamuelMarks/zero-zoo)**: Verification matrix ensuring that lightweight API shells (like `zero-pytorch`) produce float-for-float identical results compared to native frameworks.

---

## 🤝 Contribution & Development

We welcome contributions.

```bash
# Setup environment
python -m venv .venv
source .venv/bin/activate
pip install -r test-requirements.txt

# Run test suite with 100% coverage enforcement
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
dual licensed as above, without any additional terms or conditions.
