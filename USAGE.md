# Usage Guide & Examples

`ml-framework-snapshots` provides a comprehensive command-line interface, a programmatic Grounding SDK, and a Model Context Protocol (MCP) server for capturing, diffing, validating, and exporting framework signatures, compiler IRs, and hardware ISAs.

---

## Table of Contents

1. [Capturing Framework Snapshots](#1-capturing-framework-snapshots)
2. [Offline Snapshots & Cache Management](#2-offline-snapshots--cache-management)
3. [Compliance Checking](#3-compliance-checking)
4. [Structural Diffing & Semantic Changelogs](#4-structural-diffing--semantic-changelogs)
5. [Code & Schema Export](#5-code--schema-export)
6. [Hardware ISA & Compiler IR CLI Verification](#6-hardware-isa--compiler-ir-cli-verification)
7. [Anti-Hallucination Grounding SDK (Python)](#7-anti-hallucination-grounding-sdk-python)
8. [Model Context Protocol (MCP) Server](#8-model-context-protocol-mcp-server)

---

## 1. Capturing Framework Snapshots

The `capture` subcommand inspects installed ML frameworks in your environment, serializing their symbols into deterministic JSON snapshots under the Ghost Protocol.

### Capture Specific Frameworks

```bash
# Capture PyTorch, JAX, and Keras into ./snapshots
ml_framework_snapshots capture torch jax keras --out-dir ./snapshots
```

### Capture All Installed Frameworks

```bash
ml_framework_snapshots capture all --out-dir ./snapshots
```

### Include Non-Public APIs

By default, private methods and functions (prefixed with `_`) are excluded. Use `--include-nonpublic` to extract internal APIs:

```bash
ml_framework_snapshots capture torch --include-nonpublic --out-dir ./snapshots
```

### Capturing Hardware ISAs & Compiler Dialects

The tool captures pre-compiled hardware ISAs and compiler specifications without requiring GPU drivers or LLVM toolchains:

```bash
# Capture NVIDIA SASS assembly catalog
ml_framework_snapshots capture nvidia_sass --out-dir ./snapshots

# Capture AMD RDNA / CDNA assembly catalog
ml_framework_snapshots capture amd_rdna --out-dir ./snapshots

# Capture NVIDIA PTX assembly catalog
ml_framework_snapshots capture nvidia_ptx --out-dir ./snapshots

# Capture StableHLO compiler dialect
ml_framework_snapshots capture stablehlo --out-dir ./snapshots

# Capture Core MLIR dialects
ml_framework_snapshots capture mlir --out-dir ./snapshots
```

### Regenerating Exhaustive Target Catalogs

To update the exhaustive JSON dumps (e.g., when a new CUDA Toolkit, LLVM release, or StableHLO specification is published), execute the dedicated scraper tools:

```bash
# Rebuild NVIDIA SASS catalog from CUDA Toolkit tables
python -m ml_framework_snapshots.tools.scrape_nvidia_sass

# Rebuild AMD RDNA / CDNA catalog from LLVM TableGen sources
python -m ml_framework_snapshots.tools.scrape_amd_rdna

# Rebuild NVIDIA PTX catalog
python -m ml_framework_snapshots.tools.scrape_nvidia_ptx

# Rebuild Core MLIR dialect catalog
python -m ml_framework_snapshots.tools.scrape_mlir

# Rebuild StableHLO dialect catalog
python -m ml_framework_snapshots.tools.build_stablehlo_snapshot
```

---

## 2. Offline Snapshots & Cache Management

`ml-framework-snapshots` resolves snapshot references across a multi-tier search cascade:
1. Exact file paths (`./snapshots/torch_v2.4.0.json`)
2. Custom paths defined in `$ML_SNAPSHOTS_PATH` or `$ML_FRAMEWORK_SNAPSHOTS_PATH`
3. Local user cache directory (`~/.cache/ml_framework_snapshots/snapshots/`)
4. Project root `snapshots/`
5. Package-bundled offline snapshots (`src/ml_framework_snapshots/snapshots/` and `frameworks/`)

### Listing Available Snapshots

List all bundled, cached, and local snapshots available offline:

```bash
ml_framework_snapshots list-snapshots
```

### Downloading Pre-Compiled Snapshots

Fetch official pre-compiled snapshot releases from GitHub into your local cache:

```bash
# Pull a specific framework and version
ml_framework_snapshots pull torch@2.4.0

# Pull JAX snapshot
ml_framework_snapshots pull jax@0.4.30

# Download all canonical pre-compiled snapshot assets
ml_framework_snapshots download-all
```

### Cryptographic Cache Integrity Verification

Verify the SHA-256 checksums of all cached snapshot files:

```bash
ml_framework_snapshots verify-local-cache
```

### Managing the Ephemeral SQLite FTS5 Index

The ephemeral SQLite FTS5 search index provides sub-millisecond symbol lookups without loading multi-megabyte JSON trees into memory:

```bash
# Check index status and total indexed symbols
ml_framework_snapshots index --status

# Rebuild the search index across all available snapshots
ml_framework_snapshots index --rebuild

# Clear the index database
ml_framework_snapshots index --clean
```

---

## 3. Compliance Checking

The `check` subcommand performs cleanroom verification of candidate framework implementations (such as transpiled shims, WASM wrappers, or mock modules) against a reference snapshot.

### Checking a Specific Module

Test the compliance of a custom JAX wrapper against a reference JAX snapshot:

```bash
ml_framework_snapshots check
    jax
    ~/repos/ml-switcheroo/src/ml_switcheroo/frameworks/jax.py
    --reference-prefix jax
    --target-prefix ml_switcheroo.frameworks.jax
```

**Example Output:**
```text
Extracting target APIs from /Users/samuel/repos/ml-switcheroo/src/ml_switcheroo/frameworks/jax.py...
Scoring compliance...

--- Compliance Report ---
Overall Compliance: 82.5%

Breakdown by Module:
  - jax.numpy: 86.5% (45/52)

Missing APIs (2):

|   | Framework | Namespace | Symbol | FQN | Signature | Docstring |
|---|---|---|---|---|---|---|
| [ ] | jax | jax.numpy | fft | jax.numpy.fft | module | Fast Fourier Transform module. |
| [ ] | jax | jax.numpy | linalg | jax.numpy.linalg | module | Linear algebra module. |
```

### Checking an Entire Package Tree

Test an independent project like `onnx9000-jax`:

```bash
ml_framework_snapshots check
    jax
    ~/repos/ml-switcheroo/onnx9000/packages/python/onnx9000-jax
    --reference-prefix jax
    --target-prefix onnx9000.jax
```

---

## 4. Structural Diffing & Semantic Changelogs

The `diff` subcommand compares two snapshots, identifying additions, removals, parameter shifts, and backwards-incompatible changes.

### Generating a Terminal Summary

```bash
ml_framework_snapshots diff ./snapshots/torch_v2.3.0.json ./snapshots/torch_v2.4.0.json
```

**Output:**
```text
ADDED: 14
  + torch.nn.functional.scaled_dot_product_attention
  + torch.compiler.is_compiling
REMOVED: 2
  - torch.legacy_op
SIGNATURE CHANGED: 5
  * torch.sum
```

### Generating a Markdown Changelog

Use `--changelog` to emit release notes with breaking vs. non-breaking classification:

```bash
ml_framework_snapshots diff ./snapshots/jax_v0.4.30.json ./snapshots/jax_v0.4.31.json --changelog
```

**Example Output:**
```markdown
# API Changelog

## Added (2)
- `jax.numpy.trapezoid`
- `jax.experimental.shard_map`

## Removed (0)

## Breaking Signature Changes (1)
- `jax.numpy.pad`: Added required positional parameter `pad_width` without default.

## Non-Breaking Signature Changes (3)
- `jax.numpy.mean`: Added optional keyword argument `where`.
```

---

## 5. Code & Schema Export

Transform framework snapshots into diverse interface formats and schemas.

### Generate Python Type Stubs (`.pyi`)

Create PEP-484 type stub files with reconstructed module trees and overloads:

```bash
ml_framework_snapshots generate-stubs --input torch --out-dir ./stubs/
```

### Export to Pydantic V2 Models

Generate type-safe Pydantic models with `Field(description=...)` and overloaded union variants:

```bash
ml_framework_snapshots export --input torch --format pydantic --out-dir ./models/
```

### Export to OpenAPI 3.0 Specifications

Generate OpenAPI REST interface routes:

```bash
ml_framework_snapshots export --input torch --format openapi --out-dir ./openapi/
```

### Export to JSON Schema

Generate JSON Schema validation specifications:

```bash
ml_framework_snapshots export --input jax --format json_schema --out-dir ./json_schema/
```

### Export to Protobuf v3 & gRPC

Generate `.proto` message definitions, standard enums (Reduction, Padding, Layout), and service stubs:

```bash
ml_framework_snapshots export --input torch --format protobuf --out-dir ./proto/
```

### Export LLM Prompt Contexts

Generate compact Markdown specifications with parameter constraints and built-in hallucination guards:

```bash
ml_framework_snapshots export --input torch --format llm_prompt --out-dir ./prompts/
```

---

## 6. Hardware ISA & Compiler IR CLI Verification

Directly validate machine instructions and compiler operations from the terminal.

### NVIDIA SASS Instruction Validation (`check-sass`)

Validate instruction legality, operands, modifiers, and SM architecture support:

```bash
# Verify Hopper asynchronous warpgroup matrix instruction
ml_framework_snapshots check-sass WGMMA --sm-arch sm_90 --modifiers .F16

# Verify single precision fused multiply-add
ml_framework_snapshots check-sass FFMA --operands R0,R1,R2,R3 --modifiers .FTZ --sm-arch sm_80

# Verify an entire SASS assembly file
ml_framework_snapshots check-sass --file kernel.sass --sm-arch sm_90
```

### AMD RDNA / CDNA Instruction Validation (`check-rdna`)

Validate RDNA/CDNA instructions, wave sizes, modifiers, and GFX generations:

```bash
# Verify RDNA3 dual-issue instruction
ml_framework_snapshots check-rdna v_dual_fmac_f32 --gfx-arch GFX11/RDNA3

# Verify vector addition with operand modifiers
ml_framework_snapshots check-rdna v_add_f32 --operands v0,v1,v2 --modifiers clamp --gfx-arch GFX10.3/RDNA2

# Verify an entire RDNA assembly file
ml_framework_snapshots check-rdna --file kernel.s --gfx-arch GFX11/RDNA3
```

### Core MLIR Dialect Verification (`check-mlir`)

Validate MLIR operations against TableGen traits and SSA operand counts:

```bash
# Verify arith.addf operation
ml_framework_snapshots check-mlir arith.addf --operands-count 2

# Verify an entire MLIR text file
ml_framework_snapshots check-mlir --file module.mlir
```

### StableHLO Operation Verification (`check-stablehlo`)

Validate StableHLO operations against dimension number and region constraints:

```bash
# Verify stablehlo.dot_general with mandatory dot_dimension_numbers attribute
ml_framework_snapshots check-stablehlo stablehlo.dot_general
    --operands-count 2
    --attributes dot_dimension_numbers

# Verify an entire StableHLO text file
ml_framework_snapshots check-stablehlo --file graph.mlir
```

---

## 7. Anti-Hallucination Grounding SDK (Python)

Integrate the verification engine directly into your code synthesis pipelines, transpilers, or compiler passes.

```python
from ml_framework_snapshots.grounding import (
    GroundingEngine,
    validate_python_call,
    validate_sass_instruction,
    validate_rdna_instruction,
    validate_stablehlo_op,
    validate_mlir_op,
)

# Initialize in-memory grounding engine (searches bundled and cached snapshots)
engine = GroundingEngine()

# ---------------------------------------------------------
# 1. Python ML Framework Grounding
# ---------------------------------------------------------
report = validate_python_call(
    framework="torch",
    api_path="torch.sum",
    args=[],
    kwargs={"input": None, "axis": 0},  # Hallucinated: PyTorch uses 'dim'
    engine=engine,
)

print(f"Call Grounded: {report.is_grounded}")  # False
for diag in report.diagnostics:
    print(f"  [{diag.severity}] {diag.message}")
    print(f"  -> Suggested Fix: {diag.suggested_fix}")  # 'dim'

# ---------------------------------------------------------
# 2. NVIDIA SASS Hardware Grounding
# ---------------------------------------------------------
sass_report = validate_sass_instruction(
    mnemonic="WGMMA",
    architecture="sm_80",  # WGMMA requires Hopper+ (sm_90+)
    operands=["R0", "R1", "R2", "R3"],
    engine=engine,
)

if not sass_report.is_grounded:
    for diag in sass_report.diagnostics:
        print(f"SASS Error: {diag.message}")

# ---------------------------------------------------------
# 3. AMD RDNA Hardware Grounding: catching misaligned register pair
# ---------------------------------------------------------
rdna_report = validate_rdna_instruction(
    mnemonic="s_add_u32",
    gfx_arch="GFX11",
    operands=["v[1:2]", "s0", "s1"],  # Register pair v[1:2] is odd-aligned
    engine=engine,
)

if not rdna_report.is_grounded:
    for diag in rdna_report.diagnostics:
        print(f"RDNA Error: {diag.message}")
        print(f"-> Suggested Fix: {diag.suggested_fix}")

# ---------------------------------------------------------
# 4. StableHLO Compiler Dialect Grounding
# ---------------------------------------------------------
hlo_report = validate_stablehlo_op(
    op_name="stablehlo.dot_general",
    operand_types=["tensor<128x64xf32>", "tensor<64x256xf32>"],
    attributes={},  # Missing required 'dot_dimension_numbers'
    engine=engine,
)

if not hlo_report.is_grounded:
    for diag in hlo_report.diagnostics:
        print(f"StableHLO Error: {diag.message}")
```

---

## 8. Model Context Protocol (MCP) Server

The MCP server connects AI coding agents directly to ground-truth signatures and hardware rules via standard JSON-RPC 2.0.

### Starting the Server

```bash
ml_framework_snapshots mcp
```

### Example JSON-RPC Invocations

#### 1. Look Up API Signature (`get_api_signature`)

**Request:**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "get_api_signature",
    "arguments": {
      "framework": "torch",
      "api_path": "torch.nn.functional.linear"
    }
  }
}
```

**Response:**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "content": [
      {
        "type": "text",
        "text": "{\n  \"api_path\": \"torch.nn.functional.linear\",\n  \"kind\": \"function\",\n  \"params\": [\n    {\"name\": \"input\", \"kind\": \"POSITIONAL_OR_KEYWORD\", \"annotation\": \"Tensor\"},\n    {\"name\": \"weight\", \"kind\": \"POSITIONAL_OR_KEYWORD\", \"annotation\": \"Tensor\"},\n    {\"name\": \"bias\", \"kind\": \"POSITIONAL_OR_KEYWORD\", \"annotation\": \"Optional[Tensor]\", \"default\": \"None\"}\n  ],\n  \"returns_type\": \"Tensor\"\n}"
      }
    ]
  }
}
```

#### 2. Check for Hallucinated Arguments (`check_hallucination`)

**Request:**
```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/call",
  "params": {
    "name": "check_hallucination",
    "arguments": {
      "framework": "torch",
      "api_path": "torch.sum",
      "kwargs": ["input", "axis", "keepdim"]
    }
  }
}
```

**Response:**
```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "result": {
    "content": [
      {
        "type": "text",
        "text": "{\n  \"is_valid\": false,\n  \"hallucinations\": [\n    {\n      \"field\": \"kwargs.axis\",\n      \"message\": \"Unexpected keyword argument 'axis'.\",\n      \"suggested_fix\": \"dim\"\n    }\n  ]\n}"
      }
    ]
  }
}
```

#### 3. Cross-Framework Concept Translation (`translate_concept_arguments`)

**Request:**
```json
{
  "jsonrpc": "2.0",
  "id": 3,
  "method": "tools/call",
  "params": {
    "name": "translate_concept_arguments",
    "arguments": {
      "concept": "reduce_sum",
      "source_framework": "torch",
      "target_framework": "jax",
      "source_kwargs": {
        "dim": [1, 2],
        "keepdim": true
      }
    }
  }
}
```

**Response:**
```json
{
  "jsonrpc": "2.0",
  "id": 3,
  "result": {
    "content": [
      {
        "type": "text",
        "text": "{\n  \"target_framework\": \"jax\",\n  \"translated_kwargs\": {\n    \"axis\": [1, 2],\n    \"keepdims\": true\n  }\n}"
      }
    ]
  }
}
```

#### 4. Batch Code Block Verification (`check_code_block`)

**Request:**
```json
{
  "jsonrpc": "2.0",
  "id": 4,
  "method": "tools/call",
  "params": {
    "name": "check_code_block",
    "arguments": {
      "code": "import torch\ny = torch.sum(x, axis=1)\nz = torch.relu(y)",
      "framework": "torch"
    }
  }
}
```

**Response:**
```json
{
  "jsonrpc": "2.0",
  "id": 4,
  "result": {
    "content": [
      {
        "type": "text",
        "text": "{\n  \"total_analyzed\": 2,\n  \"hallucinations_detected\": 1,\n  \"is_valid\": false,\n  \"findings\": [\n    {\n      \"call\": \"torch.sum\",\n      \"argument\": \"axis\",\n      \"message\": \"Hallucinated keyword argument 'axis' for 'torch.sum'.\",\n      \"suggested_fix\": \"dim\"\n    }\n  ]\n}"
      }
    ]
  }
}
```
