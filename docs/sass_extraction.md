# NVIDIA SASS Instruction Extraction & Provenance Guide

This document details the provenance, methodology, and commands for extracting ground-truth **NVIDIA SASS (Streaming Assembler)** instruction sets across GPU microarchitectures.

---

## 1. Provenance & Toolchain Requirements

- **Primary Source:** NVIDIA CUDA Toolkit Binary Utilities (`nvdisasm`, `cuobjdump`).
- **Minimum CUDA Version:** CUDA Toolkit 12.0+ (supports `sm_70` through `sm_100`).
- **Target Architectures:**
  - `sm_70`: NVIDIA Volta (e.g. V100)
  - `sm_75`: NVIDIA Turing (e.g. T4, RTX 2080)
  - `sm_80`: NVIDIA Ampere (e.g. A100)
  - `sm_89`: NVIDIA Ada Lovelace (e.g. RTX 4090, L40)
  - `sm_90`: NVIDIA Hopper (e.g. H100)
  - `sm_100`: NVIDIA Blackwell (e.g. B200)

---

## 2. Automated Extraction Workflow

### Method A: Extracting via `nvdisasm --binary-info`

To extract instructions, operand encodings, and valid modifier options from compiled CUDA ELF binaries:

```bash
# 1. Compile a comprehensive CUDA binary targeting the architecture
nvcc -gencode arch=compute_80,code=sm_80 -gencode arch=compute_90,code=sm_90 -cubin kernel.cu -o kernel.cubin

# 2. Extract detailed instruction binary information
nvdisasm --binary-info kernel.cubin > sass_binary_info.txt

# 3. Process the disassembly into ground-truth JSON
python -m ml_framework_snapshots.tools.scrape_nvidia_sass --input sass_binary_info.txt
```

### Method B: Parsing CUDA Binary Utilities ISA Metadata

When utilizing structured ISA definitions exported from the CUDA disassembler tables:

```bash
# Provide the path via environment variable or command-line argument:
export NVIDIA_SASS_INPUT_PATH=/path/to/cuda_isa_metadata.json
python -m ml_framework_snapshots.tools.scrape_nvidia_sass
```

> **Deprecation Notice:** Reliance on unversioned `/tmp/isa.json` is deprecated. Always supply reproducible input files or capture via `nvdisasm --binary-info`.

---

## 3. Instruction Schema & Representation

Each instruction in `nvidia_sass_exhaustive.json` conforms to:

- `mnemonic`: Base opcode (e.g., `FADD`, `FFMA`, `MOV`, `LDG`).
- `architecture`: Supported SM range (e.g., `sm_80+` representing `sm_80`, `sm_89`, `sm_90`, `sm_100`).
- `description`: Formal instruction description.
- `modifiers`: Structured modifiers (`.SAT`, `.RN`, `.RZ`, `.FTZ`, `.STRONG`, `.CG`, `.CS`).
- `operands`: Concrete operand signature combinations, distinguishing:
  - `R`: 32-bit vector general-purpose register.
  - `UR`: Uniform register.
  - `P`: Predicate register.
  - `I`: Integer immediate value.
  - `FI`: Floating-point immediate value.
  - `c[bank][offset]`: Constant bank memory reference.
  - `ADDR`: Memory address expression.
