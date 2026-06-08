# Usage Examples

## Compliance Checking Examples

### Checking a specific file in `ml-switcheroo`

To check the compliance of the `ml-switcheroo` JAX framework adapter against the JAX definition snapshot:

```bash
ml_framework_snapshots check \
    ./snapshots/jax_v0.4.30.json \
    ~/repos/ml-switcheroo/src/ml_switcheroo/frameworks/jax.py \
    --reference-prefix jax \
    --target-prefix ml_switcheroo.frameworks.jax
```

**Output:**
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
| [ ] | jax | jax.numpy | fft | jax.numpy.fft | `module` | Fast Fourier Transform module. |
| [ ] | jax | jax.numpy | linalg | jax.numpy.linalg | `module` | Linear algebra module. |
```

### Checking an unrelated project (`onnx9000-jax`)

To check the compliance of an independent project like `onnx9000-jax` which has its own `pyproject.toml` and directory structure, point the `target_path` to its root (or `src` directory) and set the appropriate target prefix:

```bash
ml_framework_snapshots check \
    ./snapshots/jax_v0.4.30.json \
    ~/repos/ml-switcheroo/onnx9000/packages/python/onnx9000-jax \
    --reference-prefix jax \
    --target-prefix onnx9000.jax
```

**Output:**
```text
Extracting target APIs from /Users/samuel/repos/ml-switcheroo/onnx9000/packages/python/onnx9000-jax...
Scoring compliance...

--- Compliance Report ---
Overall Compliance: 14.2%

Breakdown by Module:
  - jax.numpy: 19.2% (10/52)

Missing APIs (2):

|   | Framework | Namespace | Symbol | FQN | Signature | Docstring |
|---|---|---|---|---|---|---|
| [ ] | jax | jax.numpy | zeros | jax.numpy.zeros | `(shape, dtype=None, ...)` | Return a new array of given shape and type, filled with zeros. |
| [ ] | jax | jax.numpy | ones | jax.numpy.ones | `(shape, dtype=None, ...)` | Return a new array of given shape and type, filled with ones. |
```
