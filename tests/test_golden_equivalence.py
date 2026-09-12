"""Golden equivalence tests asserting exact operation presence and scraper cleanliness."""

import json
import os


def test_golden_symbol_counts_and_mandatory_ops() -> None:
    """Verify that golden datasets contain all mandatory operations and minimum symbol thresholds."""
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    fw_dir = os.path.join(repo_root, "src", "ml_framework_snapshots", "frameworks")

    # 1. StableHLO
    shlo_path = os.path.join(fw_dir, "stablehlo_exhaustive.json")
    if not os.path.isfile(shlo_path):
        import pytest

        pytest.skip(
            "Exhaustive JSON files not present on clean clone; bundled into wheels and releases."
        )

    with open(shlo_path, "r", encoding="utf-8") as f:
        shlo_data = json.load(f)
    assert len(shlo_data) >= 100, f"StableHLO golden count too low: {len(shlo_data)}"
    shlo_names = {op.get("api_path") or op.get("name") for op in shlo_data}
    for req in [
        "stablehlo.dot_general",
        "stablehlo.convolution",
        "stablehlo.reduce",
        "stablehlo.while",
        "stablehlo.gather",
        "stablehlo.scatter",
        "stablehlo.add",
        "stablehlo.multiply",
    ]:
        assert req in shlo_names, f"Mandatory StableHLO operation missing: {req}"

    # 2. Core MLIR
    with open(os.path.join(fw_dir, "mlir_exhaustive.json"), "r", encoding="utf-8") as f:
        mlir_data = json.load(f)
    assert len(mlir_data) >= 300, f"MLIR golden count too low: {len(mlir_data)}"
    mlir_names = {op.get("api_path") or op.get("name") for op in mlir_data}
    for req in [
        "arith.addf",
        "arith.mulf",
        "math.exp",
        "tensor.extract",
        "linalg.matmul",
        "scf.for",
    ]:
        assert req in mlir_names, f"Mandatory MLIR operation missing: {req}"

    # 3. NVIDIA SASS
    with open(
        os.path.join(fw_dir, "nvidia_sass_exhaustive.json"), "r", encoding="utf-8"
    ) as f:
        sass_data = json.load(f)
    assert len(sass_data) >= 500, f"SASS golden count too low: {len(sass_data)}"
    sass_mnemonics = {
        (op.get("mnemonic") or op.get("name") or "").upper() for op in sass_data
    }
    for req in ["FFMA", "HMMA", "WGMMA", "LDG", "STG", "BRA", "EXIT"]:
        assert req in sass_mnemonics, f"Mandatory SASS instruction missing: {req}"

    # 4. NVIDIA PTX
    with open(
        os.path.join(fw_dir, "nvidia_ptx_exhaustive.json"), "r", encoding="utf-8"
    ) as f:
        ptx_data = json.load(f)
    assert len(ptx_data) >= 200, f"PTX golden count too low: {len(ptx_data)}"
    ptx_names = {
        (op.get("mnemonic") or op.get("name") or "").lower() for op in ptx_data
    }
    for req in ["add", "sub", "mul", "atom", "and", "ret"]:
        assert req in ptx_names, f"Mandatory PTX instruction missing: {req}"

    # 5. AMD RDNA
    with open(
        os.path.join(fw_dir, "amd_rdna_exhaustive.json"), "r", encoding="utf-8"
    ) as f:
        rdna_data = json.load(f)
    assert len(rdna_data) >= 1000, f"RDNA golden count too low: {len(rdna_data)}"
    rdna_names = {
        (op.get("mnemonic") or op.get("name") or "").upper() for op in rdna_data
    }
    for req in ["V_ADD_F32", "V_FMAC_F32", "V_MUL_F32"]:
        assert req in rdna_names, f"Mandatory RDNA instruction missing: {req}"


def test_golden_datasets_zero_leak_tokens() -> None:
    """Verify zero unparsed regex or scraper leak tokens in any exhaustive dataset."""
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    fw_dir = os.path.join(repo_root, "src", "ml_framework_snapshots", "frameworks")

    leak_tokens = [
        "???0",
        "???1",
        "SNOWFLAKE",
        "<placeholder>",
        "UNKNOWN_OPERAND",
        "AttributeOperand",
        "DescOperand",
    ]

    for fname in [
        "stablehlo_exhaustive.json",
        "mlir_exhaustive.json",
        "nvidia_sass_exhaustive.json",
        "nvidia_ptx_exhaustive.json",
        "amd_rdna_exhaustive.json",
    ]:
        fpath = os.path.join(fw_dir, fname)
        if not os.path.isfile(fpath):
            import pytest

            pytest.skip(
                f"Exhaustive JSON file {fname} not present on clean clone; bundled into wheels and releases."
            )
        with open(fpath, "r", encoding="utf-8") as f:
            content = f.read()

        for token in leak_tokens:
            assert token not in content, f"Leak token '{token}' discovered in {fname}"
