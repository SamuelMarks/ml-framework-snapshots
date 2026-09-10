"""Tests for GhostRef and GhostInspector environment_tags without host environment leakage."""

from typing import Any

from ml_framework_snapshots.models import (
    GhostInspector,
    FRAMEWORK_CAPABILITIES,
    OperandDirection,
    IRParameterRole,
    GhostResult,
    ExtendedGhostParam,
    ExtendedGhostRef,
    GhostPythonRef,
    GhostIsaRef,
    GhostMlirRef,
)
from ml_switcheroo_ir.schema.ghost import GhostRef, ParameterKind
import sys


def test_ghostref_env_tags() -> None:
    """Test GhostRef initialization with custom environment_tags."""
    ref = GhostRef(
        name="test",
        api_path="test",
        kind="function",
        environment_tags=["darwin", "cpu"],
    )
    assert ref.environment_tags is not None
    assert "darwin" in ref.environment_tags
    assert "cpu" in ref.environment_tags


def test_ghostinspector_env_tags_deterministic() -> None:
    """Test that GhostInspector derives deterministic capabilities and does not leak sys.platform."""

    def dummy() -> Any:
        """Dummy function for inspection."""
        pass  # pragma: no cover

    # Test unknown framework defaults to ['cpu']
    ref = GhostInspector.inspect(dummy, "dummy")
    assert ref.environment_tags == ["cpu"]
    assert sys.platform not in (ref.environment_tags or [])

    # Test torch maps to declared framework capabilities
    ref_torch = GhostInspector.inspect(dummy, "torch.nn.functional.relu")
    assert ref_torch.environment_tags == FRAMEWORK_CAPABILITIES["torch"]
    assert "cuda" in (ref_torch.environment_tags or [])
    assert "metal" in (ref_torch.environment_tags or [])
    assert sys.platform not in (ref_torch.environment_tags or [])

    # Test numpy maps to cpu
    ref_numpy = GhostInspector.inspect(dummy, "numpy.add")
    assert ref_numpy.environment_tags == ["cpu"]


def test_ghostinspector_env_tags_explicit_override() -> None:
    """Test that passing explicit environment_tags overrides default capabilities."""

    def dummy() -> Any:
        """Dummy function for inspection."""
        pass  # pragma: no cover

    custom_tags = ["cuda", "rocm", "custom_backend"]
    ref = GhostInspector.inspect(
        dummy, "torch.ops.custom", environment_tags=custom_tags
    )
    assert ref.environment_tags == custom_tags


def test_extended_models_schema() -> None:
    """Test ExtendedGhostParam, ExtendedGhostRef, and compiler/ISA enums."""
    param = ExtendedGhostParam(
        name="dst",
        kind=ParameterKind.POSITIONAL_ONLY,
        direction=OperandDirection.WRITE,
        role=IRParameterRole.OPERAND,
    )
    assert param.direction == OperandDirection.WRITE
    assert param.role == IRParameterRole.OPERAND

    result = GhostResult(
        name="%0",
        type="tensor<2x2xf32>",
        description="Output SSA value",
    )
    assert result.name == "%0"
    assert result.type == "tensor<2x2xf32>"

    ref = ExtendedGhostRef(
        name="test_op",
        api_path="test.op",
        kind="function",
        params=[param],
        returns=[result],
        domain_metadata={"modifiers": [".SAT", ".RN"]},
    )
    assert ref.returns is not None
    assert len(ref.returns) == 1
    assert ref.domain_metadata == {"modifiers": [".SAT", ".RN"]}


def test_polymorphic_ref_models_and_extra_allow() -> None:
    """Test GhostPythonRef, GhostIsaRef, GhostMlirRef, and that extra metadata is not stripped."""
    # 1. Extra fields preserved by extra='allow'
    param_data = {
        "name": "x",
        "kind": "POSITIONAL_OR_KEYWORD",
        "direction": "READ",
        "role": "OPERAND",
        "custom_ir_tag": "llvm_lowered",
    }
    p = ExtendedGhostParam.model_validate(param_data)
    assert getattr(p, "custom_ir_tag") == "llvm_lowered"

    ref_data = {
        "name": "op",
        "api_path": "pkg.op",
        "kind": "function",
        "domain_type": "python",
        "custom_backend_meta": {"cost": 10},
    }
    # 2. Polymorphic hydration
    py_ref = GhostInspector.hydrate(ref_data)
    assert isinstance(py_ref, GhostPythonRef)
    assert py_ref.domain_type == "python"
    assert getattr(py_ref, "custom_backend_meta") == {"cost": 10}

    isa_data = {
        "name": "FADD",
        "api_path": "nvidia_sass.inst.FADD",
        "kind": "function",
        "domain_type": "isa",
        "predicate_guards": ["@P0", "@!P0"],
        "register_classes": {"op0": "VGPR_32"},
        "control_codes": {"stall_count": 1},
        "instruction_modifiers": [".SAT", ".FTZ"],
        "supported_architectures": ["sm_80", "sm_90"],
    }
    isa_ref = GhostInspector.hydrate(isa_data)
    assert isinstance(isa_ref, GhostIsaRef)
    assert isa_ref.domain_type == "isa"
    assert isa_ref.predicate_guards == ["@P0", "@!P0"]
    assert isa_ref.register_classes == {"op0": "VGPR_32"}
    assert isa_ref.control_codes == {"stall_count": 1}
    assert isa_ref.instruction_modifiers == [".SAT", ".FTZ"]
    assert isa_ref.supported_architectures == ["sm_80", "sm_90"]

    mlir_data = {
        "name": "AddFOp",
        "api_path": "arith.addf",
        "kind": "function",
        "domain_type": "mlir",
        "traits": ["SameOperandsAndResultType", "Commutative"],
        "attributes": {"fastmath": "fast"},
        "regions": {"body": []},
        "successors": ["bb1"],
    }
    mlir_ref = GhostInspector.hydrate(mlir_data)
    assert isinstance(mlir_ref, GhostMlirRef)
    assert mlir_ref.domain_type == "mlir"
    assert mlir_ref.traits == ["SameOperandsAndResultType", "Commutative"]
    assert mlir_ref.attributes == {"fastmath": "fast"}
    assert mlir_ref.regions == {"body": []}
    assert mlir_ref.successors == ["bb1"]

    # Default fallback
    base_data = {
        "name": "GenericOp",
        "api_path": "generic.op",
        "kind": "function",
    }
    base_ref = GhostInspector.hydrate(base_data)
    assert isinstance(base_ref, ExtendedGhostRef)

    # 3. Raw dumps hydration synthesis
    raw_isa_data = {
        "mnemonic": "FADD_RAW",
        "architecture": ["sm_80", "sm_90"],
        "operands": [["R", "R"], ["R"]],
        "modifiers": ["FTZ"],
    }
    raw_isa_ref = GhostInspector.hydrate(raw_isa_data)
    assert isinstance(raw_isa_ref, GhostIsaRef)
    assert raw_isa_ref.name == "FADD_RAW"
    assert raw_isa_ref.environment_tags == ["sm_80", "sm_90"]

    raw_isa_str = {
        "mnemonic": "FADD_STR",
        "architecture": "sm_70",
        "operands": [["R"]],
    }
    raw_str_ref = GhostInspector.hydrate(raw_isa_str)
    assert raw_str_ref.environment_tags == ["sm_70"]

    raw_isa_no_ops = {"mnemonic": "NOP"}
    raw_nop_ref = GhostInspector.hydrate(raw_isa_no_ops)
    assert isinstance(raw_nop_ref, GhostIsaRef)
    assert len(raw_nop_ref.params) == 0

    raw_mlir_data = {
        "api_path": "test.op",
        "operands": ["a"],
        "attributes": ["b"],
        "results": [{"name": "out", "type": "tensor"}],
    }
    raw_mlir_ref = GhostInspector.hydrate(raw_mlir_data)
    assert isinstance(raw_mlir_ref, GhostMlirRef)
    assert raw_mlir_ref.name == "op"

    raw_mlir_dict_data = {
        "api_path": "test.op2",
        "operands": [{"name": "x"}],
        "attributes": [{"name": "attr"}],
        "results": ["res"],
    }
    raw_mlir_dict_ref = GhostInspector.hydrate(raw_mlir_dict_data)
    assert isinstance(raw_mlir_dict_ref, GhostMlirRef)
    assert raw_mlir_dict_ref.name == "op2"
