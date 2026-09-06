"""Module docstring."""

from typing import Any
from unittest.mock import patch

import pytest

from ml_switcheroo_ir.schema.ghost import GhostParam
from ml_switcheroo_ir.schema.ghost import GhostRef
from ml_framework_snapshots.export import (
    to_json_schema,
    to_openapi,
    _ghost_to_cdd_ir,
    to_pydantic,
    to_protobuf,
    export_llm_prompt_context,
    export_sass_prompt_context,
    export_mlir_prompt_context,
    export_scoped_prompt_context,
)


@pytest.fixture
def sample_ghost_ref() -> None:
    """Function docstring.

    Returns:
        Return value.
    """
    return GhostRef(  # type: ignore
        name="Linear",
        api_path="torch.nn.Linear",
        kind="class",
        docstring="Applies a linear transformation to the incoming data.",
        params=[
            GhostParam(
                name="in_features",
                kind="POSITIONAL_OR_KEYWORD",
                annotation="int",
                description="size of each input sample",
            ),
            GhostParam(
                name="out_features",
                kind="POSITIONAL_OR_KEYWORD",
                annotation="int",
                description="size of each output sample",
            ),
            GhostParam(
                name="bias",
                kind="POSITIONAL_OR_KEYWORD",
                annotation="bool",
                default="True",
                description='If set to "False", the layer will not learn an additive bias.',
            ),
            GhostParam(name="args", kind="VAR_POSITIONAL", annotation="Any"),
        ],
        returns_type="torch.Tensor",
        returns_description="A tensor of shape",
    )


def test_ghost_to_cdd_ir(sample_ghost_ref: Any) -> None:
    """Function docstring.

    Args:
        sample_ghost_ref: description
    """
    ir = _ghost_to_cdd_ir(sample_ghost_ref)

    assert ir["name"] == "Linear"
    assert ir["type"] == "class"
    assert ir["doc"] == "Applies a linear transformation to the incoming data."

    params = ir["params"]
    assert "in_features" in params
    assert params["in_features"]["typ"] == "int"
    assert params["in_features"]["doc"] == "size of each input sample"

    assert "out_features" in params
    assert params["out_features"]["typ"] == "int"

    assert "bias" in params
    assert params["bias"]["typ"] == "bool"
    assert params["bias"]["default"] == "True"

    returns = ir["returns"]
    assert returns is not None
    assert "return_type" in returns
    assert returns["return_type"]["typ"] == "torch.Tensor"
    assert returns["return_type"]["doc"] == "A tensor of shape"


def test_to_json_schema(sample_ghost_ref: Any) -> None:
    """Function docstring.

    Args:
        sample_ghost_ref: description
    """
    schema = to_json_schema(sample_ghost_ref)

    assert schema["$id"] == "torch.nn.Linear"
    assert schema["description"].startswith(
        "Applies a linear transformation to the incoming data."
    )
    assert schema["type"] == "object"

    properties = schema.get("properties", {})
    assert "in_features" in properties
    assert "out_features" in properties
    assert "bias" in properties


def test_to_openapi(sample_ghost_ref: Any) -> None:
    """Function docstring.

    Args:
        sample_ghost_ref: description
    """
    openapi_dict = to_openapi([sample_ghost_ref])

    assert openapi_dict["openapi"] == "3.0.0"

    paths = openapi_dict.get("paths", {})
    assert "/torch/nn/Linear" in paths

    components = openapi_dict.get("components", {})
    schemas = components.get("schemas", {})
    assert "Linear" in schemas


def test_to_pydantic(sample_ghost_ref: Any) -> None:
    """Function docstring.

    Args:
        sample_ghost_ref: description
    """
    code = to_pydantic(sample_ghost_ref)
    assert "class Linear(BaseModel):" in code
    assert (
        'in_features: int = Field(..., description="size of each input sample")' in code
    )
    assert (
        'out_features: int = Field(..., description="size of each output sample")'
        in code
    )
    assert (
        "bias: bool = Field(default=True, description=\"If set to 'False', the layer will not learn an additive bias.\")"
        in code
    )
    assert "args" not in code  # VAR_POSITIONAL skipped


def test_to_pydantic_empty() -> None:
    """Function docstring."""
    ref = GhostRef(name="Empty", api_path="a.Empty", kind="class", params=[])
    code = to_pydantic(ref)
    assert "class Empty(BaseModel):" in code
    assert "pass" in code


def test_to_protobuf(sample_ghost_ref: Any) -> None:
    """Function docstring.

    Args:
        sample_ghost_ref: description
    """
    code = to_protobuf(sample_ghost_ref)
    assert 'syntax = "proto3";' in code
    assert "message Linear {" in code
    assert "int64 in_features = 1;" in code
    assert "int64 out_features = 2;" in code
    assert "optional bool bias = 3;" in code
    assert "args" not in code


def test_to_protobuf_types() -> None:
    """Function docstring."""
    ref = GhostRef(
        name="Types",
        api_path="a.Types",
        kind="class",
        params=[
            GhostParam(name="s", kind="POSITIONAL_ONLY", annotation="str"),
            GhostParam(name="l", kind="POSITIONAL_ONLY", annotation="list[int]"),
            GhostParam(name="d", kind="POSITIONAL_ONLY", annotation="dict[str, int]"),
            GhostParam(name="f", kind="POSITIONAL_ONLY", annotation="float"),
            GhostParam(name="u", kind="POSITIONAL_ONLY", annotation=None),
            GhostParam(name="unknown", kind="POSITIONAL_ONLY", annotation="weird_type"),
        ],
    )
    code = to_protobuf(ref)
    assert "string s = 1;" in code
    assert "repeated string l = 2;" in code
    assert "map<string, string> d = 3;" in code
    assert "double f = 4;" in code
    assert "string u = 5;" in code
    assert "string unknown = 6;" in code


def test_export_branches() -> None:
    """Function docstring."""
    from ml_framework_snapshots.export import (
        _py_type_to_proto,
        to_pydantic,
        to_json_schema,
        _ghost_to_cdd_ir,
    )

    assert _py_type_to_proto(None) == "string"  # type: ignore
    assert _py_type_to_proto("") == "string"

    from ml_switcheroo_ir.schema.ghost import GhostParam
    from ml_switcheroo_ir.schema.ghost import GhostRef

    r = GhostRef(
        name="X",
        api_path="X",
        kind="function",
        params=[
            GhostParam(
                name="p1", kind="KEYWORD_ONLY", default="None", annotation="int"
            ),
            GhostParam(
                name="p2", kind="VAR_POSITIONAL", default="None", annotation="int"
            ),
        ],
    )
    to_pydantic(r)
    to_json_schema(r)

    r_empty_anno = GhostRef(
        name="Y",
        api_path="Y",
        kind="function",
        params=[GhostParam(name="p_empty", kind="KEYWORD_ONLY")],
        returns_type="str",
    )  # No returns_description
    _ghost_to_cdd_ir(r_empty_anno)

    r_ret_desc = GhostRef(
        name="Z", api_path="Z", kind="function", params=[], returns_description="desc"
    )
    _ghost_to_cdd_ir(r_ret_desc)


def test_export_branches_more() -> None:
    """Function docstring."""
    from ml_framework_snapshots.export import (
        to_pydantic,
        to_openapi,
        to_json_schema,
    )
    from ml_switcheroo_ir.schema.ghost import GhostParam
    from ml_switcheroo_ir.schema.ghost import GhostRef

    r = GhostRef(
        name="X",
        api_path="X",
        kind="class",
        docstring="doc",
        params=[
            GhostParam(
                name="p1", kind="KEYWORD_ONLY", default="None", annotation="int"
            ),
            GhostParam(
                name="p2", kind="VAR_POSITIONAL", default="None", annotation="int"
            ),
        ],
    )
    to_pydantic(r)
    to_openapi([r])
    to_json_schema(r)

    r2 = GhostRef(
        name="X",
        api_path="X",
        kind="function",
        returns_type="int",
        returns_description="desc",
        params=[
            GhostParam(
                name="p1",
                kind="KEYWORD_ONLY",
                default="None",
                annotation="int",
                description="desc",
            ),
        ],
    )
    to_pydantic(r2)


def test_export_llm_prompt_context_full(sample_ghost_ref: GhostRef) -> None:
    """Test export_llm_prompt_context with complete GhostRef containing raises, env tags, returns, docstring."""
    sample_ghost_ref.raises = ["ValueError"]
    sample_ghost_ref.environment_tags = ["cpu", "cuda"]
    context = export_llm_prompt_context([sample_ghost_ref])
    assert "### `torch.nn.Linear`" in context
    assert "- **Signature**:" in context
    assert "- **Parameters**:" in context
    assert "-> torch.Tensor" in context
    assert "- **Raises**:" in context
    assert "ValueError" in context
    assert "- **Environments**: cpu, cuda" in context
    assert "- **Summary**:" in context

    # Test minimal GhostRef covering all False branches in export_llm_prompt_context
    bare_ref = GhostRef(name="bare", api_path="pkg.bare", kind="function", params=[])
    context_bare = export_llm_prompt_context([bare_ref])
    assert "### `pkg.bare`" in context_bare
    assert "- **Parameters**:" not in context_bare


def test_export_sass_prompt_context() -> None:
    """Test exporting SASS instruction prompt context with assembly syntax templates."""
    from ml_framework_snapshots.models import (
        ExtendedGhostRef,
        ExtendedGhostParam,
        OperandDirection,
        IRParameterRole,
    )

    sass_ref = ExtendedGhostRef(
        name="FADD",
        api_path="nvidia_sass.inst.FADD",
        kind="function",
        docstring="Floating point add instruction.",
        params=[
            ExtendedGhostParam(
                name="op0",
                kind="POSITIONAL_ONLY",
                annotation="R",
                standardized_name="dst",
                direction=OperandDirection.WRITE,
                role=IRParameterRole.OPERAND,
            ),
            ExtendedGhostParam(
                name="op1",
                kind="POSITIONAL_ONLY",
                annotation="R",
                standardized_name="src0",
                direction=OperandDirection.READ,
                role=IRParameterRole.OPERAND,
            ),
        ],
        domain_metadata={
            "modifiers": [".FTZ", ".SAT"],
            "valid_architectures": ["sm_70", "sm_80", "sm_90"],
            "operand_signatures": [["R", "R", "R"]],
        },
    )

    context = export_sass_prompt_context([sass_ref])
    assert "### `FADD`" in context
    assert "[@P0] FADD" in context
    assert "sm_70" in context
    assert ".FTZ" in context
    assert "WRITE" in context

    # Test minimal SASS ref with no operands, no modifiers, and no metadata
    sass_minimal = ExtendedGhostRef(
        name="NOP",
        api_path="nvidia_sass.inst.NOP",
        kind="function",
        params=[
            ExtendedGhostParam(
                name="raw_op",
                kind="POSITIONAL_ONLY",
            )
        ],
    )
    context_min = export_sass_prompt_context([sass_minimal])
    assert "### `NOP`" in context_min
    assert "[@P0] NOP;" in context_min
    assert "raw_op" in context_min

    # Test SASS ref with only "cuda" in environment_tags and no params/modifiers
    sass_cuda_only = ExtendedGhostRef(
        name="SYNC",
        api_path="nvidia_sass.inst.SYNC",
        kind="function",
        environment_tags=["cuda"],
        params=[],
    )
    context_cuda = export_sass_prompt_context([sass_cuda_only])
    assert "### `SYNC`" in context_cuda
    assert "Architectures" not in context_cuda
    assert "Modifiers" not in context_cuda


def test_export_mlir_prompt_context() -> None:
    """Test exporting MLIR/StableHLO prompt context with SSA syntax templates."""
    from ml_framework_snapshots.models import (
        ExtendedGhostRef,
        ExtendedGhostParam,
        GhostResult,
        IRParameterRole,
    )

    mlir_ref = ExtendedGhostRef(
        name="AddFOp",
        api_path="arith.addf",
        kind="function",
        docstring="Floating point add operation.",
        returns_type="FloatLike",
        returns=[GhostResult(name="result", type="FloatLike")],
        params=[
            ExtendedGhostParam(
                name="lhs",
                kind="POSITIONAL_OR_KEYWORD",
                annotation="FloatLike",
                role=IRParameterRole.OPERAND,
            ),
            ExtendedGhostParam(
                name="fastmath",
                kind="KEYWORD_ONLY",
                annotation="FastMathFlagsAttr",
                role=IRParameterRole.ATTRIBUTE,
            ),
            ExtendedGhostParam(
                name="body_reg",
                kind="KEYWORD_ONLY",
                annotation="Region",
                role=IRParameterRole.REGION,
            ),
            GhostParam(
                name="region_untyped",
                kind="KEYWORD_ONLY",
                annotation="Region",
            ),
        ],
        domain_metadata={
            "traits": ["Commutative", "Pure"],
            "regions": ["body"],
        },
    )

    context = export_mlir_prompt_context([mlir_ref])
    assert "### `arith.addf`" in context
    assert "%res = arith.addf(%lhs)" in context
    assert "{fastmath = ...}" in context
    assert "%lhs" in context
    assert "FastMathFlagsAttr" in context
    assert "%result" in context
    assert "Commutative" in context
    assert "**Regions**: body" in context

    # Test minimal MLIR ref without operands or attributes or returns
    mlir_minimal = ExtendedGhostRef(
        name="BarrierOp",
        api_path="gpu.barrier",
        kind="function",
    )
    context_min = export_mlir_prompt_context([mlir_minimal])
    assert "### `gpu.barrier`" in context_min
    assert "%res = gpu.barrier() : none" in context_min


def test_export_scoped_prompt_context() -> None:
    """Test export_scoped_prompt_context with hierarchical indexing and scoping across domains."""
    # 1. Scoped torch context with module prefix
    torch_ctx = export_scoped_prompt_context(
        "torch", module_prefix="torch.nn", max_symbols=3
    )
    assert "# Framework Grounding Context: `torch`" in torch_ctx
    assert "## Index of Available Operations" in torch_ctx
    assert "## Detailed Operation Signatures" in torch_ctx
    assert "torch.nn" in torch_ctx
    assert "*(Showing 3 of" in torch_ctx

    # 2. Scoped SASS context
    sass_ctx = export_scoped_prompt_context("nvidia_sass", max_symbols=5)
    assert "# Framework Grounding Context: `nvidia_sass`" in sass_ctx
    assert "- **Operands**:" in sass_ctx or "- **Modifiers**:" in sass_ctx

    # 3. Scoped StableHLO context
    stablehlo_ctx = export_scoped_prompt_context("stablehlo", max_symbols=5)
    assert "# Framework Grounding Context: `stablehlo`" in stablehlo_ctx
    assert "- **Syntax**:" in stablehlo_ctx

    # 4. Scoped context when total_matching <= max_symbols and with corrupt item
    mock_snap = {
        "categories": {
            "test": [
                {"name": "op1", "api_path": "test.op1", "kind": "function"},
                "corrupt_non_dict_item",
                {"api_path": "test.invalid_fields", "kind": 12345},
            ]
        }
    }
    with patch(
        "ml_framework_snapshots.mcp_server.get_framework_snapshot",
        return_value=mock_snap,
    ):
        ctx = export_scoped_prompt_context("test_fw", max_symbols=50)
        assert "# Framework Grounding Context: `test_fw`" in ctx
        assert "*(Showing" not in ctx
