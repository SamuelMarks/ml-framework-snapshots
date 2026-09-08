"""Unit tests for Section 6 Schema Export & Tooling Fidelity.

Tests Pydantic V2 model generation with overloaded signatures and validated tuples,
Protobuf structured type mapping and enum preservation, and Scoped LLM prompt context export.
"""

from unittest.mock import patch

from ml_framework_snapshots.export import (
    COMMON_HALLUCINATION_GUARDS,
    export_scoped_prompt_context,
    to_protobuf,
    to_pydantic,
)
from ml_switcheroo_ir.schema.ghost import GhostParam, GhostRef


def test_to_pydantic_overloaded_signatures() -> None:
    """Test to_pydantic generating typing.Union of Pydantic models for overloaded signatures."""
    primary = GhostRef(
        name="add",
        api_path="torch.add",
        kind="function",
        docstring="Add two tensors.",
        params=[
            GhostParam(
                name="input",
                kind="POSITIONAL_OR_KEYWORD",
                annotation="Tensor",
                description="The input tensor.",
            ),
            GhostParam(
                name="other",
                kind="POSITIONAL_OR_KEYWORD",
                annotation="Tensor",
                description="The second tensor.",
            ),
        ],
        overloads=[
            GhostRef(
                name="add",
                api_path="torch.add",
                kind="function",
                docstring="Add tensor and scalar.",
                params=[
                    GhostParam(
                        name="input",
                        kind="POSITIONAL_OR_KEYWORD",
                        annotation="Tensor",
                    ),
                    GhostParam(
                        name="other",
                        kind="POSITIONAL_OR_KEYWORD",
                        annotation="float",
                    ),
                ],
            )
        ],
    )

    code = to_pydantic(primary)
    assert "class addVariant0(BaseModel):" in code
    assert "class addVariant1(BaseModel):" in code
    assert "add = Union[addVariant0, addVariant1]" in code
    assert 'description="The input tensor."' in code
    assert 'description="The second tensor."' in code


def test_to_pydantic_variable_positional_tuple() -> None:
    """Test to_pydantic representing variable positional and keyword arguments as validated fields."""
    ref = GhostRef(
        name="VarOp",
        api_path="my.VarOp",
        kind="function",
        params=[
            GhostParam(
                name="args",
                kind="VAR_POSITIONAL",
                annotation="int",
                description="Variable positional integer arguments.",
            ),
            GhostParam(
                name="kwargs",
                kind="VAR_KEYWORD",
                annotation="Any",
                description="Arbitrary keyword arguments.",
            ),
        ],
    )

    code = to_pydantic(ref, validate_varargs=True)
    assert "class VarOp(BaseModel):" in code
    assert "args: Tuple[int, ...] = Field(default=(), description=" in code
    assert "kwargs: Dict[str, Any] = Field(default_factory=dict, description=" in code


def test_to_protobuf_structured_types_and_enums() -> None:
    """Test to_protobuf mapping TensorProto, repeated int64 shape, and preserving enums."""
    ref = GhostRef(
        name="Conv2dOp",
        api_path="torch.nn.functional.conv2d",
        kind="function",
        docstring="Applies 2D convolution.",
        params=[
            GhostParam(
                name="input",
                kind="POSITIONAL_OR_KEYWORD",
                annotation="Tensor",
                description="Input feature map tensor.",
            ),
            GhostParam(
                name="shape",
                kind="POSITIONAL_OR_KEYWORD",
                annotation="Tuple[int, ...]",
                description="Kernel spatial dimensions.",
            ),
            GhostParam(
                name="reduction",
                kind="POSITIONAL_OR_KEYWORD",
                annotation="str",
                default="mean",
                description="Loss reduction mode.",
            ),
            GhostParam(
                name="padding",
                kind="POSITIONAL_OR_KEYWORD",
                annotation="str",
                default="zeros",
                description="Padding mode.",
            ),
            GhostParam(
                name="layout",
                kind="POSITIONAL_OR_KEYWORD",
                annotation="str",
                default="NCHW",
                description="Memory layout format.",
            ),
        ],
    )

    proto = to_protobuf(ref)
    assert 'syntax = "proto3";' in proto
    # TensorProto structured message
    assert "message TensorProto {" in proto
    assert "TensorProto input = 1;" in proto
    # repeated int64 shape
    assert "repeated int64 shape = 2;" in proto
    # Preserved enums
    assert "enum ReductionType {" in proto
    assert "optional ReductionType reduction = 3;" in proto
    assert "enum PaddingMode {" in proto
    assert "optional PaddingMode padding = 4;" in proto
    assert "enum LayoutMode {" in proto
    assert "optional LayoutMode layout = 5;" in proto


def test_export_scoped_prompt_context_with_hallucination_guards() -> None:
    """Test export_scoped_prompt_context injecting hallucination guards and parameter constraints."""
    mock_torch_snap = {
        "categories": {
            "math": [
                {
                    "name": "inv",
                    "api_path": "torch.linalg.inv",
                    "kind": "function",
                    "docstring": "Computes inverse of square matrix.",
                    "params": [
                        {
                            "name": "A",
                            "kind": "POSITIONAL_OR_KEYWORD",
                            "annotation": "Tensor",
                            "dtypes": [
                                "float32",
                                "float64",
                                "complex64",
                                "complex128",
                            ],
                            "rank": ">=2",
                        }
                    ],
                }
            ]
        }
    }

    with patch(
        "ml_framework_snapshots.mcp_server.get_framework_snapshot",
        return_value=mock_torch_snap,
    ):
        ctx = export_scoped_prompt_context("torch", max_symbols=10)
        # Verify common hallucination guards
        assert "## Common Hallucination Guards & Anti-Patterns" in ctx
        for guard in COMMON_HALLUCINATION_GUARDS["torch"]:
            assert guard in ctx

        # Verify dtypes and rank in parameter constraints
        assert "dtypes: ['float32', 'float64', 'complex64', 'complex128']" in ctx
        assert "rank: >=2" in ctx


def test_export_scoped_prompt_context_module_prefix_filtering() -> None:
    """Test export_scoped_prompt_context filtering with module_prefix."""
    mock_snap = {
        "categories": {
            "all": [
                {
                    "name": "Linear",
                    "api_path": "torch.nn.Linear",
                    "kind": "class",
                    "docstring": "Applies a linear transformation.",
                    "params": [],
                },
                {
                    "name": "Adam",
                    "api_path": "torch.optim.Adam",
                    "kind": "class",
                    "docstring": "Implements Adam algorithm.",
                    "params": [],
                },
            ]
        }
    }

    with patch(
        "ml_framework_snapshots.mcp_server.get_framework_snapshot",
        return_value=mock_snap,
    ):
        ctx_filtered = export_scoped_prompt_context("torch", module_prefix="torch.nn")
        assert "torch.nn.Linear" in ctx_filtered
        assert "torch.optim.Adam" not in ctx_filtered
