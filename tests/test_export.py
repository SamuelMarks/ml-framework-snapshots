"""Module docstring."""

import pytest

from ml_switcheroo_ir.schema.ghost import GhostParam
from ml_switcheroo_ir.schema.ghost import GhostRef
from ml_framework_snapshots.export import (
    to_json_schema,
    to_openapi,
    _ghost_to_cdd_ir,
    to_pydantic,
    to_protobuf,
)


@pytest.fixture
def sample_ghost_ref():
    """Function docstring."""
    return GhostRef(
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


def test_ghost_to_cdd_ir(sample_ghost_ref):
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


def test_to_json_schema(sample_ghost_ref):
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


def test_to_openapi(sample_ghost_ref):
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


def test_to_pydantic(sample_ghost_ref):
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


def test_to_pydantic_empty():
    """Function docstring."""
    ref = GhostRef(name="Empty", api_path="a.Empty", kind="class", params=[])
    code = to_pydantic(ref)
    assert "class Empty(BaseModel):" in code
    assert "pass" in code


def test_to_protobuf(sample_ghost_ref):
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


def test_to_protobuf_types():
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
