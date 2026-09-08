"""Unit test suite for ATen operator schema introspection.

Tests native_functions.yaml parsing, tensor dtypes, rank constraints,
and semantic default value preservation.
"""

import inspect
from typing import Any, List

from ml_framework_snapshots.frameworks.torch import (
    extract_aten_c_extension_signature,
    get_aten_op_schema,
    infer_torch_dtype_and_rank,
    parse_native_functions_yaml,
)
from ml_framework_snapshots.mcp_server import check_hallucination, handle_mcp_message
from ml_framework_snapshots.models import (
    ExtendedGhostParam,
    GhostInspector,
    sanitize_param_default,
)


def test_sanitize_param_default_semantic_literals() -> None:
    """Test preservation of semantic literals and distinguishing mandatory from explicit defaults."""
    # Mandatory parameter (empty default)
    def_val, factory_str, is_mand = sanitize_param_default(inspect.Parameter.empty)
    assert def_val is None
    assert factory_str is None
    assert is_mand is True

    # Explicit None default
    def_val, factory_str, is_mand = sanitize_param_default(None)
    assert def_val == "None"
    assert factory_str is None
    assert is_mand is False

    # Boolean literals
    assert sanitize_param_default(False) == ("False", None, False)
    assert sanitize_param_default(True) == ("True", None, False)

    # String literal
    assert sanitize_param_default("mean") == ("'mean'", None, False)

    # Numeric literals
    assert sanitize_param_default(1e-5) == ("1e-05", None, False)
    assert sanitize_param_default(42) == ("42", None, False)

    # Ellipsis
    assert sanitize_param_default(...) == ("...", None, False)


def test_sanitize_param_default_callables_and_memory_addresses() -> None:
    """Test scrubbing non-deterministic memory addresses and handling callables."""

    # Named callable
    def dummy_factory() -> int:
        """Dummy factory for test.

        Returns:
            Integer constant 1.
        """
        return 1

    def_val, factory_str, is_mand = sanitize_param_default(dummy_factory)
    assert def_val == "<factory_default>"
    assert factory_str == "<function dummy_factory>"
    assert is_mand is False

    # Anonymous lambda
    anon = lambda x: x  # noqa: E731
    def_val, factory_str, is_mand = sanitize_param_default(anon)
    assert def_val == "<factory_default>"
    assert factory_str is not None
    assert "<lambda>" in factory_str
    assert is_mand is False

    # Object with memory address
    class CustomObj:
        """Custom object with memory address in repr."""

        def __repr__(self) -> str:
            """Representation with memory address.

            Returns:
                String representation containing hex memory address.
            """
            return "<CustomObj object at 0x7fa1234abcd>"

    def_val, factory_str, is_mand = sanitize_param_default(CustomObj())
    assert def_val == "<factory_default>"
    assert "0x" not in def_val
    assert is_mand is False

    # String with memory address
    class CustomStrObj:
        """Custom object with memory address in str."""

        def __repr__(self) -> str:
            """Safe repr.

            Returns:
                Safe class name string.
            """
            return "CustomStrObj"

        def __str__(self) -> str:
            """String with memory address.

            Returns:
                String containing hex memory address.
            """
            return "CustomStrObj at 0x7fa1234abcd"

    def_val, factory_str, is_mand = sanitize_param_default(CustomStrObj())
    assert def_val == "<factory_default>"
    assert "0x" not in def_val


def test_extended_ghost_param_fields() -> None:
    """Test ExtendedGhostParam fields: dtypes, rank, default_factory, is_mandatory."""
    param = ExtendedGhostParam(
        name="input",
        kind="POSITIONAL_OR_KEYWORD",
        default="None",
        annotation="Tensor",
        dtypes=["float32", "bfloat16", "float16"],
        rank=2,
        default_factory=None,
        is_mandatory=False,
    )
    assert param.name == "input"
    assert param.dtypes == ["float32", "bfloat16", "float16"]
    assert param.rank == 2
    assert param.is_mandatory is False

    dumped = param.model_dump()
    assert dumped["dtypes"] == ["float32", "bfloat16", "float16"]
    assert dumped["rank"] == 2
    assert dumped["is_mandatory"] is False


def test_infer_torch_dtype_and_rank() -> None:
    """Test infer_torch_dtype_and_rank for linear algebra, math, and matrix operators."""
    # linalg.inv / linalg_inv: float & complex only, rank >= 2
    dtypes, rank = infer_torch_dtype_and_rank("linalg_inv", "A", "Tensor")
    assert dtypes == ["float32", "float64", "complex64", "complex128"]
    assert rank == ">=2"

    # cholesky: float & complex only, rank >= 2
    dtypes_chol, rank_chol = infer_torch_dtype_and_rank("cholesky", "input", "Tensor")
    assert dtypes_chol == ["float32", "float64", "complex64", "complex128"]
    assert rank_chol == ">=2"

    _, rank_self = infer_torch_dtype_and_rank("cholesky", "self", "Tensor")
    assert rank_self == ">=2"

    _, rank_a = infer_torch_dtype_and_rank("linalg_inv", "a", "Tensor")
    assert rank_a == ">=2"

    # mm: 2D matrix multiplication
    dtypes_mm, rank_mm = infer_torch_dtype_and_rank("mm", "mat2", "Tensor")
    assert rank_mm == 2

    # bmm: 3D batch matrix multiplication
    dtypes_bmm, rank_bmm = infer_torch_dtype_and_rank("bmm", "mat2", "Tensor")
    assert rank_bmm == 3

    # mv: matrix-vector
    _, rank_mat = infer_torch_dtype_and_rank("mv", "mat", "Tensor")
    _, rank_vec = infer_torch_dtype_and_rank("mv", "vec", "Tensor")
    assert rank_mat == 2
    assert rank_vec == 1

    # dot / vdot: 1D vectors
    _, rank_dot = infer_torch_dtype_and_rank("dot", "input", "Tensor")
    assert rank_dot == 1

    # Floating point ops (sin, cos, exp)
    dtypes_sin, _ = infer_torch_dtype_and_rank("sin", "input", "Tensor")
    assert dtypes_sin == ["float16", "bfloat16", "float32", "float64"]

    # Explicit float in type
    dtypes_flt, _ = infer_torch_dtype_and_rank("custom_op", "x", "float")
    assert dtypes_flt == ["float16", "bfloat16", "float32", "float64"]

    # Explicit int in type
    dtypes_int, _ = infer_torch_dtype_and_rank("custom_op", "n", "int")
    assert dtypes_int == ["int8", "int16", "int32", "int64"]

    # Unknown param on known op
    dtypes_mm_unk, rank_mm_unk = infer_torch_dtype_and_rank("mm", "unknown_param")
    assert dtypes_mm_unk is None
    assert rank_mm_unk is None

    # Unknown
    dtypes_unk, rank_unk = infer_torch_dtype_and_rank("unknown_op", "unknown_param")
    assert dtypes_unk is None
    assert rank_unk is None


def test_get_aten_op_schema_real() -> None:
    """Test get_aten_op_schema querying real PyTorch runtime ATen schemas."""
    # torch.ops.aten.add has overloads: Tensor, Scalar, out, etc.
    schemas_add = get_aten_op_schema("add")
    assert schemas_add is not None
    assert len(schemas_add) > 1
    overload_names = [ov["overload_name"] for ov in schemas_add]
    assert "Tensor" in overload_names
    assert "Scalar" in overload_names
    assert "out" in overload_names

    # Check parameter structure of Tensor overload
    tensor_ov = next(ov for ov in schemas_add if ov["overload_name"] == "Tensor")
    param_names = [p["name"] for p in tensor_ov["params"]]
    assert "input" in param_names
    assert "other" in param_names
    assert "alpha" in param_names

    # In-place operator add_
    schemas_add_ = get_aten_op_schema("add_")
    assert schemas_add_ is not None
    assert all(ov["is_inplace"] is True for ov in schemas_add_)

    # Out-variant detection
    out_ov = next(ov for ov in schemas_add if ov["overload_name"] == "out")
    assert out_ov["is_out"] is True
    assert any(p["is_out"] for p in out_ov["params"])

    # Non-existent op
    assert get_aten_op_schema("completely_fictional_op") is None


def test_extract_aten_c_extension_signature_real() -> None:
    """Test extract_aten_c_extension_signature extracting full signature and overloads."""
    import torch

    sig = extract_aten_c_extension_signature(torch.add, "torch.add")
    assert sig is not None
    assert len(sig) >= 3  # input, other, alpha
    assert sig.overloads is not None
    assert len(sig.overloads) > 1

    # In-place target
    sig_inplace = extract_aten_c_extension_signature(None, "torch.add_")
    assert sig_inplace is not None
    assert len(sig_inplace.overloads) >= 1

    # Method call target (stripping receiver 'self')
    sig_method = extract_aten_c_extension_signature(None, "torch.Tensor.add")
    assert sig_method is not None
    assert sig_method[0][0] != "self"

    # Non-existent target
    assert extract_aten_c_extension_signature(None, "torch.not_existing_at_all") is None


def test_parse_native_functions_yaml(tmp_path: Any) -> None:
    """Test parse_native_functions_yaml parsing PyTorch native function YAML definitions.

    Args:
        tmp_path: Pytest temporary directory fixture.
    """
    sample_yaml = """
- not_a_dict_with_func: true
- func: invalid_format_no_parens
- func: single_word_decl(arg) -> Tensor
- func: add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
  variants: function, method
  dispatch:
    CPU: add_cpu
    CUDA: add_cuda

- func: add.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor
  variants: function, method

- func: add.out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
  variants: function

- func: relu(Tensor self) -> Tensor
  variants: function, method

- func: relu_(Tensor(a!) self) -> Tensor(a!)
  variants: function, method
"""
    # Test reading from actual file on disk
    yaml_file = tmp_path / "native_functions.yaml"
    yaml_file.write_text(sample_yaml, encoding="utf-8")
    parsed = parse_native_functions_yaml(str(yaml_file))
    assert "add" in parsed
    assert "relu" in parsed
    assert "relu_" in parsed
    assert "single_word_decl" in parsed

    # 3 overloads for add
    add_overloads = parsed["add"]
    assert len(add_overloads) == 3
    ov_names = [ov["overload_name"] for ov in add_overloads]
    assert "Tensor" in ov_names
    assert "Scalar" in ov_names
    assert "out" in ov_names

    # Check Tensor overload parameters
    tensor_ov = next(ov for ov in add_overloads if ov["overload_name"] == "Tensor")
    assert tensor_ov["params"][0]["name"] == "input"
    assert tensor_ov["params"][1]["name"] == "other"
    assert tensor_ov["params"][2]["name"] == "alpha"
    assert tensor_ov["params"][2]["kind"] == "KEYWORD_ONLY"
    assert tensor_ov["params"][2]["default"] == "1"

    # Check out overload
    out_ov = next(ov for ov in add_overloads if ov["overload_name"] == "out")
    assert out_ov["is_out"] is True
    assert any(p["name"] == "out" and p["is_out"] for p in out_ov["params"])

    # Check in-place relu_
    relu_inplace = parsed["relu_"][0]
    assert relu_inplace["is_inplace"] is True

    # Empty or invalid YAML
    assert parse_native_functions_yaml("") == {}
    assert parse_native_functions_yaml("not a list") == {}


def test_ghost_inspector_aten_torch_no_decay() -> None:
    """Test that GhostInspector inspects PyTorch ATen operators without decaying to (*args, **kwargs)."""
    import torch

    # Inspect torch.add
    ref_add = GhostInspector.inspect(torch.add, "torch.add")
    assert ref_add.name == "add"
    assert not (
        len(ref_add.params) == 2
        and ref_add.params[0].name == "args"
        and ref_add.params[1].name == "kwargs"
    )
    param_names = [p.name for p in ref_add.params]
    assert "input" in param_names
    assert "other" in param_names
    # Verify overloads populated
    assert len(ref_add.overloads) > 0

    # Inspect torch.relu
    ref_relu = GhostInspector.inspect(torch.relu, "torch.relu")
    assert ref_relu.name == "relu"
    assert not (
        len(ref_relu.params) == 2
        and ref_relu.params[0].name == "args"
        and ref_relu.params[1].name == "kwargs"
    )
    assert any(p.name == "input" for p in ref_relu.params)

    # Inspect torch.matmul
    ref_matmul = GhostInspector.inspect(torch.matmul, "torch.matmul")
    assert ref_matmul.name == "matmul"
    assert not (
        len(ref_matmul.params) == 2
        and ref_matmul.params[0].name == "args"
        and ref_matmul.params[1].name == "kwargs"
    )
    assert any(p.name == "input" for p in ref_matmul.params)
    assert any(p.name == "other" for p in ref_matmul.params)


def test_check_hallucination_dtype_and_rank_validation() -> None:
    """Test check_hallucination flagging invalid tensor dtypes and wrong ranks."""
    # Mock torch snapshot with linalg.inv and mm
    mock_snap = {
        "categories": {
            "linalg": [
                {
                    "name": "inv",
                    "api_path": "torch.linalg.inv",
                    "params": [
                        {
                            "name": "A",
                            "kind": "POSITIONAL_OR_KEYWORD",
                            "dtypes": [
                                "float32",
                                "float64",
                                "complex64",
                                "complex128",
                            ],
                            "rank": ">=2",
                        }
                    ],
                },
                {
                    "name": "cholesky",
                    "api_path": "torch.cholesky",
                    "params": [
                        {
                            "name": "input",
                            "kind": "POSITIONAL_OR_KEYWORD",
                            "dtypes": [
                                "float32",
                                "float64",
                                "complex64",
                                "complex128",
                            ],
                            "rank": ">=2",
                        },
                        {
                            "name": "upper",
                            "kind": "POSITIONAL_OR_KEYWORD",
                            "default": "False",
                        },
                    ],
                },
                {
                    "name": "mm",
                    "api_path": "torch.mm",
                    "params": [
                        {
                            "name": "input",
                            "kind": "POSITIONAL_OR_KEYWORD",
                            "rank": 2,
                        },
                        {
                            "name": "mat2",
                            "kind": "POSITIONAL_OR_KEYWORD",
                            "rank": 2,
                        },
                    ],
                },
            ]
        }
    }

    from unittest.mock import patch

    with (
        patch(
            "ml_framework_snapshots.mcp_server.get_framework_snapshot",
            return_value=mock_snap,
        ),
        patch(
            "ml_framework_snapshots.index.lookup_symbol",
            return_value=None,
        ),
    ):
        # 1. Flag int32 passed to torch.linalg.inv
        res_inv_bad = check_hallucination(
            "torch",
            "torch.linalg.inv",
            arg_dtypes=["int32"],
        )
        assert res_inv_bad["is_hallucinated"] is True
        assert "not supported" in res_inv_bad["reason"]
        assert "int32" in res_inv_bad["reason"]

        # Valid float32 passed to torch.linalg.inv
        res_inv_ok = check_hallucination(
            "torch",
            "torch.linalg.inv",
            arg_dtypes=["float32"],
        )
        assert res_inv_ok["is_hallucinated"] is False

        # 2. Flag int32 passed to torch.cholesky
        res_chol_bad = check_hallucination(
            "torch",
            "torch.cholesky",
            kwarg_dtypes={"input": "int32"},
        )
        assert res_chol_bad["is_hallucinated"] is True
        assert "not supported" in res_chol_bad["reason"]

        # 3. Flag rank mismatch for torch.mm (e.g. 1D vector instead of 2D matrix)
        res_mm_bad = check_hallucination(
            "torch",
            "torch.mm",
            arg_ranks=[1, 2],
        )
        assert res_mm_bad["is_hallucinated"] is True
        assert "Rank 1 is not supported" in res_mm_bad["reason"]

        # Valid rank 2 for both inputs in torch.mm
        res_mm_ok = check_hallucination(
            "torch",
            "torch.mm",
            arg_ranks=[2, 2],
        )
        assert res_mm_ok["is_hallucinated"] is False

        # Rank >= 2 check for torch.linalg.inv
        res_inv_rank_bad = check_hallucination(
            "torch",
            "torch.linalg.inv",
            arg_ranks=[1],
        )
        assert res_inv_rank_bad["is_hallucinated"] is True
        assert "Rank 1 is not supported" in res_inv_rank_bad["reason"]


def test_handle_mcp_message_dtypes_and_ranks() -> None:
    """Test handle_mcp_message dispatching arg_dtypes and arg_ranks."""
    mock_snap = {
        "categories": {
            "linalg": [
                {
                    "name": "inv",
                    "api_path": "torch.linalg.inv",
                    "params": [
                        {
                            "name": "A",
                            "kind": "POSITIONAL_OR_KEYWORD",
                            "dtypes": ["float32", "float64"],
                        }
                    ],
                }
            ]
        }
    }

    from unittest.mock import patch

    with patch(
        "ml_framework_snapshots.mcp_server.get_framework_snapshot",
        return_value=mock_snap,
    ):
        msg = {
            "jsonrpc": "2.0",
            "id": 42,
            "method": "tools/call",
            "params": {
                "name": "check_hallucination",
                "arguments": {
                    "framework": "torch",
                    "api_path": "torch.linalg.inv",
                    "arg_dtypes": ["int32"],
                },
            },
        }
        resp = handle_mcp_message(msg)
        assert "result" in resp
        assert '"is_hallucinated": true' in resp["result"]["content"][0]["text"]
        assert "not supported" in resp["result"]["content"][0]["text"]


def test_aten_edge_cases_and_mocks(mocker: Any) -> None:
    """Test edge branches of get_aten_op_schema and extract_aten_c_extension_signature with mocks.

    Args:
        mocker: Pytest mocker fixture.
    """
    import torch

    # 1. Test get_aten_op_schema when schema is None on an overload
    class DummyOp:
        """Dummy operator."""

        def overloads(self) -> List[str]:
            """List overloads.

            Returns:
                List of overload names.
            """
            return ["ov1"]

        ov1 = type("DummyOverload", (), {"_schema": None})()

    mocker.patch.object(torch.ops.aten, "dummy_no_schema", DummyOp(), create=True)
    assert get_aten_op_schema("dummy_no_schema") is None

    # 2. Test extract_aten_c_extension_signature when overload_name is custom (not Tensor or default)
    mock_schemas = [
        {
            "overload_name": "custom_name",
            "params": [
                {
                    "name": "x",
                    "kind": "POSITIONAL_OR_KEYWORD",
                    "default": None,
                    "annotation": "Tensor",
                }
            ],
            "returns_type": "Tensor",
        },
        {
            "overload_name": "other_variant",
            "params": [
                {
                    "name": "y",
                    "kind": "POSITIONAL_OR_KEYWORD",
                    "default": None,
                    "annotation": "Tensor",
                }
            ],
            "returns_type": "Tensor",
        },
    ]
    mocker.patch(
        "ml_framework_snapshots.frameworks.torch.get_aten_op_schema",
        return_value=mock_schemas,
    )
    sig = extract_aten_c_extension_signature(None, "custom_op")
    assert sig is not None
    assert sig[0][0] == "x"
    assert len(sig.overloads) == 1
    assert sig.overloads[0][0][0] == "y"
