"""Tests for build_stablehlo_snapshot.py."""

from unittest import mock
import sys
from pathlib import Path

# Need to import the script to test it
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import build_stablehlo_snapshot


def test_is_attribute() -> None:
    """Test is attribute."""
    assert build_stablehlo_snapshot.is_attribute("constant")
    assert build_stablehlo_snapshot.is_attribute("enum of `DEFAULT`")
    assert build_stablehlo_snapshot.is_attribute("`si64`")
    assert build_stablehlo_snapshot.is_attribute("function")
    assert not build_stablehlo_snapshot.is_attribute("tensor")
    assert not build_stablehlo_snapshot.is_attribute("tensor or quantized tensor")


@mock.patch("urllib.request.urlopen")
def test_extract_ops(mock_urlopen: mock.MagicMock) -> None:
    """Test extract ops.

    Args:
        mock_urlopen: Mocked urlopen.
    """
    # Mock spec.md content
    mock_content = b"""
# StableHLO Specification

### abs

#### Semantics
Performs element-wise abs operation.

#### Inputs

| Label | Name      | Type                                                                                     | Constraints |
|-------|-----------|------------------------------------------------------------------------------------------|-------------|
| (I1)  | `operand` | tensor of signed integer, floating-point, or complex type or per-tensor quantized tensor | (C1-C2)     |

#### Outputs

| Name     | Type                                                                           | Constraints |
|----------|--------------------------------------------------------------------------------|-------------|
| `result` | tensor of signed integer or floating-point type or per-tensor quantized tensor | (C2)        |

### add

#### Semantics
Adds two tensors.

#### Inputs

| Label | Name | Type | Constraints |
|-------|------|------|-------------|
| (I1) | `lhs` | tensor | (C1) |
| (I2) | `rhs` | tensor | (C1) |

#### Outputs
| Name | Type | Constraints |
|------|------|-------------|
| `result` | tensor | (C1) |

### dot_general
#### Semantics
Dot general op.
#### Inputs
| Label | Name | Type | Constraints |
|-------|------|------|-------------|
| (I1) | `lhs` | tensor | (C1) |
| (I2) | `rhs` | tensor | (C1) |
| (I3) | `precision_config` | enum of `DEFAULT` | (C2) |
#### Outputs
| Name | Type | Constraints |
|------|------|-------------|
| `result` | tensor | (C1) |

"""
    mock_response = mock.MagicMock()
    mock_response.read.return_value = mock_content
    mock_urlopen.return_value = mock_response

    ghost_refs = build_stablehlo_snapshot.extract_ops()
    assert len(ghost_refs) == 3

    # Check abs
    abs_op = next(r for r in ghost_refs if r["name"] == "abs")
    assert abs_op["api_path"] == "stablehlo.abs"
    assert len(abs_op["params"]) == 1
    assert abs_op["params"][0]["name"] == "operand"
    assert "tensor of signed integer" in abs_op["params"][0]["annotation"]
    assert (
        abs_op["returns_type"]
        == "tensor of signed integer or floating-point type or per-tensor quantized tensor"
    )
    assert abs_op["docstring"] == "Performs element-wise abs operation."

    # Check add
    add_op = next(r for r in ghost_refs if r["name"] == "add")
    assert len(add_op["params"]) == 2
    assert add_op["params"][0]["name"] == "lhs"
    assert add_op["params"][1]["name"] == "rhs"
    assert add_op["returns_type"] == "tensor"

    # Check dot_general (attribute extraction)
    dot_op = next(r for r in ghost_refs if r["name"] == "dot_general")
    assert len(dot_op["params"]) == 3
    assert dot_op["params"][2]["name"] == "precision_config"
    assert dot_op["params"][2]["kind"] == "KEYWORD_ONLY"


@mock.patch("build_stablehlo_snapshot.extract_ops")
@mock.patch("builtins.open", new_callable=mock.mock_open)
def test_main(mock_open: mock.MagicMock, mock_extract: mock.MagicMock) -> None:
    """Test main.

    Args:
        mock_open: Mocked open.
        mock_extract: Mocked extract.
    """
    mock_extract.return_value = [{"name": "fake_op"}]
    build_stablehlo_snapshot.main()
    mock_extract.assert_called_once()
    mock_open.assert_called_once()

    # Assert JSON was written
    written_content = "".join(
        call.args[0] for call in mock_open.return_value.write.call_args_list
    )
    assert "stablehlo_op" in written_content
    assert "fake_op" in written_content


@mock.patch("urllib.request.urlopen")
def test_extract_ops_empty_op(mock_urlopen: mock.MagicMock) -> None:
    """Test extract ops empty op.

    Args:
        mock_urlopen: Mocked urlopen.
    """
    # Mock spec.md content with no inputs/outputs
    mock_content = b"""
# StableHLO Specification

### empty_op

#### Semantics
Empty op.
"""
    mock_response = mock.MagicMock()
    mock_response.read.return_value = mock_content
    mock_urlopen.return_value = mock_response

    ghost_refs = build_stablehlo_snapshot.extract_ops()
    assert len(ghost_refs) == 0


@mock.patch("urllib.request.urlopen")
def test_extract_ops_missing_semantics(mock_urlopen: mock.MagicMock) -> None:
    """Test extract ops missing semantics.

    Args:
        mock_urlopen: Mocked urlopen.
    """
    # Mock spec.md content with no semantics
    mock_content = b"""
# StableHLO Specification

### weird_op

#### Inputs

| Label | Name | Type | Constraints |
|-------|------|------|-------------|
| (I1) | `lhs` | tensor | (C1) |
"""
    mock_response = mock.MagicMock()
    mock_response.read.return_value = mock_content
    mock_urlopen.return_value = mock_response

    ghost_refs = build_stablehlo_snapshot.extract_ops()
    assert len(ghost_refs) == 1
    assert ghost_refs[0]["docstring"] is None


@mock.patch("urllib.request.urlopen")
def test_extract_ops_multi_output(mock_urlopen: mock.MagicMock) -> None:
    """Test extract ops multi output.

    Args:
        mock_urlopen: Mocked urlopen.
    """
    mock_content = b"""
# StableHLO Specification

### weird_op2

#### Inputs

| Name | Type |
|------|------|
| `lhs` | tensor |

#### Outputs

| Name | Type |
|------|------|
| `res1` | tensor |
| `res2` | tensor |
"""
    mock_response = mock.MagicMock()
    mock_response.read.return_value = mock_content
    mock_urlopen.return_value = mock_response

    ghost_refs = build_stablehlo_snapshot.extract_ops()
    assert len(ghost_refs) == 1
    assert ghost_refs[0]["returns_type"] == "tuple[tensor, tensor]"


@mock.patch("urllib.request.urlopen")
def test_extract_ops_ignore_unnamed_columns(mock_urlopen: mock.MagicMock) -> None:
    """Test extract ops ignore unnamed columns.

    Args:
        mock_urlopen: Mocked urlopen.
    """
    # Tests a case where parts don't meet len >= 2 or contain "---" or "Label"
    mock_content = b"""
# StableHLO Specification

### test_op

#### Inputs
| Label | Name | Type | Constraints |
|---|---|---|---|
| (I1) | `lhs` | tensor | (C1) |
| | | | |
| --- | --- | --- | --- |

#### Outputs
| Label | Name | Type | Constraints |
| (I1) | `res` | tensor | (C1) |
"""
    mock_response = mock.MagicMock()
    mock_response.read.return_value = mock_content
    mock_urlopen.return_value = mock_response

    ghost_refs = build_stablehlo_snapshot.extract_ops()
    assert len(ghost_refs) == 1
    assert ghost_refs[0]["returns_type"] in ["tensor", "", "`res`", None]


@mock.patch("urllib.request.urlopen")
def test_extract_ops_missing_type_column_outputs(mock_urlopen: mock.MagicMock) -> None:
    """Test extract ops missing type column outputs.

    Args:
        mock_urlopen: Mocked urlopen.
    """
    # Tests an output row that only has a Name
    mock_content = b"""
# StableHLO Specification

### test_op

#### Inputs
| Name | Type |
|------|------|
| `inp` | tensor |

#### Outputs
| Name |
|------|
| `res` |
"""
    mock_response = mock.MagicMock()
    mock_response.read.return_value = mock_content
    mock_urlopen.return_value = mock_response

    ghost_refs = build_stablehlo_snapshot.extract_ops()
    assert len(ghost_refs) == 1
    assert ghost_refs[0]["returns_type"] in ["tensor", "", "`res`", None]


@mock.patch("urllib.request.urlopen")
def test_extract_ops_coverage_gaps(mock_urlopen: mock.MagicMock) -> None:
    """Test extract ops coverage gaps.

    Args:
        mock_urlopen: Mocked urlopen.
    """
    # Tests a case with an invalid op name and an unknown section
    mock_content = b"""
# StableHLO Specification

### invalid-op-name

#### UnknownSection
| Name | Type |
|------|------|
| `inp` | tensor |

### test_op

#### Inputs
| Name | Type |
|------|------|
| `inp` | tensor |

#### UnknownSection
Some extra text

"""
    mock_response = mock.MagicMock()
    mock_response.read.return_value = mock_content
    mock_urlopen.return_value = mock_response

    ghost_refs = build_stablehlo_snapshot.extract_ops()
    assert len(ghost_refs) == 1
    assert ghost_refs[0]["name"] == "test_op"
