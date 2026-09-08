"""Tests for build_stablehlo_snapshot.py."""

from unittest import mock

from ml_framework_snapshots.tools import build_stablehlo_snapshot


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


@mock.patch("ml_framework_snapshots.tools.build_stablehlo_snapshot.extract_ops")
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
    assert mock_open.call_count == 2

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


def test_parse_stablehlo_tablegen() -> None:
    """Test parsing StableHLO TableGen ODS definitions into structured operations."""
    sample_tablegen = """
def StableHLO_AddOp : StableHLO_ElementwiseBinaryOp<"add", [Commutative, HLO_BroadcastingElementwise, SameOperandsAndResultType]> {
  let summary = "Addition operation";
  let arguments = (ins
    HLO_Tensor:$lhs,
    HLO_Tensor:$rhs
  );
  let results = (outs
    HLO_Tensor:$result
  );
}

def StableHLO_ReduceOp : StableHLO_Op<"reduce", [SameOperandsAndResultElementType]> {
  let summary = "Reduce operation";
  let arguments = (ins
    Variadic<HLO_Tensor>:$inputs,
    Variadic<HLO_Tensor>:$init_values,
    I64ElementsAttr:$dimensions
  );
  let results = (outs
    Variadic<HLO_Tensor>:$results
  );
  let regions = (region
    SizedRegion<1>:$body
  );
}

def StableHLO_WhileOp : StableHLO_Op<"while", []> {
  let summary = "While operation";
  let arguments = (ins
    Variadic<HLO_TensorOrToken>:$operand
  );
  let results = (outs
    Variadic<HLO_TensorOrToken>:$res1,
    Variadic<HLO_TensorOrToken>:$res2
  );
  let regions = (region
    SizedRegion<1>:$cond,
    SizedRegion<1>:$body
  );
}
"""
    refs = build_stablehlo_snapshot.parse_stablehlo_tablegen(sample_tablegen)
    assert len(refs) == 3

    add_op = next(r for r in refs if r["name"] == "add")
    assert add_op["class_name"] == "AddOp"
    assert "Commutative" in add_op["traits"]
    assert len(add_op["params"]) == 2
    assert add_op["params"][0]["name"] == "lhs"
    assert add_op["params"][0]["kind"] == "POSITIONAL_OR_KEYWORD"
    assert add_op["returns_type"] == "HLO_Tensor"

    reduce_op = next(r for r in refs if r["name"] == "reduce")
    assert reduce_op["class_name"] == "ReduceOp"
    dim_attr = next(p for p in reduce_op["params"] if p["name"] == "dimensions")
    assert dim_attr["kind"] == "KEYWORD_ONLY"
    assert "body" in reduce_op["regions"]

    while_op = next(r for r in refs if r["name"] == "while")
    assert "cond" in while_op["regions"]
    assert "body" in while_op["regions"]
    assert (
        while_op["returns_type"]
        == "tuple[Variadic<HLO_TensorOrToken>, Variadic<HLO_TensorOrToken>]"
    )

    # Test TableGen edge cases: colon without dollar, single unnamed region, empty results/args, trailing commas
    edge_tablegen = """
def StableHLO_EdgeOp : StableHLO_Op<"edge"> {
  let arguments = (ins HLO_Tensor:in_no_dollar, HLO_Tensor, );
  let regions = (region single_region, );
  let results = (outs);
}

def StableHLO_NoArgsOp : StableHLO_Op<"no_args"> {
  let summary = "No args op";
}
"""
    edge_refs = build_stablehlo_snapshot.parse_stablehlo_tablegen(edge_tablegen)
    assert len(edge_refs) == 2
    assert edge_refs[0]["name"] == "edge"
    assert len(edge_refs[0]["params"]) == 3  # 2 args + 1 region
    assert edge_refs[0]["returns_type"] is None
    assert edge_refs[1]["name"] == "no_args"
    assert len(edge_refs[1]["params"]) == 0
    assert edge_refs[1]["returns_type"] is None

    # Test extract_ops with direct TableGen string
    tablegen_refs = build_stablehlo_snapshot.extract_ops(edge_tablegen)
    assert len(tablegen_refs) == 2


def test_parse_stablehlo_tablegen_comprehensive_coverage() -> None:
    """Test TableGen parsing base classes, inheritance recursion, mnemonics, traits, fallbacks, and delimiters."""
    tablegen_source = """
    // Base class with let arguments and let results in body
    class StableHLO_BaseWithBody<string mnem> : SuperClass {
      let arguments = (ins HLO_Tensor:$x);
      let results = (outs HLO_Tensor:$y);
      let nested_block = { inner { deep } };
    }

    // Base class with Arguments and Results in supers
    class StableHLO_BaseWithSupers<string mnem> : Arguments<(ins HLO_Tensor:$in)>, Results<(outs HLO_Tensor:$out)> {
      let summary = "Super base";
    }

    // Class terminated with semicolon (no body) inheriting from BaseWithSupers
    class StableHLO_IntermediateBase : StableHLO_BaseWithSupers;

    // Def inheriting recursively via IntermediateBase -> BaseWithSupers
    def StableHLO_InheritedOp : StableHLO_IntermediateBase<"inherited"> {
      let summary = "Inherited op";
    }

    // Def with parenthesis in targs, nested braces in body, let mnemonic, and body traits
    def StableHLO_ComplexOp : StableHLO_Op<(some_targ), "ignored"> {
      let mnemonic = "complex_op";
      let traits = [Commutative, Pure];
      let nested_block = { inner { deep } };
      let arguments = (ins HLO_Tensor:$lhs, , HLO_Tensor:$rhs);
      let results = (outs HLO_Tensor:$r1, , HLO_Tensor:$r2, );
    }

    // Def with unquoted targs (falls back to camel-case class name)
    def StableHLO_UnquotedTargsOp : StableHLO_Op<12345> {
      let summary = "Unquoted targs";
    }

    // Def terminated with semicolon (empty targs and empty body)
    def StableHLO_SemicolonDefOp : StableHLO_Op;

    // Fallback args and results: Binary
    def StableHLO_BinaryFallbackOp : ElementwiseBinary {
      let summary = "Binary fallback";
    }

    // Fallback args and results: Unary
    def StableHLO_UnaryFallbackOp : ElementwiseUnary {
      let summary = "Unary fallback";
    }

    // Fallback args and results: Cast
    def StableHLO_CastFallbackOp : CastOpBase {
      let summary = "Cast fallback";
    }
    """
    refs = build_stablehlo_snapshot.parse_stablehlo_tablegen(tablegen_source)
    assert len(refs) == 7

    # 1. InheritedOp
    inherited = next(r for r in refs if r["name"] == "inherited")
    assert len(inherited["params"]) == 1
    assert inherited["params"][0]["name"] == "in"
    assert inherited["returns_type"] == "HLO_Tensor"

    # 2. ComplexOp (explicit mnemonic, body traits, empty token comma skips, multi-result tuple)
    complex_op = next(r for r in refs if r["name"] == "complex_op")
    assert "Commutative" in complex_op["traits"]
    assert "Pure" in complex_op["traits"]
    assert len(complex_op["params"]) == 2
    assert complex_op["returns_type"] == "tuple[HLO_Tensor, HLO_Tensor]"

    # 3. UnquotedTargsOp (camel case converted to unquoted_targs)
    unquoted = next(r for r in refs if r["name"] == "unquoted_targs")
    assert unquoted["class_name"] == "UnquotedTargsOp"

    # 4. SemicolonDefOp (camel case converted to semicolon_def)
    semi = next(r for r in refs if r["name"] == "semicolon_def")
    assert semi["class_name"] == "SemicolonDefOp"

    # 5. Binary fallback
    bin_op = next(r for r in refs if r["name"] == "binary_fallback")
    assert len(bin_op["params"]) == 2
    assert bin_op["params"][0]["name"] == "lhs"
    assert bin_op["returns_type"] == "HLO_Tensor"

    # 6. Unary fallback
    un_op = next(r for r in refs if r["name"] == "unary_fallback")
    assert len(un_op["params"]) == 1
    assert un_op["params"][0]["name"] == "operand"
    assert un_op["returns_type"] == "HLO_Tensor"

    # 7. Cast fallback
    cast_op = next(r for r in refs if r["name"] == "cast_fallback")
    assert len(cast_op["params"]) == 1
    assert cast_op["params"][0]["name"] == "operand"
    assert cast_op["returns_type"] == "HLO_Tensor"


def test_parse_stablehlo_tablegen_supers_loop_and_empty_results() -> None:
    """Test superclasses returning None in resolve_field loop, empty res_types, and unterminated def."""
    tablegen_source = """
    class SuperEmpty1;
    class SuperEmpty2;
    class BaseNoFields : SuperEmpty1, SuperEmpty2;

    def StableHLO_NoFieldsOp : BaseNoFields<"no_fields"> {
      let results = (outs , );
    }

    def StableHLO_UnterminatedOp : BaseNoFields"""

    refs = build_stablehlo_snapshot.parse_stablehlo_tablegen(tablegen_source)
    assert len(refs) == 2
    assert refs[0]["name"] == "no_fields"
    assert refs[0]["returns_type"] is None


def test_parse_stablehlo_tablegen_verification_rules() -> None:
    """Test TableGen extraction of formal verification rules and region signatures."""
    tablegen_source = """
    class StableHLO_Op<string mnem> {
      let mnemonic = mnem;
    }

    def StableHLO_DotGeneralOp : StableHLO_Op<"dot_general">;
    def StableHLO_ConvolutionOp : StableHLO_Op<"convolution">;
    def StableHLO_ScatterOp : StableHLO_Op<"scatter">;
    def StableHLO_GatherOp : StableHLO_Op<"gather">;
    def StableHLO_BroadcastInDimOp : StableHLO_Op<"broadcast_in_dim">;
    def StableHLO_SortOp : StableHLO_Op<"sort">;
    """

    refs = build_stablehlo_snapshot.parse_stablehlo_tablegen(tablegen_source)
    ref_map = {r["name"]: r for r in refs}

    assert (
        ref_map["dot_general"]["verification_rules"]["dimension_numbers"]
        == "DotDimensionNumbersAttr"
    )
    assert (
        ref_map["convolution"]["verification_rules"]["dimension_numbers"]
        == "ConvDimensionNumbersAttr"
    )
    assert (
        ref_map["scatter"]["verification_rules"]["dimension_numbers"]
        == "ScatterDimensionNumbersAttr"
    )
    assert (
        ref_map["gather"]["verification_rules"]["dimension_numbers"]
        == "GatherDimensionNumbersAttr"
    )
    assert (
        ref_map["broadcast_in_dim"]["verification_rules"]["broadcast"]
        == "broadcast_dimensions"
    )
    assert "comparator" in ref_map["sort"]["region_signatures"]


def test_parse_stablehlo_llvm_tblgen_json() -> None:
    """Test parse_stablehlo_llvm_tblgen_json parsing DAG and list records."""
    mock_json = {
        "!instanceof": {
            "Op": ["StableHLO_DotGeneralOp", "StableHLO_ReduceOp", "StableHLO_WhileOp"]
        },
        "StableHLO_DotGeneralOp": {
            "!name": "StableHLO_DotGeneralOp",
            "!superclasses": ["StableHLO_Op", "Op"],
            "mnemonic": "dot_general",
            "summary": "Dot general operation.",
            "arguments": {
                "kind": "dag",
                "args": [
                    [{"def": "HLO_Tensor"}, "lhs"],
                    [{"def": "HLO_Tensor"}, "rhs"],
                    [{"def": "DotDimensionNumbersAttr"}, "dot_dimension_numbers"],
                ],
            },
            "results": {
                "kind": "dag",
                "args": [
                    [{"def": "HLO_Tensor"}, "result"],
                    [{"def": "HLO_Tensor"}, "extra_result"],
                ],
            },
            "traits": [{"def": "Commutative"}],
        },
        "StableHLO_ReduceOp": {
            "!name": "StableHLO_ReduceOp",
            "!superclasses": ["StableHLO_Op"],
            "mnemonic": "reduce",
            "arguments": [
                {"name": "inputs", "type": "HLO_Tensor"},
                {"name": "dims", "type": "I64ElementsAttr"},
            ],
            "results": [{"name": "out", "type": "HLO_Tensor"}],
            "regions": {
                "kind": "dag",
                "args": [[{"def": "Region"}, "body"]],
            },
            "traits": ["Elementwise"],
        },
        "StableHLO_WhileOp": {
            "!name": "StableHLO_WhileOp",
            "!superclasses": ["StableHLO_Op"],
            "mnemonic": "while",
            "arguments": [{"name": "operand", "type": "HLO_Tensor"}],
            "results": [{"name": "res", "type": "HLO_Tensor"}],
            "regions": ["cond", "body"],
        },
        "NonOpRecord": {
            "!name": "NonOpRecord",
            "!superclasses": ["Dialect"],
        },
    }

    # Dict input
    refs = build_stablehlo_snapshot.parse_stablehlo_llvm_tblgen_json(mock_json)
    assert len(refs) == 3

    dot = next(r for r in refs if r["name"] == "dot_general")
    assert dot["api_path"] == "stablehlo.dot_general"
    assert len(dot["operands"]) == 2
    assert len(dot["attributes"]) == 1
    assert dot["verification_rules"]["dimension_numbers"] == "DotDimensionNumbersAttr"
    assert dot["returns_type"] == "tuple[HLO_Tensor, HLO_Tensor]"
    assert "Commutative" in dot["traits"]

    reduce_op = next(r for r in refs if r["name"] == "reduce")
    assert reduce_op["api_path"] == "stablehlo.reduce"
    assert len(reduce_op["operands"]) == 1
    assert len(reduce_op["attributes"]) == 1
    assert reduce_op["returns_type"] == "HLO_Tensor"
    assert "body" in reduce_op["region_signatures"]
    assert "Elementwise" in reduce_op["traits"]

    while_op = next(r for r in refs if r["name"] == "while")
    assert "cond" in while_op["region_signatures"]
    assert "body" in while_op["region_signatures"]

    # String input
    import json

    refs_str = build_stablehlo_snapshot.parse_stablehlo_llvm_tblgen_json(
        json.dumps(mock_json)
    )
    assert len(refs_str) == 3

    # Test extract_ops with dict and string JSON
    assert len(build_stablehlo_snapshot.extract_ops(mock_json)) == 3
    assert len(build_stablehlo_snapshot.extract_ops(json.dumps(mock_json))) == 3


def test_parse_stablehlo_llvm_tblgen_json_all_rules() -> None:
    """Test parse_stablehlo_llvm_tblgen_json with all verification rules, list args/results, and nameless records."""
    assert (
        build_stablehlo_snapshot.strip_tablegen_comments(
            "/* block */ int x = 1; // comment\n"
        ).strip()
        == "int x = 1;"
    )

    mock_json = {
        "!meta": "some_metadata",
        "non_dict_key": 42,
        "StableHLO_ConvolutionOp": {
            "!name": "StableHLO_ConvolutionOp",
            "!superclasses": ["StableHLO_Op"],
            "mnemonic": "convolution",
            "arguments": [
                {"name": "lhs", "type": "HLO_Tensor"},
                "str_operand",
            ],
            "results": [
                {"name": "res", "type": "HLO_Tensor"},
                "str_result",
            ],
            "regions": ["reg0"],
            "traits": [{"def": None}, {"def": "SomeTrait"}],
        },
        "StableHLO_ScatterOp": {
            "!name": "StableHLO_ScatterOp",
            "!superclasses": ["StableHLO_Op"],
            "mnemonic": "scatter",
            "arguments": [],
            "results": [],
        },
        "StableHLO_GatherOp": {
            "!name": "StableHLO_GatherOp",
            "!superclasses": ["StableHLO_Op"],
            "mnemonic": "gather",
            "arguments": [],
            "results": [],
        },
        "StableHLO_BroadcastInDimOp": {
            "!name": "StableHLO_BroadcastInDimOp",
            "!superclasses": ["StableHLO_Op"],
            "mnemonic": "broadcast_in_dim",
            "arguments": [],
            "results": [],
        },
        "StableHLO_SortOp": {
            "!name": "StableHLO_SortOp",
            "!superclasses": ["StableHLO_Op"],
            "mnemonic": "sort",
            "arguments": [],
            "results": [],
            "regions": ["comparator"],
        },
        "StableHLO_CustomBare": {
            "!name": "StableHLO_CustomBare",
            "!superclasses": ["StableHLO_Op"],
            "arguments": [],
            "results": [],
        },
    }

    refs = build_stablehlo_snapshot.parse_stablehlo_llvm_tblgen_json(mock_json)
    assert len(refs) == 6

    ref_map = {r["name"]: r for r in refs}
    assert (
        ref_map["convolution"]["verification_rules"]["dimension_numbers"]
        == "ConvDimensionNumbersAttr"
    )
    assert (
        ref_map["scatter"]["verification_rules"]["dimension_numbers"]
        == "ScatterDimensionNumbersAttr"
    )
    assert (
        ref_map["gather"]["verification_rules"]["dimension_numbers"]
        == "GatherDimensionNumbersAttr"
    )
    assert (
        ref_map["broadcast_in_dim"]["verification_rules"]["broadcast"]
        == "broadcast_dimensions"
    )
    assert "comparator" in ref_map["sort"]["region_signatures"]
    assert ref_map["custom_bare"]["class_name"] == "CustomBareOp"


def test_parse_stablehlo_llvm_tblgen_json_edge_branches() -> None:
    """Test parse_stablehlo_llvm_tblgen_json with malformed dag entries, empty traits, and None values."""
    mock_json = {
        "!instanceof": {
            "Op": ["StableHLO_MalformedOp", "StableHLO_NoneOp"],
        },
        "StableHLO_MalformedOp": {
            "!name": "StableHLO_MalformedOp",
            "!superclasses": ["StableHLO_Op"],
            "mnemonic": "malformed",
            "arguments": {
                "kind": "dag",
                "args": [
                    ["only_one"],
                    "not_a_list",
                ],
            },
            "regions": {
                "kind": "dag",
                "args": [
                    ["only_one"],
                    123,
                ],
            },
            "results": {
                "kind": "dag",
                "args": [
                    ["only_one"],
                    "not_a_list",
                ],
            },
            "traits": ["", {"def": ""}],
        },
        "StableHLO_NoneOp": {
            "!name": "StableHLO_NoneOp",
            "!superclasses": ["StableHLO_Op"],
            "mnemonic": "none_op",
            "arguments": None,
            "regions": None,
            "results": None,
            "traits": None,
        },
    }

    refs = build_stablehlo_snapshot.parse_stablehlo_llvm_tblgen_json(mock_json)
    assert len(refs) == 2
    malformed = next(r for r in refs if r["name"] == "malformed")
    assert len(malformed["operands"]) == 0
    assert len(malformed["results"]) == 0
    assert len(malformed["traits"]) == 0
    none_op = next(r for r in refs if r["name"] == "none_op")
    assert len(none_op["operands"]) == 0
    assert len(none_op["results"]) == 0
