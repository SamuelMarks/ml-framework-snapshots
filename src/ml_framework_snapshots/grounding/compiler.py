"""Compiler IR grounding verifiers for StableHLO and MLIR core dialects."""

from typing import Any, Dict, List, Optional

from ml_framework_snapshots.grounding.engine import GroundingEngine
from ml_framework_snapshots.grounding.models import (
    DiagnosticSeverity,
    GroundingReport,
)


def validate_stablehlo_op(
    op_name: str,
    operand_types: List[str],
    attributes: Dict[str, Any],
    engine: Optional[GroundingEngine] = None,
) -> GroundingReport:
    """Verify a StableHLO operation against the official StableHLO specification.

    Args:
        op_name: Operation name (e.g., 'stablehlo.dot_general', 'dot_general', 'stablehlo.add').
        operand_types: Types of SSA input operands (e.g., ['tensor<128x64xf32>', 'tensor<64x256xf32>']).
        attributes: Dictionary of operation attributes (e.g., {'dot_dimension_numbers': {...}}).
        engine: Optional GroundingEngine instance.

    Returns:
        GroundingReport containing diagnostic results and grounding status.
    """
    eng = engine or GroundingEngine()
    full_op_name = (
        op_name if op_name.startswith("stablehlo.") else f"stablehlo.{op_name}"
    )

    report = GroundingReport(
        is_grounded=True,
        target="stablehlo",
        symbol=full_op_name,
    )

    ref = eng.get_symbol("stablehlo", full_op_name)
    if not ref:
        suggested = eng.suggest_closest_symbol("stablehlo", full_op_name)
        report.add_diagnostic(
            field="operation",
            message=f"Unrecognized StableHLO operation '{full_op_name}'.",
            severity=DiagnosticSeverity.ERROR,
            suggested_fix=suggested,
        )
        return report

    report.matched_ref = ref

    # Verify mandatory operation attributes
    if full_op_name == "stablehlo.dot_general":
        if (
            "dot_dimension_numbers" not in attributes
            and "dimension_numbers" not in attributes
        ):
            report.add_diagnostic(
                field="attributes.dot_dimension_numbers",
                message="Operation 'stablehlo.dot_general' requires 'dot_dimension_numbers' attribute.",
                severity=DiagnosticSeverity.ERROR,
                suggested_fix="dot_dimension_numbers",
            )

    if full_op_name == "stablehlo.convolution":
        if "dimension_numbers" not in attributes:
            report.add_diagnostic(
                field="attributes.dimension_numbers",
                message="Operation 'stablehlo.convolution' requires 'dimension_numbers' attribute.",
                severity=DiagnosticSeverity.ERROR,
                suggested_fix="dimension_numbers",
            )

    # Verify operand counts
    raw_operands = getattr(ref, "operands", None)
    expected_operands = (
        raw_operands
        if raw_operands is not None
        else [p for p in ref.params if getattr(p, "role", None) != "ATTRIBUTE"]
    )
    if expected_operands:
        # Check if variadic
        is_variadic = getattr(ref, "has_varargs", False)
        if not is_variadic and len(operand_types) != len(expected_operands):
            report.add_diagnostic(
                field="operands",
                message=(
                    f"Operation '{full_op_name}' expects {len(expected_operands)} operands, "
                    f"but received {len(operand_types)}."
                ),
                severity=DiagnosticSeverity.ERROR,
            )

    return report


def validate_mlir_op(
    dialect: str,
    op_name: str,
    operand_count: int,
    attributes: Dict[str, Any],
    engine: Optional[GroundingEngine] = None,
) -> GroundingReport:
    """Verify an MLIR operation against core MLIR TableGen definitions.

    Args:
        dialect: The MLIR dialect (e.g., 'arith', 'math', 'tensor', 'linalg', 'scf').
        op_name: The operation name or mnemonic (e.g., 'addf', 'arith.addf', 'matmul').
        operand_count: Number of SSA value operands passed to the operation.
        attributes: Dictionary of buildable MLIR attributes.
        engine: Optional GroundingEngine instance.

    Returns:
        GroundingReport containing diagnostic results and grounding status.
    """
    eng = engine or GroundingEngine()
    full_path = op_name if op_name.startswith(f"{dialect}.") else f"{dialect}.{op_name}"

    report = GroundingReport(
        is_grounded=True,
        target="mlir",
        symbol=full_path,
    )

    ref = eng.get_symbol("mlir", full_path)
    if not ref:
        suggested = eng.suggest_closest_symbol("mlir", full_path)
        report.add_diagnostic(
            field="operation",
            message=f"Unrecognized MLIR operation '{full_path}'.",
            severity=DiagnosticSeverity.ERROR,
            suggested_fix=suggested,
        )
        return report

    report.matched_ref = ref

    # Verify operands count
    raw_operands = getattr(ref, "operands", None)
    expected_operands = (
        raw_operands
        if raw_operands is not None
        else [p for p in ref.params if getattr(p, "role", None) != "ATTRIBUTE"]
    )
    if expected_operands and operand_count != len(expected_operands):
        report.add_diagnostic(
            field="operand_count",
            message=(
                f"MLIR operation '{full_path}' expects {len(expected_operands)} operands, "
                f"but received {operand_count}."
            ),
            severity=DiagnosticSeverity.ERROR,
        )

    return report
