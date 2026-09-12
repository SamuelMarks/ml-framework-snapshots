"""Hardware instruction grounding verifiers for NVIDIA SASS and AMD RDNA/CDNA."""

import re
from typing import List, Optional

from ml_framework_snapshots.grounding.engine import GroundingEngine
from ml_framework_snapshots.grounding.models import (
    DiagnosticSeverity,
    GroundingReport,
)


def validate_sass_instruction(
    mnemonic: str,
    architecture: str,
    operands: List[str],
    modifiers: Optional[List[str]] = None,
    engine: Optional[GroundingEngine] = None,
) -> GroundingReport:
    """Verify an NVIDIA SASS instruction against the ground-truth offline ISA catalog.

    Args:
        mnemonic: The instruction mnemonic (e.g., 'FFMA', 'HMMA', 'WGMMA', 'LDG').
        architecture: Target SM architecture (e.g., 'sm_70', 'sm_80', 'sm_90', 'sm_100').
        operands: List of operand expressions (e.g., ['R1', 'R2', 'R3', 'R4']).
        modifiers: Optional instruction modifier flags (e.g., ['.F16', '.FTZ']).
        engine: Optional GroundingEngine instance. If omitted, a singleton is used.

    Returns:
        GroundingReport containing diagnostic results and grounding status.
    """
    eng = engine or GroundingEngine()
    clean_mnemonic = mnemonic.upper().strip()
    clean_arch = architecture.lower().strip()
    mods = modifiers or []

    report = GroundingReport(
        is_grounded=True,
        target="nvidia_sass",
        symbol=clean_mnemonic,
    )

    ref = eng.get_symbol("nvidia_sass", clean_mnemonic)
    if not ref:
        suggested = eng.suggest_closest_symbol("nvidia_sass", clean_mnemonic)
        report.add_diagnostic(
            field="mnemonic",
            message=f"Unrecognized SASS instruction mnemonic '{clean_mnemonic}'.",
            severity=DiagnosticSeverity.ERROR,
            suggested_fix=suggested,
        )
        return report

    report.matched_ref = ref

    # Verify architecture support
    valid_archs = ref.environment_tags or []
    if clean_arch not in valid_archs and valid_archs:
        report.add_diagnostic(
            field="architecture",
            message=(
                f"Instruction '{clean_mnemonic}' is not supported on target architecture '{clean_arch}'. "
                f"Supported architectures: {', '.join(valid_archs)}"
            ),
            severity=DiagnosticSeverity.ERROR,
            suggested_fix=valid_archs[-1] if valid_archs else None,
        )

    # Verify modifier legality
    domain_meta = ref.domain_metadata or {}
    allowed_modifiers = domain_meta.get("modifiers") or []
    for mod in mods:
        if allowed_modifiers and mod not in allowed_modifiers:
            report.add_diagnostic(
                field="modifiers",
                message=f"Modifier '{mod}' is not recognized for '{clean_mnemonic}'.",
                severity=DiagnosticSeverity.WARNING,
            )

    # Verify operand counts
    allowed_sigs = domain_meta.get("operand_signatures") or []
    if allowed_sigs:
        matching_count_sig = [sig for sig in allowed_sigs if len(sig) == len(operands)]
        if not matching_count_sig:
            allowed_counts = sorted(list({len(sig) for sig in allowed_sigs}))
            report.add_diagnostic(
                field="operands",
                message=(
                    f"Instruction '{clean_mnemonic}' expects {allowed_counts} operands, "
                    f"but received {len(operands)}: {operands}."
                ),
                severity=DiagnosticSeverity.ERROR,
            )

    return report


def validate_rdna_instruction(
    mnemonic: str,
    gfx_arch: str,
    operands: List[str],
    modifiers: Optional[List[str]] = None,
    engine: Optional[GroundingEngine] = None,
) -> GroundingReport:
    """Verify an AMD RDNA / CDNA instruction against the ground-truth offline ISA catalog.

    Args:
        mnemonic: The instruction mnemonic (e.g., 'V_ADD_F32', 'V_DUAL_FMAC_F32').
        gfx_arch: Target GFX generation (e.g., 'GFX9', 'GFX10', 'GFX11', 'GFX12').
        operands: List of operand expressions (e.g., ['v0', 'v1', 's0']).
        modifiers: Optional modifier flags (e.g., ['clamp', 'omod']).
        engine: Optional GroundingEngine instance.

    Returns:
        GroundingReport containing diagnostic results and grounding status.
    """
    eng = engine or GroundingEngine()
    clean_mnemonic = mnemonic.upper().strip()

    report = GroundingReport(
        is_grounded=True,
        target="amd_rdna",
        symbol=clean_mnemonic,
    )

    ref = eng.get_symbol("amd_rdna", clean_mnemonic)
    if not ref:
        suggested = eng.suggest_closest_symbol("amd_rdna", clean_mnemonic)
        report.add_diagnostic(
            field="mnemonic",
            message=f"Unrecognized AMD RDNA instruction mnemonic '{clean_mnemonic}'.",
            severity=DiagnosticSeverity.ERROR,
            suggested_fix=suggested,
        )
        return report

    report.matched_ref = ref

    # Check 64-bit even-index register alignment: v[n:n+1] requires n to be even
    even_reg_pattern = re.compile(r"^[vs]\[(\d+):(\d+)\]$", re.IGNORECASE)
    for idx, op in enumerate(operands):
        match = even_reg_pattern.match(op.strip())
        if match:
            start_reg = int(match.group(1))
            end_reg = int(match.group(2))
            if end_reg - start_reg == 1 and (start_reg % 2 != 0):
                report.add_diagnostic(
                    field=f"operands[{idx}]",
                    message=(
                        f"64-bit register pair '{op}' must be even-aligned. "
                        f"Register index {start_reg} is odd."
                    ),
                    severity=DiagnosticSeverity.ERROR,
                    suggested_fix=f"v[{start_reg - 1}:{start_reg}]",
                )

    return report
