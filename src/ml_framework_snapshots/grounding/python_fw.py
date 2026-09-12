"""Python ML framework signature and argument grounding verifier."""

from typing import Any, Dict, List, Optional

from ml_framework_snapshots.grounding.engine import GroundingEngine
from ml_framework_snapshots.grounding.models import (
    DiagnosticSeverity,
    GroundingReport,
)


def validate_python_call(
    framework: str,
    api_path: str,
    args: List[Any],
    kwargs: Dict[str, Any],
    engine: Optional[GroundingEngine] = None,
) -> GroundingReport:
    """Verify a Python framework API call against the ground-truth offline snapshot.

    Args:
        framework: Framework name (e.g., 'torch', 'jax', 'numpy', 'tensorflow').
        api_path: Fully qualified API symbol path (e.g., 'torch.sum', 'jax.numpy.mean').
        args: List of positional arguments provided to the call.
        kwargs: Dictionary of keyword arguments provided to the call.
        engine: Optional GroundingEngine instance.

    Returns:
        GroundingReport containing validation status, diagnostics, and suggested fixes.
    """
    eng = engine or GroundingEngine()
    clean_fw = framework.lower().strip()

    report = GroundingReport(
        is_grounded=True,
        target=clean_fw,
        symbol=api_path,
    )

    ref = eng.get_symbol(clean_fw, api_path)
    if not ref:
        suggested = eng.suggest_closest_symbol(clean_fw, api_path)
        report.add_diagnostic(
            field="api_path",
            message=f"API '{api_path}' was not found in ground-truth snapshot for '{clean_fw}'.",
            severity=DiagnosticSeverity.ERROR,
            suggested_fix=suggested,
        )
        return report

    report.matched_ref = ref

    # Build known parameter sets
    param_names = {p.name for p in ref.params if p.name}
    has_var_kwargs = (
        getattr(ref, "has_varargs", False)
        or any("VAR_KEYWORD" in str(getattr(p, "kind", "")) for p in ref.params)
        or bool(getattr(ref, "accepted_kwargs", None))
    )

    # Check for hallucinated kwargs
    for kw in kwargs:
        if kw not in param_names and not has_var_kwargs:
            # Common cross-framework hallucinations
            suggested_kw: Optional[str] = None
            if clean_fw == "torch" and kw == "axis" and "dim" in param_names:
                suggested_kw = "dim"
            elif clean_fw in ("numpy", "jax") and kw == "dim" and "axis" in param_names:
                suggested_kw = "axis"

            report.add_diagnostic(
                field=f"kwargs.{kw}",
                message=f"Unexpected or hallucinated keyword argument '{kw}' for '{api_path}'.",
                severity=DiagnosticSeverity.ERROR,
                suggested_fix=suggested_kw,
            )

    # Check positional argument count
    pos_params = [
        p
        for p in ref.params
        if "KEYWORD_ONLY" not in str(getattr(p, "kind", ""))
        and "VAR_POSITIONAL" not in str(getattr(p, "kind", ""))
    ]
    has_var_args = getattr(ref, "has_varargs", False) or any(
        "VAR_POSITIONAL" in str(getattr(p, "kind", "")) for p in ref.params
    )

    if not has_var_args and len(args) > len(pos_params) and pos_params:
        report.add_diagnostic(
            field="args",
            message=(
                f"Too many positional arguments passed to '{api_path}'. "
                f"Expected at most {len(pos_params)}, but received {len(args)}."
            ),
            severity=DiagnosticSeverity.ERROR,
        )

    return report
