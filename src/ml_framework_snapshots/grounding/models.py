"""Data models and diagnostic report structures for the Grounding SDK."""

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field

from ml_framework_snapshots.models import ExtendedGhostRef


class DiagnosticSeverity(str, Enum):
    """Severity classification for grounding diagnostics."""

    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class GroundingDiagnostic(BaseModel):
    """Specific diagnostic message emitted during symbol or operation verification."""

    field: str = Field(description="The component, attribute, or operand evaluated.")
    message: str = Field(description="Human-readable diagnostic explanation.")
    severity: DiagnosticSeverity = Field(
        default=DiagnosticSeverity.ERROR,
        description="Severity level of the diagnostic.",
    )
    suggested_fix: Optional[str] = Field(
        default=None,
        description="Suggested replacement or typo correction.",
    )


class GroundingReport(BaseModel):
    """Comprehensive validation report for a verified operation or symbol."""

    is_grounded: bool = Field(
        description="True if the symbol is grounded and valid without fatal errors."
    )
    target: str = Field(description="Target framework, dialect, or ISA.")
    symbol: str = Field(
        description="Queried symbol, mnemonic, or operation identifier."
    )
    diagnostics: List[GroundingDiagnostic] = Field(
        default_factory=list,
        description="Collection of diagnostics produced during verification.",
    )
    matched_ref: Optional[ExtendedGhostRef] = Field(
        default=None,
        description="Resolved ground-truth reference object if discovered.",
    )

    @property
    def has_errors(self) -> bool:
        """Check if any diagnostics have ERROR severity.

        Returns:
            True if any diagnostic is an ERROR, False otherwise.
        """
        return any(d.severity == DiagnosticSeverity.ERROR for d in self.diagnostics)

    def add_diagnostic(
        self,
        field: str,
        message: str,
        severity: DiagnosticSeverity = DiagnosticSeverity.ERROR,
        suggested_fix: Optional[str] = None,
    ) -> None:
        """Append a new diagnostic message to the report.

        Args:
            field: Component or operand under evaluation.
            message: Explanatory error or warning text.
            severity: Severity classification level.
            suggested_fix: Optional typo correction or valid value suggestion.
        """
        self.diagnostics.append(
            GroundingDiagnostic(
                field=field,
                message=message,
                severity=severity,
                suggested_fix=suggested_fix,
            )
        )
        if severity == DiagnosticSeverity.ERROR:
            self.is_grounded = False
