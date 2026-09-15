"""ML-Switcheroo IR API Snapshot Extractor.

Extracts symbols and operations from ml_switcheroo_ir.
"""

from typing import List
from ml_switcheroo_ir.schema.ghost import (
    GhostParam,
    GhostRef,
    ParameterKind,
    SemanticTier,
)


def collect_api(
    category: SemanticTier, include_nonpublic: bool = False
) -> List[GhostRef]:
    """Collect IR API symbols for the given category.

    Args:
        category: The SemanticTier category.
        include_nonpublic: Whether to include private symbols.

    Returns:
        List of GhostRef definitions.
    """
    refs: List[GhostRef] = []

    if category in (SemanticTier.ARRAY_API, SemanticTier.NEURAL, SemanticTier.UTIL):
        # Core symbols
        refs.append(
            GhostRef(
                name="LogicalGraph",
                kind="class",
                api_path="ml_switcheroo_ir.LogicalGraph",
                docstring="Universal computation graph for Deep Learning models.",
                params=[
                    GhostParam(
                        name="name",
                        kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                        default="'Model'",
                        annotation="str",
                    ),
                    GhostParam(
                        name="nodes",
                        kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                        default="None",
                        annotation="dict | list",
                    ),
                    GhostParam(
                        name="edges",
                        kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                        default="None",
                        annotation="list",
                    ),
                    GhostParam(
                        name="mesh",
                        kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                        default="None",
                        annotation="LogicalMesh | None",
                    ),
                ],
            )
        )
        refs.append(
            GhostRef(
                name="LogicalNode",
                kind="class",
                api_path="ml_switcheroo_ir.LogicalNode",
                docstring="Individual operation node within a LogicalGraph.",
                params=[
                    GhostParam(
                        name="id",
                        kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                        annotation="str",
                    ),
                    GhostParam(
                        name="kind",
                        kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                        annotation="str",
                    ),
                    GhostParam(
                        name="domain",
                        kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                        default="'ai.onnx'",
                        annotation="str",
                    ),
                    GhostParam(
                        name="version",
                        kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                        default="1",
                        annotation="int",
                    ),
                    GhostParam(
                        name="metadata",
                        kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                        default="None",
                        annotation="dict",
                    ),
                    GhostParam(
                        name="sharding",
                        kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                        default="None",
                        annotation="PartitionSpec | None",
                    ),
                ],
            )
        )
        refs.append(
            GhostRef(
                name="topological_sort",
                kind="function",
                api_path="ml_switcheroo_ir.topological_sort",
                docstring="Sort graph nodes in dependency order.",
                params=[
                    GhostParam(
                        name="graph",
                        kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                        annotation="LogicalGraph",
                    ),
                ],
            )
        )

    return refs
