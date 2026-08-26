"""TikZ DSL API Snapshot Extractor.

Provides a static snapshot of standard TikZ drawing commands and node shapes.
"""

from typing import List

from ml_switcheroo_ir.schema.ghost import GhostRef
from ml_switcheroo_ir.schema.ghost import SemanticTier
from ml_switcheroo_ir.schema.ghost import GhostParam

_TIKZ_COMMANDS = [
    "draw",
    "fill",
    "filldraw",
    "pattern",
    "shade",
    "shadedraw",
    "clip",
    "useasboundingbox",
    "node",
    "coordinate",
    "matrix",
    "pic",
    "path",
]

_TIKZ_SHAPES = [
    "circle",
    "rectangle",
    "ellipse",
    "diamond",
    "triangle",
    "star",
    "regular polygon",
    "cylinder",
    "kite",
    "dart",
    "trapezium",
]


def collect_api(
    category: SemanticTier, include_nonpublic: bool = False
) -> List[GhostRef]:
    """Entrypoint to collect the TikZ DSL API signature.

    Args:
        category: The category of API to collect.
        include_nonpublic: Whether to include non-public APIs.

    Returns:
        A list of GhostRef items discovered.
    """
    if category != SemanticTier.UTIL:
        return []

    refs = []

    for cmd in _TIKZ_COMMANDS:
        params = [
            GhostParam(name="options", kind="KEYWORD_ONLY"),
            GhostParam(name="path", kind="POSITIONAL_OR_KEYWORD"),
        ]
        refs.append(
            GhostRef(
                name=cmd,
                api_path=f"tikz.{cmd}",
                kind="function",
                params=params,
                docstring=f"TikZ \\{cmd} command.",
            )
        )

    for shape in _TIKZ_SHAPES:
        clean_name = shape.replace(" ", "_")
        params = [
            GhostParam(name="name", kind="POSITIONAL_OR_KEYWORD"),
            GhostParam(name="at", kind="KEYWORD_ONLY"),
            GhostParam(name="options", kind="KEYWORD_ONLY"),
        ]
        refs.append(
            GhostRef(
                name=clean_name,
                api_path=f"tikz.shape.{clean_name}",
                kind="CLASS",
                params=params,
                docstring=f"TikZ {shape} shape node.",
            )
        )

    return refs
