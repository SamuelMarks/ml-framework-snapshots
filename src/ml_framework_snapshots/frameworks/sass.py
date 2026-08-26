"""SASS DSL API Snapshot Extractor.

Provides a static snapshot of standard SASS mixins and properties.
"""

from typing import List

from ml_switcheroo_ir.schema.ghost import GhostRef
from ml_switcheroo_ir.schema.ghost import SemanticTier
from ml_switcheroo_ir.schema.ghost import GhostParam

_SASS_FEATURES = [
    "mixin",
    "include",
    "extend",
    "function",
    "return",
    "if",
    "else",
    "for",
    "each",
    "while",
    "import",
    "use",
    "forward",
    "warn",
    "error",
    "debug",
]

_CSS_PROPERTIES = [
    "color",
    "background",
    "background-color",
    "margin",
    "padding",
    "border",
    "border-radius",
    "width",
    "height",
    "display",
    "position",
    "top",
    "right",
    "bottom",
    "left",
    "flex",
    "grid",
    "font-family",
    "font-size",
    "font-weight",
    "text-align",
    "text-decoration",
    "line-height",
    "opacity",
    "transition",
    "transform",
    "animation",
    "box-shadow",
    "cursor",
    "z-index",
    "overflow",
]


def collect_api(
    category: SemanticTier, include_nonpublic: bool = False
) -> List[GhostRef]:
    """Entrypoint to collect the SASS DSL API signature.

    Args:
        category: The category of API to collect.
        include_nonpublic: Whether to include non-public APIs.

    Returns:
        A list of GhostRef items discovered.
    """
    if category != SemanticTier.UTIL:
        return []

    refs = []

    for feat in _SASS_FEATURES:
        params = [
            GhostParam(name="args", kind="VAR_POSITIONAL"),
            GhostParam(name="kwargs", kind="VAR_KEYWORD"),
        ]
        refs.append(
            GhostRef(
                name=feat,
                api_path=f"sass.at_{feat}",
                kind="function",
                params=params,
                docstring=f"SASS @{feat} directive.",
            )
        )

    for prop in _CSS_PROPERTIES:
        clean_name = prop.replace("-", "_")
        params = [
            GhostParam(name="value", kind="POSITIONAL_OR_KEYWORD"),
        ]
        refs.append(
            GhostRef(
                name=clean_name,
                api_path=f"sass.prop.{clean_name}",
                kind="function",
                params=params,
                docstring=f"CSS property {prop}.",
            )
        )

    return refs
