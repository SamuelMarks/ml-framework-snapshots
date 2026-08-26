"""HTML DSL API Snapshot Extractor.

Provides a static snapshot of standard HTML tags and attributes for the HTML DSL.
"""

from typing import List

from ml_switcheroo_ir.schema.ghost import GhostRef
from ml_switcheroo_ir.schema.ghost import SemanticTier
from ml_switcheroo_ir.schema.ghost import GhostParam

_HTML_TAGS = [
    "div",
    "span",
    "a",
    "p",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "ul",
    "ol",
    "li",
    "table",
    "tr",
    "td",
    "th",
    "thead",
    "tbody",
    "img",
    "form",
    "input",
    "button",
    "select",
    "option",
    "textarea",
    "label",
    "br",
    "hr",
    "strong",
    "em",
    "head",
    "body",
    "html",
    "title",
    "meta",
    "link",
    "style",
    "script",
    "nav",
    "header",
    "footer",
    "main",
    "section",
    "article",
    "aside",
    "figure",
    "figcaption",
]


def collect_api(
    category: SemanticTier, include_nonpublic: bool = False
) -> List[GhostRef]:
    """Entrypoint to collect the HTML DSL API signature.

    Args:
        category: The category of API to collect.
        include_nonpublic: Whether to include non-public APIs.

    Returns:
        A list of GhostRef items discovered.
    """
    if category != SemanticTier.UTIL:
        return []

    refs = []
    for tag in _HTML_TAGS:
        params = [
            GhostParam(name="children", kind="POSITIONAL_OR_KEYWORD"),
            GhostParam(name="id", kind="KEYWORD_ONLY"),
            GhostParam(name="class_name", kind="KEYWORD_ONLY"),
            GhostParam(name="style", kind="KEYWORD_ONLY"),
            GhostParam(name="kwargs", kind="VAR_KEYWORD"),
        ]

        refs.append(
            GhostRef(
                name=tag,
                api_path=f"html.{tag}",
                kind="CLASS",
                params=params,
                docstring=f"HTML <{tag}> element.",
            )
        )
    return refs
