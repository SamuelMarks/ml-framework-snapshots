"""MaxText API Snapshot Extractor.

Extracts layer and model configurations from MaxText via AST parsing
to avoid complex dependency chains.
"""

import ast
import glob
import os
from typing import List

from ml_switcheroo_ir.schema.ghost import GhostRef
from ml_switcheroo_ir.schema.ghost import SemanticTier
from ml_switcheroo_ir.schema.ghost import GhostParam

import typing

try:
    import maxtext as _maxtext  # pragma: no cover

    maxtext: typing.Any = _maxtext  # pragma: no cover
except ImportError:  # pragma: no cover
    maxtext = None


def _parse_maxtext_classes(dir_path: str, module_prefix: str) -> List[GhostRef]:
    """Parse MaxText model classes from source files.

    Args:
        dir_path: The path to the models directory.
        module_prefix: The module prefix to use.

    Returns:
        A list of GhostRef items.
    """
    refs = []
    for file in glob.glob(os.path.join(dir_path, "**", "*.py"), recursive=True):
        try:
            with open(file, "r", encoding="utf-8") as f:
                tree = ast.parse(f.read())

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    params = []
                    docstring = ast.get_docstring(node) or ""

                    for subnode in node.body:
                        if (
                            isinstance(subnode, ast.FunctionDef)
                            and subnode.name == "__init__"
                        ):
                            for arg in subnode.args.args:
                                if arg.arg != "self":
                                    params.append(
                                        GhostParam(
                                            name=arg.arg,
                                            kind="POSITIONAL_OR_KEYWORD",
                                        )
                                    )
                            for arg in subnode.args.kwonlyargs:
                                params.append(
                                    GhostParam(name=arg.arg, kind="KEYWORD_ONLY")
                                )

                    refs.append(
                        GhostRef(
                            name=node.name,
                            api_path=f"{module_prefix}.{node.name}",
                            kind="CLASS",
                            params=params,
                            docstring=docstring,
                        )
                    )
        except Exception:  # pragma: no cover
            pass
    return refs


def collect_api(
    category: SemanticTier, include_nonpublic: bool = False
) -> List[GhostRef]:
    """Entrypoint to collect the MaxText API signature for a given category.

    Args:
        category: The category of API to collect.
        include_nonpublic: Whether to include non-public APIs.

    Returns:
        A list of GhostRef items discovered for the requested category.
    """
    if not maxtext or category != SemanticTier.MODEL:
        return []

    if getattr(maxtext, "__path__", None):
        maxtext_dir = maxtext.__path__[0]
        models_dir = os.path.join(maxtext_dir, "models")
        return _parse_maxtext_classes(models_dir, "maxtext.models")
    return []
