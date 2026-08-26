"""MLIR Dialect API Snapshot Extractor.

Extracts MLIR dialects and operations (e.g., arith, math, scf, mhlo) by parsing
the python bindings typically provided by jaxlib.mlir or similar packages.
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
    import jaxlib.mlir.dialects as _mlir_dialects

    mlir_dialects: typing.Any = _mlir_dialects
except ImportError:  # pragma: no cover
    mlir_dialects = None


def _parse_mlir_ops(dir_path: str) -> List[GhostRef]:
    """Parse MLIR operation classes from *_ops_gen.py files.

    Args:
        dir_path: The directory path containing the dialects.

    Returns:
        A list of GhostRef items for the operations.
    """
    refs = []
    for file in glob.glob(os.path.join(dir_path, "*_ops_gen.py")):
        # Extract dialect name from filename (e.g., _arith_ops_gen.py -> arith)
        basename = os.path.basename(file)
        if basename.startswith("_") and basename.endswith("_ops_gen.py"):
            try:
                with open(file, "r", encoding="utf-8") as f:
                    tree = ast.parse(f.read())

                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        # Extract the operation name if present (e.g., OPERATION_NAME = "arith.addf")
                        op_name = None
                        for subnode in node.body:
                            if isinstance(subnode, ast.Assign):
                                for target in subnode.targets:
                                    if getattr(target, "id", None) == "OPERATION_NAME":
                                        if isinstance(subnode.value, ast.Constant):
                                            op_name = subnode.value.value

                        if op_name:
                            params = []
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
                                            GhostParam(
                                                name=arg.arg, kind="KEYWORD_ONLY"
                                            )
                                        )

                            refs.append(
                                GhostRef(
                                    name=node.name,
                                    api_path=op_name,
                                    kind="CLASS",
                                    params=params,
                                )
                            )
            except Exception:  # pragma: no cover
                pass
    return refs


def collect_api(
    category: SemanticTier, include_nonpublic: bool = False
) -> List[GhostRef]:
    """Entrypoint to collect the MLIR API signature for a given category.

    Args:
        category: The category of API to collect.
        include_nonpublic: Whether to include non-public APIs.

    Returns:
        A list of GhostRef items discovered for the requested category.
    """
    if not mlir_dialects or category != SemanticTier.UTIL:
        return []

    if getattr(mlir_dialects, "__path__", None):
        dialects_dir = mlir_dialects.__path__[0]
        return _parse_mlir_ops(dialects_dir)
    return []
