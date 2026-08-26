"""LaTeX DSL API Snapshot Extractor.

Provides a static snapshot of standard LaTeX math environments and formatting macros.
"""

from typing import List

from ml_switcheroo_ir.schema.ghost import GhostRef
from ml_switcheroo_ir.schema.ghost import SemanticTier
from ml_switcheroo_ir.schema.ghost import GhostParam

_LATEX_MACROS = [
    "frac",
    "sqrt",
    "sum",
    "prod",
    "int",
    "iint",
    "iiint",
    "oint",
    "alpha",
    "beta",
    "gamma",
    "delta",
    "epsilon",
    "varepsilon",
    "zeta",
    "eta",
    "theta",
    "vartheta",
    "iota",
    "kappa",
    "lambda",
    "mu",
    "nu",
    "xi",
    "pi",
    "varpi",
    "rho",
    "varrho",
    "sigma",
    "varsigma",
    "tau",
    "upsilon",
    "phi",
    "varphi",
    "chi",
    "psi",
    "omega",
    "Gamma",
    "Delta",
    "Theta",
    "Lambda",
    "Xi",
    "Pi",
    "Sigma",
    "Upsilon",
    "Phi",
    "Psi",
    "Omega",
    "sin",
    "cos",
    "tan",
    "csc",
    "sec",
    "cot",
    "sinh",
    "cosh",
    "tanh",
    "arcsin",
    "arccos",
    "arctan",
    "ln",
    "log",
    "exp",
    "lim",
    "limsup",
    "liminf",
    "max",
    "min",
    "inf",
    "sup",
    "mathbf",
    "mathit",
    "mathrm",
    "mathsf",
    "mathtt",
    "mathcal",
    "mathbb",
    "mathfrak",
    "text",
    "textbf",
    "textit",
    "textrm",
    "textsf",
    "texttt",
    "left",
    "right",
    "langle",
    "rangle",
    "lbrace",
    "rbrace",
    "lceil",
    "rceil",
    "lfloor",
    "rfloor",
    "vert",
    "Vert",
]

_LATEX_ENVIRONMENTS = [
    "equation",
    "equation*",
    "align",
    "align*",
    "gather",
    "gather*",
    "matrix",
    "pmatrix",
    "bmatrix",
    "Bmatrix",
    "vmatrix",
    "Vmatrix",
    "cases",
    "array",
]


def collect_api(
    category: SemanticTier, include_nonpublic: bool = False
) -> List[GhostRef]:
    """Entrypoint to collect the LaTeX DSL API signature.

    Args:
        category: The category of API to collect.
        include_nonpublic: Whether to include non-public APIs.

    Returns:
        A list of GhostRef items discovered.
    """
    if category != SemanticTier.UTIL:
        return []

    refs = []

    # Macros
    for macro in _LATEX_MACROS:
        params = [GhostParam(name="args", kind="VAR_POSITIONAL")]
        refs.append(
            GhostRef(
                name=macro,
                api_path=f"latex.{macro}",
                kind="function",
                params=params,
                docstring=f"LaTeX \\{macro} macro.",
            )
        )

    # Environments
    for env in _LATEX_ENVIRONMENTS:
        params = [GhostParam(name="content", kind="POSITIONAL_OR_KEYWORD")]
        refs.append(
            GhostRef(
                name=env.replace("*", "_star"),
                api_path=f"latex.env.{env.replace('*', '_star')}",
                kind="CLASS",
                params=params,
                docstring=f"LaTeX \\begin{{{env}}} environment.",
            )
        )

    return refs
