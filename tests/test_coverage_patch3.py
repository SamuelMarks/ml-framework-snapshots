"""Module docstring."""

# --- DIFF ---
"""Module docstring."""


def test_diff_branches_more() -> None:
    """Function docstring."""
    from ml_framework_snapshots.diff import diff_snapshots

    s1 = {
        "categories": {
            "MODEL": [
                {
                    "name": "A",
                    "api_path": "A",
                    "kind": "function",
                    "params": [
                        {
                            "name": "p1",
                            "kind": "POSITIONAL_OR_KEYWORD",
                            "default": "1",
                            "annotation": "int",
                        }
                    ],
                }
            ]
        }
    }
    s2 = {
        "categories": {
            "MODEL": [
                {
                    "name": "A",
                    "api_path": "A",
                    "kind": "function",
                    "params": [
                        {
                            "name": "p1",
                            "kind": "POSITIONAL_OR_KEYWORD",
                            "default": "2",
                            "annotation": "int",
                        }
                    ],
                }
            ]
        }
    }
    res = diff_snapshots(s1, s2)
    assert len(res.signature_changed) > 0


# --- EXPORT ---
def test_export_branches_more() -> None:
    """Function docstring."""
    from ml_framework_snapshots.export import (
        to_pydantic,
        to_openapi,
        to_json_schema,
    )
    from ml_switcheroo_ir.schema.ghost import GhostParam
    from ml_switcheroo_ir.schema.ghost import GhostRef

    r = GhostRef(
        name="X",
        api_path="X",
        kind="class",
        docstring="doc",
        params=[
            GhostParam(
                name="p1", kind="KEYWORD_ONLY", default="None", annotation="int"
            ),
            GhostParam(
                name="p2", kind="VAR_POSITIONAL", default="None", annotation="int"
            ),
        ],
    )
    to_pydantic(r)
    to_openapi([r])
    to_json_schema(r)

    r2 = GhostRef(
        name="X",
        api_path="X",
        kind="function",
        returns_type="int",
        returns_description="desc",
        params=[
            GhostParam(
                name="p1",
                kind="KEYWORD_ONLY",
                default="None",
                annotation="int",
                description="desc",
            ),
        ],
    )
    to_pydantic(r2)


# --- MODELS ---
def test_models_branches() -> None:
    """Function docstring."""
    from ml_framework_snapshots.models import sanitize_type_str

    assert sanitize_type_str("typing.List") == "list"
    assert sanitize_type_str("builtins.str") == "builtins.str"


# --- UTILS ---
def test_utils_branches() -> None:
    """Function docstring."""
    from ml_framework_snapshots.utils import (
        extract_c_extension_signature,
    )

    class Dummy:
        """Class docstring."""

        pass

    Dummy.__doc__ = "\n"
    assert extract_c_extension_signature(Dummy, "X") is None
