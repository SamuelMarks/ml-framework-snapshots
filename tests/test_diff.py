"""Module docstring."""

from ml_framework_snapshots.diff import diff_snapshots, generate_changelog


def test_diff_snapshots() -> None:
    """Function docstring."""
    snap1 = {
        "categories": {
            "layer": [
                {
                    "api_path": "a.b",
                    "params": [
                        {
                            "name": "x",
                            "kind": "POSITIONAL_OR_KEYWORD",
                            "default": "1",
                            "annotation": "int",
                        }
                    ],
                },
                {"api_path": "a.c", "params": []},
                {
                    "api_path": "a.non_breaking",
                    "params": [
                        {
                            "name": "x",
                            "kind": "POSITIONAL_OR_KEYWORD",
                            "default": "1",
                            "annotation": "int",
                        }
                    ],
                },
                {
                    "api_path": "a.removed_param",
                    "params": [
                        {"name": "x", "kind": "POSITIONAL_OR_KEYWORD", "default": "1"}
                    ],
                },
                {
                    "api_path": "a.added_param_no_default",
                    "params": [
                        {"name": "x", "kind": "POSITIONAL_OR_KEYWORD", "default": "1"}
                    ],
                },
                {
                    "api_path": "a.kind_change_restrictive",
                    "params": [
                        {"name": "x", "kind": "POSITIONAL_OR_KEYWORD", "default": "1"}
                    ],
                },
                {
                    "api_path": "a.kind_change_pos_to_kw",
                    "params": [
                        {"name": "x", "kind": "POSITIONAL_ONLY", "default": "1"}
                    ],
                },
                {
                    "api_path": "a.kind_change_to_var",
                    "params": [
                        {"name": "x", "kind": "POSITIONAL_OR_KEYWORD", "default": "1"}
                    ],
                },
                {
                    "api_path": "a.default_removed",
                    "params": [
                        {"name": "x", "kind": "POSITIONAL_OR_KEYWORD", "default": "1"}
                    ],
                },
            ]
        }
    }
    snap2 = {
        "categories": {
            "layer": [
                {
                    "api_path": "a.b",
                    "params": [
                        {
                            "name": "x",
                            "kind": "POSITIONAL_OR_KEYWORD",
                            "default": "2",  # Default changed (breaking)
                            "annotation": "int",
                        }
                    ],
                },
                {"api_path": "a.d", "params": []},
                {
                    "api_path": "a.non_breaking",
                    "params": [
                        {
                            "name": "x",
                            "kind": "POSITIONAL_OR_KEYWORD",
                            "default": "1",
                            "annotation": "int",
                        },
                        {
                            "name": "y",
                            "kind": "KEYWORD_ONLY",
                            "default": "2",  # Added with default (non-breaking)
                            "annotation": "int",
                        },
                    ],
                },
                {
                    "api_path": "a.removed_param",
                    "params": [],  # Parameter removed (breaking)
                },
                {
                    "api_path": "a.added_param_no_default",
                    "params": [
                        {"name": "x", "kind": "POSITIONAL_OR_KEYWORD", "default": "1"},
                        {
                            "name": "y",
                            "kind": "POSITIONAL_OR_KEYWORD",
                            "default": None,
                        },  # Added without default (breaking)
                    ],
                },
                {
                    "api_path": "a.kind_change_restrictive",
                    "params": [
                        {
                            "name": "x",
                            "kind": "POSITIONAL_ONLY",
                            "default": "1",
                        }  # More restrictive (breaking)
                    ],
                },
                {
                    "api_path": "a.kind_change_pos_to_kw",
                    "params": [
                        {
                            "name": "x",
                            "kind": "KEYWORD_ONLY",
                            "default": "1",
                        }  # Pos to Kw (breaking)
                    ],
                },
                {
                    "api_path": "a.kind_change_to_var",
                    "params": [
                        {
                            "name": "x",
                            "kind": "VAR_POSITIONAL",
                            "default": None,
                        }  # Change to var (breaking)
                    ],
                },
                {
                    "api_path": "a.default_removed",
                    "params": [
                        {
                            "name": "x",
                            "kind": "POSITIONAL_OR_KEYWORD",
                            "default": None,
                        }  # Default removed (breaking)
                    ],
                },
            ]
        }
    }

    res = diff_snapshots(snap1, snap2)
    assert res.added == ["a.d"]
    assert res.removed == ["a.c"]
    assert "a.b" in res.signature_changed
    assert "a.non_breaking" in res.signature_changed

    expected_breaking = [
        "a.added_param_no_default",
        "a.b",
        "a.default_removed",
        "a.kind_change_pos_to_kw",
        "a.kind_change_restrictive",
        "a.kind_change_to_var",
        "a.removed_param",
    ]
    assert res.breaking_signature_changed == expected_breaking
    assert res.non_breaking_signature_changed == ["a.non_breaking"]

    # Test identical
    res2 = diff_snapshots(snap1, snap1)
    assert not res2.added
    assert not res2.removed
    assert not res2.signature_changed


def test_diff_empty() -> None:
    """Function docstring."""
    res = diff_snapshots({}, {})
    assert not res.added
    assert not res.removed
    assert not res.signature_changed


def test_generate_changelog() -> None:
    """Function docstring."""
    snap1 = {
        "categories": {
            "layer": [
                {"api_path": "a.removed", "params": []},
                {
                    "api_path": "a.breaking",
                    "params": [
                        {"name": "x", "kind": "POSITIONAL_OR_KEYWORD", "default": "1"}
                    ],
                },
                {
                    "api_path": "a.non_breaking",
                    "params": [
                        {"name": "x", "kind": "POSITIONAL_OR_KEYWORD", "default": "1"}
                    ],
                },
            ]
        }
    }
    snap2 = {
        "categories": {
            "layer": [
                {"api_path": "a.added", "params": []},
                {
                    "api_path": "a.breaking",
                    "params": [
                        {"name": "x", "kind": "POSITIONAL_OR_KEYWORD", "default": "2"}
                    ],
                },
                {
                    "api_path": "a.non_breaking",
                    "params": [
                        {"name": "x", "kind": "POSITIONAL_OR_KEYWORD", "default": "1"},
                        {"name": "y", "kind": "KEYWORD_ONLY", "default": "2"},
                    ],
                },
            ]
        }
    }

    res = diff_snapshots(snap1, snap2)
    changelog = generate_changelog(res)

    assert "## Added" in changelog
    assert "- `a.added`" in changelog
    assert "## Removed (Breaking)" in changelog
    assert "- `a.removed`" in changelog
    assert "## Breaking Signature Changes" in changelog
    assert "- `a.breaking`" in changelog
    assert "## Non-Breaking Signature Changes" in changelog
    assert "- `a.non_breaking`" in changelog


def test_generate_changelog_empty() -> None:
    """Function docstring."""
    res = diff_snapshots({}, {})
    changelog = generate_changelog(res)
    assert "No changes detected." in changelog


def test_changelog_combinations() -> None:
    """Function docstring."""
    from ml_framework_snapshots.diff import DiffResult

    # Only added
    res1 = DiffResult(
        added=["a"],
        removed=[],
        signature_changed=[],
        breaking_signature_changed=[],
        non_breaking_signature_changed=[],
    )
    changelog = generate_changelog(res1)
    assert "## Added" in changelog
    assert "## Removed" not in changelog

    # Only removed
    res2 = DiffResult(
        added=[],
        removed=["b"],
        signature_changed=[],
        breaking_signature_changed=[],
        non_breaking_signature_changed=[],
    )
    changelog = generate_changelog(res2)
    assert "## Added" not in changelog
    assert "## Removed" in changelog

    # Only breaking
    res3 = DiffResult(
        added=[],
        removed=[],
        signature_changed=["c"],
        breaking_signature_changed=["c"],
        non_breaking_signature_changed=[],
    )
    changelog = generate_changelog(res3)
    assert "## Breaking Signature Changes" in changelog

    # Only non-breaking
    res4 = DiffResult(
        added=[],
        removed=[],
        signature_changed=["d"],
        breaking_signature_changed=[],
        non_breaking_signature_changed=["d"],
    )
    changelog = generate_changelog(res4)
    assert "## Non-Breaking Signature Changes" in changelog
