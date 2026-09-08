"""Module docstring."""

from typing import Any, Dict, List
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


def test_diff_branches() -> None:
    """Function docstring."""
    from ml_framework_snapshots.diff import diff_snapshots

    # Hit line 81->87 (_compare_params empty)
    s1 = {
        "categories": {
            "MODEL": [{"name": "A", "api_path": "A", "kind": "function", "params": []}]
        }
    }
    s2 = {
        "categories": {
            "MODEL": [{"name": "A", "api_path": "A", "kind": "function", "params": []}]
        }
    }
    diff_snapshots(s1, s2)

    # Hit branch where kind changed but it's not breaking (e.g. POSITIONAL_ONLY -> POSITIONAL_OR_KEYWORD)
    s3 = {
        "categories": {
            "MODEL": [
                {
                    "name": "B",
                    "api_path": "B",
                    "kind": "function",
                    "params": [{"name": "p", "kind": "POSITIONAL_ONLY"}],
                }
            ]
        }
    }
    s4 = {
        "categories": {
            "MODEL": [
                {
                    "name": "B",
                    "api_path": "B",
                    "kind": "function",
                    "params": [{"name": "p", "kind": "POSITIONAL_OR_KEYWORD"}],
                }
            ]
        }
    }
    diff_snapshots(s3, s4)


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


def test_diff_domain_metadata_breaking_changes() -> None:
    """Test breaking changes detected when assembly modifiers or architecture support is removed."""
    s1 = {
        "categories": {
            "UTIL": [
                {
                    "name": "FADD",
                    "api_path": "nvidia_sass.inst.FADD",
                    "kind": "function",
                    "params": [],
                    "domain_metadata": {
                        "modifiers": [".SAT", ".FTZ"],
                        "valid_architectures": ["sm_70", "sm_80", "sm_90"],
                    },
                }
            ]
        }
    }
    # Removed .SAT modifier (breaking)
    s2 = {
        "categories": {
            "UTIL": [
                {
                    "name": "FADD",
                    "api_path": "nvidia_sass.inst.FADD",
                    "kind": "function",
                    "params": [],
                    "domain_metadata": {
                        "modifiers": [".FTZ"],
                        "valid_architectures": ["sm_70", "sm_80", "sm_90"],
                    },
                }
            ]
        }
    }
    res = diff_snapshots(s1, s2)
    assert "nvidia_sass.inst.FADD" in res.breaking_signature_changed

    # Removed sm_70 architecture (breaking)
    s3 = {
        "categories": {
            "UTIL": [
                {
                    "name": "FADD",
                    "api_path": "nvidia_sass.inst.FADD",
                    "kind": "function",
                    "params": [],
                    "domain_metadata": {
                        "modifiers": [".SAT", ".FTZ"],
                        "valid_architectures": ["sm_80", "sm_90"],
                    },
                }
            ]
        }
    }
    res_arch = diff_snapshots(s1, s3)
    assert "nvidia_sass.inst.FADD" in res_arch.breaking_signature_changed

    # Identical metadata with non-breaking parameter addition (covers line 61->64)
    s4 = {
        "categories": {
            "UTIL": [
                {
                    "name": "FADD",
                    "api_path": "nvidia_sass.inst.FADD",
                    "kind": "function",
                    "params": [{"name": "opt", "default": "0", "kind": "KEYWORD_ONLY"}],
                    "domain_metadata": {
                        "modifiers": [".SAT", ".FTZ"],
                        "valid_architectures": ["sm_70", "sm_80", "sm_90"],
                    },
                }
            ]
        }
    }
    res_non_breaking = diff_snapshots(s1, s4)
    assert "nvidia_sass.inst.FADD" in res_non_breaking.non_breaking_signature_changed


def test_diff_snapshots_list_and_non_dict() -> None:
    """Test diff_snapshots with list inputs, invalid types, and empty items."""
    # List format
    list_snap1: List[Dict[str, Any]] = [{"api_path": "foo.bar", "params": []}]
    list_snap2: List[Dict[str, Any]] = [
        {"api_path": "foo.bar", "params": [{"name": "x"}]}
    ]
    res_list = diff_snapshots(list_snap1, list_snap2)
    assert "foo.bar" in res_list.signature_changed

    # Non-dict and non-list format fallback
    res_invalid = diff_snapshots(None, 123)
    assert res_invalid.added == []
    assert res_invalid.removed == []

    # Item without api_path, name, or mnemonic
    empty_item_snap1: Dict[str, Any] = {"categories": {"misc": [{}]}}
    empty_item_snap2: Dict[str, Any] = {"categories": {"misc": [{"other_key": "val"}]}}
    res_empty_item = diff_snapshots(empty_item_snap1, empty_item_snap2)
    assert res_empty_item.added == []
    assert res_empty_item.removed == []


def test_diff_metadata_and_deprecated_capabilities() -> None:
    """Test tracking top-level envelope metadata changes and deprecated capabilities."""
    snap_old: Dict[str, Any] = {
        "schema_version": "1.0.0",
        "target": "nvidia_sass",
        "version": "11.8.0",
        "source_type": "binary_disassembly",
        "upstream_commit": "cuda-11.8",
        "environment_tags": ["sm_50", "sm_60", "sm_70", "sm_80"],
        "categories": {
            "UTIL": [{"name": "FADD", "api_path": "FADD", "params": []}],
        },
    }
    snap_new: Dict[str, Any] = {
        "schema_version": "1.1.0",
        "target": "nvidia_sass",
        "version": "12.6.0",
        "source_type": "binary_disassembly",
        "upstream_commit": "cuda-12.6",
        "environment_tags": ["sm_70", "sm_80", "sm_90"],  # sm_50, sm_60 deprecated
        "categories": {
            "UTIL": [{"name": "FADD", "api_path": "FADD", "params": []}],
        },
    }

    diff_res = diff_snapshots(snap_old, snap_new)
    assert "schema_version" in diff_res.metadata_changed
    assert diff_res.metadata_changed["schema_version"] == ("1.0.0", "1.1.0")
    assert diff_res.metadata_changed["version"] == ("11.8.0", "12.6.0")
    assert diff_res.metadata_changed["upstream_commit"] == ("cuda-11.8", "cuda-12.6")
    assert "sm_50" in diff_res.deprecated_capabilities
    assert "sm_60" in diff_res.deprecated_capabilities

    # Test changelog rendering
    changelog = generate_changelog(diff_res)
    assert "## Metadata Changes" in changelog
    assert "- **schema_version**: `1.0.0` -> `1.1.0`" in changelog
    assert "- **version**: `11.8.0` -> `12.6.0`" in changelog
    assert "## Deprecated Capabilities" in changelog
    assert "- `sm_50`" in changelog
    assert "- `sm_60`" in changelog
