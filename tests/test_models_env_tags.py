"""Module docstring."""

from typing import Any


from ml_framework_snapshots.models import GhostInspector
from ml_switcheroo_ir.schema.ghost import GhostRef
import sys


def test_ghostref_env_tags() -> None:
    """Function docstring."""
    ref = GhostRef(
        name="test",
        api_path="test",
        kind="function",
        environment_tags=["darwin", "cpu"],
    )
    assert "darwin" in ref.environment_tags
    assert "cpu" in ref.environment_tags


def test_ghostinspector_env_tags() -> None:
    """Function docstring."""

    def dummy() -> Any:
        """Function docstring."""
        pass

    ref = GhostInspector.inspect(dummy, "dummy")
    assert sys.platform in ref.environment_tags
    assert "cpu" in ref.environment_tags or "cuda" in ref.environment_tags


def test_ghostinspector_env_tags_no_torch(monkeypatch: Any) -> None:
    """Function docstring."""
    import sys

    monkeypatch.setitem(sys.modules, "torch", None)

    def dummy() -> Any:
        """Function docstring."""
        pass

    ref = GhostInspector.inspect(dummy, "dummy")
    assert "cpu" in ref.environment_tags


def test_ghostinspector_env_tags_cuda_true(mocker: Any, monkeypatch: Any) -> None:
    """Function docstring."""
    import sys

    class MockCuda:
        """Class docstring."""

        @staticmethod
        def is_available() -> Any:
            """Function docstring."""
            return True

    class MockTorch:
        """Class docstring."""

        cuda = MockCuda()

    monkeypatch.setitem(sys.modules, "torch", MockTorch())

    def dummy() -> Any:
        """Function docstring."""
        pass

    ref = GhostInspector.inspect(dummy, "dummy")
    assert "cuda" in ref.environment_tags
