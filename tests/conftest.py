"""Pytest global configuration and fixtures.

Adjusts process file descriptor limits for concurrent test executions and
configures environment defaults.
"""

from typing import Any


def pytest_configure(config: Any) -> None:
    """Configure test environment and raise open file limits if possible.

    Args:
        config: Pytest configuration object.
    """
    try:
        import resource

        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        target = min(4096, hard) if hard > 0 else 4096
        if soft < target:
            resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
    except (ImportError, OSError, ValueError):
        pass
