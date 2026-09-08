"""Offline mode verification test suite.

Ensures that all MCP grounding tools (check_hallucination, check_sass_instruction,
check_rdna_instruction, check_ptx_instruction, check_mlir_op, check_stablehlo_op)
operate strictly without network access, querying bundled ground-truth JSON files.
"""

import socket
from typing import Any
from unittest import mock
import pytest

from ml_framework_snapshots import mcp_server


@pytest.fixture(autouse=True)
def forbid_network_connections() -> Any:
    """Fixture that forbids any outgoing socket connections during offline test execution.

    Yields:
        None after patching socket.socket.connect.
    """

    def _block_connect(*args: Any, **kwargs: Any) -> None:
        """Blocked connect function.

        Args:
            *args: Positional arguments for socket.connect.
            **kwargs: Keyword arguments for socket.connect.

        Raises:
            RuntimeError: Always raised to simulate offline isolated environment.
        """
        raise RuntimeError("Network access forbidden in offline mode")

    with mock.patch.object(socket.socket, "connect", side_effect=_block_connect):
        yield


def test_offline_mcp_sass_grounding() -> None:
    """Verify SASS instruction verification operates offline using bundled JSON."""
    res = mcp_server.check_sass_instruction("FADD", operands=["R0", "R1", "R2"])
    assert res["is_valid"] is True
    assert res["mnemonic_exists"] is True


def test_offline_mcp_rdna_grounding() -> None:
    """Verify AMD RDNA instruction verification operates offline using bundled JSON."""
    res = mcp_server.check_rdna_instruction("v_add_f32", operands=["v0", "v1", "v2"])
    assert res["is_valid"] is True
    assert res["mnemonic_exists"] is True


def test_offline_mcp_ptx_grounding() -> None:
    """Verify NVIDIA PTX instruction verification operates offline using bundled JSON."""
    res = mcp_server.check_ptx_instruction(
        "add",
        types=[".f32"],
        operands=["%f0", "%f1", "%f2"],
    )
    assert res["is_valid"] is True
    assert res["mnemonic_exists"] is True


def test_offline_mcp_mlir_grounding() -> None:
    """Verify MLIR operation verification operates offline using bundled JSON."""
    res = mcp_server.check_mlir_op("arith.addf")
    assert res["is_valid"] is True
    assert res["op_exists"] is True


def test_offline_mcp_stablehlo_grounding() -> None:
    """Verify StableHLO operation verification operates offline using bundled JSON."""
    res = mcp_server.check_stablehlo_op("stablehlo.add")
    assert res["is_valid"] is True
    assert res["op_exists"] is True


def test_offline_mcp_hallucination_check() -> None:
    """Verify API hallucination checking operates offline using local snapshot."""
    # Test checking a known PyTorch API offline
    res = mcp_server.check_hallucination(
        framework="torch",
        api_path="torch.relu",
        strict_kwargs=True,
    )
    assert res["api_exists"] is True
    assert res["is_hallucinated"] is False
