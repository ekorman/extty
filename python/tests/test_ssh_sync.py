"""Tests for SSH sync functionality."""

from unittest import mock

import pytest

# Try to import paramiko, skip tests if not available
try:
    import paramiko
    PARAMIKO_AVAILABLE = True
except ImportError:
    PARAMIKO_AVAILABLE = False

from extty.ssh_sync import (
    parse_ssh_host,
    _parse_metric_csv_string,
    _parse_system_csv_string,
)


class TestParseSSHHost:
    def test_user_and_host(self) -> None:
        username, hostname = parse_ssh_host("alice@example.com")
        assert username == "alice"
        assert hostname == "example.com"

    def test_host_only(self) -> None:
        with mock.patch("getpass.getuser", return_value="bob"):
            username, hostname = parse_ssh_host("example.com")
            assert username == "bob"
            assert hostname == "example.com"

    def test_complex_hostname(self) -> None:
        username, hostname = parse_ssh_host("user@192.168.1.100")
        assert username == "user"
        assert hostname == "192.168.1.100"


class TestParseMetricCSV:
    def test_parse_simple_csv(self) -> None:
        csv_content = """step,timestamp,value
0,1705330320.123456,0.5
1,1705330321.234567,0.45
2,1705330322.345678,0.42"""

        points = _parse_metric_csv_string(csv_content)
        assert len(points) == 3
        assert points[0]["step"] == 0
        assert points[0]["value"] == 0.5
        assert points[1]["step"] == 1
        assert points[2]["value"] == 0.42

    def test_parse_empty_csv(self) -> None:
        csv_content = "step,timestamp,value\n"
        points = _parse_metric_csv_string(csv_content)
        assert len(points) == 0

    def test_parse_csv_with_extra_columns(self) -> None:
        csv_content = """step,timestamp,value,extra
0,1705330320.123456,0.5,ignored"""

        points = _parse_metric_csv_string(csv_content)
        assert len(points) == 1
        assert points[0]["step"] == 0
        assert points[0]["value"] == 0.5


class TestParseSystemCSV:
    def test_parse_system_csv_with_gpu(self) -> None:
        csv_content = """timestamp,ram_used_gb,ram_total_gb,gpu_mem_used_gb,gpu_mem_total_gb,gpu_util_pct
1705330320.123456,8.50,32.00,4.25,24.00,50.5
1705330325.234567,8.75,32.00,5.00,24.00,62.3"""

        points = _parse_system_csv_string(csv_content)
        assert len(points) == 2
        assert points[0]["ram_used_gb"] == 8.50
        assert points[0]["gpu_mem_used_gb"] == 4.25
        assert points[1]["gpu_util_pct"] == 62.3

    def test_parse_system_csv_without_gpu(self) -> None:
        csv_content = """timestamp,ram_used_gb,ram_total_gb,gpu_mem_used_gb,gpu_mem_total_gb,gpu_util_pct
1705330320.123456,8.50,32.00,,,"""

        points = _parse_system_csv_string(csv_content)
        assert len(points) == 1
        assert points[0]["ram_used_gb"] == 8.50
        assert points[0]["gpu_mem_used_gb"] is None
        assert points[0]["gpu_util_pct"] is None

    def test_parse_empty_system_csv(self) -> None:
        csv_content = "timestamp,ram_used_gb,ram_total_gb,gpu_mem_used_gb,gpu_mem_total_gb,gpu_util_pct\n"
        points = _parse_system_csv_string(csv_content)
        assert len(points) == 0


@pytest.mark.skipif(not PARAMIKO_AVAILABLE, reason="paramiko not installed")
class TestSSHConnection:
    """Tests that require paramiko to be installed."""

    def test_ensure_paramiko_with_import(self) -> None:
        """Test that _ensure_paramiko doesn't raise when paramiko is available."""
        from extty.ssh_sync import _ensure_paramiko
        _ensure_paramiko()  # Should not raise

    def test_create_ssh_client_connection_failure(self) -> None:
        """Test SSH connection failure handling."""
        from extty.ssh_sync import create_ssh_client

        with pytest.raises(RuntimeError, match="Failed to connect"):
            # Try to connect to a non-existent host
            create_ssh_client("testuser", "invalid-host-that-does-not-exist.local")
