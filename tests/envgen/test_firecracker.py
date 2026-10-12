"""Tests for Firecracker MicroVM runtime, client, jailer, and runtime factory."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

from forge.contracts.sandbox import (
    FirecrackerVMConfig,
    IsolationBackendType,
    SandboxIsolationBackend,
)
from forge.envgen.container import ContainerRuntime
from forge.envgen.firecracker.client import FirecrackerAPIError, FirecrackerClient
from forge.envgen.firecracker.jailer import FirecrackerJailer, JailerConfig
from forge.envgen.firecracker.rootfs import RootfsManager
from forge.envgen.firecracker.runtime import FirecrackerMicroVMRuntime
from forge.envgen.runtime_factory import (
    get_sandbox_runtime,
    is_firecracker_available,
    is_kvm_available,
)


# ==============================================================================
# Firecracker REST Client Tests
# ==============================================================================

class _MockFirecrackerTransport(httpx.BaseTransport):
    """Simulates Firecracker UDS REST API responses."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict]] = []

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8")) if request.content else {}
        self.calls.append((request.method, str(request.url.path), body))

        if request.url.path == "/machine-config" and request.method == "GET":
            return httpx.Response(200, json={"vcpu_count": 2, "mem_size_mib": 1024})
        if request.url.path == "/error-trigger":
            return httpx.Response(400, json={"fault_message": "Invalid boot source"})
        return httpx.Response(204)


def test_firecracker_client_configure_and_boot(tmp_path: Path):
    sock_path = tmp_path / "fc.sock"
    sock_path.touch()
    client = FirecrackerClient(sock_path)
    transport = _MockFirecrackerTransport()

    # Patch client transport
    with patch.object(client, "_get_client", return_value=httpx.Client(transport=transport, base_url="http://localhost")):
        # Machine config
        client.set_machine_config(vcpu_count=4, mem_size_mib=2048)
        assert transport.calls[-1] == ("PUT", "/machine-config", {
            "vcpu_count": 4, "mem_size_mib": 2048, "smt": False, "track_dirty_pages": False
        })

        # Boot source
        client.set_boot_source(
            kernel_image_path="/boot/vmlinux",
            boot_args="console=ttyS0 root=/dev/vda",
        )
        assert transport.calls[-1] == ("PUT", "/boot-source", {
            "kernel_image_path": "/boot/vmlinux",
            "boot_args": "console=ttyS0 root=/dev/vda",
        })

        # Drive
        client.attach_drive("rootfs", path_on_host="/var/run/rootfs.ext4", is_root_device=True)
        assert transport.calls[-1] == ("PUT", "/drives/rootfs", {
            "drive_id": "rootfs",
            "path_on_host": "/var/run/rootfs.ext4",
            "is_root_device": True,
            "is_read_only": False,
        })

        # Network interface
        client.attach_network_interface("eth0", host_dev_name="tap0", guest_mac="AA:FC:00:00:00:01")
        assert transport.calls[-1] == ("PUT", "/network-interfaces/eth0", {
            "iface_id": "eth0",
            "host_dev_name": "tap0",
            "guest_mac": "AA:FC:00:00:00:01",
        })

        # Start instance
        client.start_instance()
        assert transport.calls[-1] == ("PUT", "/actions", {"action_type": "InstanceStart"})

        # Send ACPI shutdown
        client.send_ctrl_alt_del()
        assert transport.calls[-1] == ("PUT", "/actions", {"action_type": "SendCtrlAltDel"})


def test_firecracker_client_handles_api_errors(tmp_path: Path):
    sock_path = tmp_path / "fc.sock"
    sock_path.touch()
    client = FirecrackerClient(sock_path)
    transport = _MockFirecrackerTransport()

    with patch.object(client, "_get_client", return_value=httpx.Client(transport=transport, base_url="http://localhost")):
        with pytest.raises(FirecrackerAPIError) as exc_info:
            client._request("PUT", "/error-trigger")
        assert exc_info.value.status_code == 400
        assert "Invalid boot source" in str(exc_info.value)


# ==============================================================================
# AWS Jailer Tests
# ==============================================================================

def test_jailer_command_construction():
    jailer = FirecrackerJailer(jailer_bin="/usr/bin/jailer", firecracker_bin="/usr/bin/firecracker")
    config = JailerConfig(
        vm_id="vm-test-123",
        exec_file="firecracker",
        uid=1001,
        gid=1001,
        chroot_base_dir="/srv/jailer",
        netns="/var/run/netns/testns",
    )

    cmd = jailer.build_jailer_command(config, socket_path="run/firecracker.sock")
    assert "/usr/bin/jailer" in cmd
    assert "--id" in cmd and "vm-test-123" in cmd
    assert "--uid" in cmd and "1001" in cmd
    assert "--gid" in cmd and "1001" in cmd
    assert "--chroot-base-dir" in cmd and "/srv/jailer" in cmd
    assert "--netns" in cmd and "/var/run/netns/testns" in cmd
    assert "--" in cmd
    assert "--api-sock" in cmd and "run/firecracker.sock" in cmd

    expected_chroot = Path("/srv/jailer/firecracker/vm-test-123/root")
    assert jailer.jail_chroot_dir(config) == expected_chroot


def test_jailer_direct_command():
    jailer = FirecrackerJailer(firecracker_bin="/usr/local/bin/firecracker")
    cmd = jailer.build_direct_command("/tmp/vm.sock")
    assert cmd == ["/usr/local/bin/firecracker", "--api-sock", "/tmp/vm.sock"]


# ==============================================================================
# Rootfs Manager Tests
# ==============================================================================

def test_rootfs_manager_clone_and_cleanup(tmp_path: Path):
    storage = tmp_path / "images"
    mgr = RootfsManager(storage_dir=storage)

    base = tmp_path / "base.ext4"
    base.write_text("master-rootfs-content")

    # Ephemeral clone
    cloned = mgr.clone_ephemeral_rootfs(base, "vm-abc", destination_dir=storage)
    assert cloned.exists()
    assert cloned.name == "vm-abc-rootfs.ext4"
    assert cloned.read_text() == "master-rootfs-content"

    # Cleanup
    mgr.cleanup_rootfs(cloned)
    assert not cloned.exists()


# ==============================================================================
# MicroVM Runtime Protocol & Lifecycle Tests
# ==============================================================================

def test_firecracker_runtime_implements_protocol(tmp_path: Path):
    runtime = FirecrackerMicroVMRuntime(base_socket_dir=tmp_path)
    assert isinstance(runtime, SandboxIsolationBackend)
    assert runtime.backend_type == IsolationBackendType.FIRECRACKER


def test_container_runtime_implements_protocol():
    runtime = ContainerRuntime()
    assert isinstance(runtime, SandboxIsolationBackend)
    assert runtime.backend_type == IsolationBackendType.DOCKER


def test_firecracker_runtime_lifecycle_mocked(tmp_path: Path):
    mgr = RootfsManager(storage_dir=tmp_path / "images")
    runtime = FirecrackerMicroVMRuntime(base_socket_dir=tmp_path, rootfs_mgr=mgr)

    # Mock process spawn
    mock_proc = MagicMock()
    mock_proc.poll.return_value = None
    mock_proc.terminate.return_value = None

    with patch.object(runtime.jailer, "spawn", return_value=mock_proc):
        with patch.object(runtime, "_wait_for_socket", return_value=True):
            with patch.object(FirecrackerClient, "set_machine_config") as mock_mc:
                with patch.object(FirecrackerClient, "set_boot_source") as mock_bs:
                    with patch.object(FirecrackerClient, "attach_drive") as mock_ad:
                        with patch.object(FirecrackerClient, "attach_network_interface") as mock_ani:
                            with patch.object(FirecrackerClient, "start_instance") as mock_start:
                                base = tmp_path / "rootfs.ext4"
                                base.write_text("rootfs")
                                instance_id, port = runtime.run("test-env", image_or_rootfs=str(base))

                                assert instance_id.startswith("fc-test-env-")
                                assert port == 8000
                                assert mock_mc.called
                                assert mock_bs.called
                                assert mock_ad.called
                                assert mock_ani.called
                                assert mock_start.called

                                # Check health
                                with patch.object(FirecrackerClient, "is_alive", return_value=True):
                                    assert runtime.is_healthy(instance_id) is True

                                # Stop
                                runtime.stop(instance_id)
                                assert instance_id not in runtime._active_vms
                                assert mock_proc.terminate.called


def _runtime_with_base(tmp_path: Path):
    mgr = RootfsManager(storage_dir=tmp_path / "images")
    runtime = FirecrackerMicroVMRuntime(base_socket_dir=tmp_path, rootfs_mgr=mgr)
    base = tmp_path / "rootfs.ext4"
    base.write_text("rootfs")
    proc = MagicMock()
    return runtime, base, proc


def test_a_vm_whose_socket_never_comes_up_fails_and_is_cleaned_up(tmp_path: Path):
    runtime, base, proc = _runtime_with_base(tmp_path)

    with patch.object(runtime.jailer, "spawn", return_value=proc), \
            patch.object(runtime, "_wait_for_socket", return_value=False):
        with pytest.raises(RuntimeError, match="did not become ready"):
            runtime.run("test-env", image_or_rootfs=str(base))

    assert runtime._active_vms == {}
    assert proc.terminate.called
    assert list((tmp_path / "images" / "active").iterdir()) == []


def test_a_vm_that_fails_to_boot_raises_and_is_cleaned_up(tmp_path: Path):
    runtime, base, proc = _runtime_with_base(tmp_path)

    with patch.object(runtime.jailer, "spawn", return_value=proc), \
            patch.object(runtime, "_wait_for_socket", return_value=True), \
            patch.object(FirecrackerClient, "set_machine_config", side_effect=FirecrackerAPIError(400, "no kvm")):
        with pytest.raises(RuntimeError, match="failed to boot"):
            runtime.run("test-env", image_or_rootfs=str(base))

    assert runtime._active_vms == {}
    assert proc.terminate.called
    assert list((tmp_path / "images" / "active").iterdir()) == []


def test_cloning_a_missing_base_rootfs_raises_instead_of_making_an_empty_disk(tmp_path: Path):
    mgr = RootfsManager(storage_dir=tmp_path / "images")

    with pytest.raises(FileNotFoundError, match="missing.ext4"):
        mgr.clone_ephemeral_rootfs(tmp_path / "missing.ext4", "vm-abc")

    assert not (tmp_path / "images" / "active" / "vm-abc-rootfs.ext4").exists()


def test_firecracker_runtime_cloned_episode_context(tmp_path: Path):
    runtime = FirecrackerMicroVMRuntime(base_socket_dir=tmp_path)

    with patch.object(runtime, "run", return_value=("fc-ep-1", 8000)):
        with patch.object(runtime, "stop") as mock_stop:
            with runtime.cloned_episode("test-env", "dummy-snap") as (instance_id, port):
                assert instance_id == "fc-ep-1"
                assert port == 8000
            mock_stop.assert_called_once_with("fc-ep-1")


# ==============================================================================
# Runtime Factory & Auto-Detection Tests
# ==============================================================================

def test_runtime_factory_explicit_docker():
    backend = get_sandbox_runtime("docker")
    assert backend.backend_type == IsolationBackendType.DOCKER


def test_runtime_factory_auto_on_mac_without_kvm():
    with patch("forge.envgen.runtime_factory.is_kvm_available", return_value=False):
        backend = get_sandbox_runtime()
        assert backend.backend_type == IsolationBackendType.DOCKER


def test_runtime_factory_auto_uses_docker_even_with_kvm_and_firecracker():
    # The microVM backend cannot run worker episodes yet (no guest networking,
    # no image store), so auto-detection never picks it.
    with patch("forge.envgen.runtime_factory.is_kvm_available", return_value=True):
        with patch("forge.envgen.runtime_factory.is_firecracker_available", return_value=True):
            backend = get_sandbox_runtime()
            assert backend.backend_type == IsolationBackendType.DOCKER


def test_runtime_factory_explicit_firecracker_with_kvm_and_binary():
    with patch("forge.envgen.runtime_factory.is_kvm_available", return_value=True):
        with patch("forge.envgen.runtime_factory.is_firecracker_available", return_value=True):
            backend = get_sandbox_runtime("firecracker")
            assert backend.backend_type == IsolationBackendType.FIRECRACKER


def test_runtime_factory_firecracker_enforced_error_without_kvm():
    with patch("forge.envgen.runtime_factory.is_kvm_available", return_value=False):
        with pytest.raises(RuntimeError, match="not accessible"):
            get_sandbox_runtime("firecracker", allow_fallback=False)


def test_runtime_factory_firecracker_graceful_fallback():
    with patch("forge.envgen.runtime_factory.is_kvm_available", return_value=False):
        backend = get_sandbox_runtime("firecracker", allow_fallback=True)
        assert backend.backend_type == IsolationBackendType.DOCKER
