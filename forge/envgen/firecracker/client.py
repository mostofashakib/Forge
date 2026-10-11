"""Client for Firecracker MicroVM REST API over Unix Domain Sockets (UDS).

Firecracker serves a REST API exclusively over an AF_UNIX domain socket.
This client communicates with Firecracker instances to configure vCPUs, memory,
boot parameters, drives, network interfaces, and lifecycle actions.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class FirecrackerAPIError(Exception):
    """Raised when the Firecracker REST API returns an error response."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(f"Firecracker API error {status_code}: {message}")
        self.status_code = status_code
        self.message = message


class FirecrackerClient:
    """Synchronous REST client for controlling a single Firecracker microVM instance."""

    def __init__(self, socket_path: Path | str, timeout: float = 5.0) -> None:
        self.socket_path = Path(socket_path)
        self.timeout = timeout
        self._http_client: httpx.Client | None = None

    def _get_client(self) -> httpx.Client:
        """Create or return an active HTTP client bound to the microVM's Unix domain socket."""
        if self._http_client is None or self._http_client.is_closed:
            transport = httpx.HTTPTransport(uds=str(self.socket_path))
            self._http_client = httpx.Client(
                transport=transport,
                base_url="http://localhost",
                timeout=self.timeout,
            )
        return self._http_client

    def close(self) -> None:
        """Close underlying HTTP transport and client session."""
        if self._http_client is not None and not self._http_client.is_closed:
            self._http_client.close()
            self._http_client = None

    def _request(self, method: str, path: str, json_data: dict[str, Any] | None = None) -> dict[str, Any]:
        """Send an HTTP request over UDS and handle errors."""
        try:
            client = self._get_client()
            response = client.request(method, path, json=json_data)
        except Exception as exc:
            logger.debug("Failed Firecracker request %s %s over %s: %s", method, path, self.socket_path, exc)
            raise ConnectionError(f"Could not connect to Firecracker socket {self.socket_path}: {exc}") from exc

        # Firecracker returns 204 No Content on successful PUT/PATCH actions
        if response.status_code in (200, 201, 204):
            return response.json() if response.content else {}

        error_msg = response.text
        try:
            parsed = response.json()
            if "fault_message" in parsed:
                error_msg = parsed["fault_message"]
        except Exception:
            pass

        raise FirecrackerAPIError(response.status_code, error_msg)

    def set_machine_config(
        self,
        vcpu_count: int = 2,
        mem_size_mib: int = 1024,
        smt: bool = False,
        track_dirty_pages: bool = False,
    ) -> dict[str, Any]:
        """Configure microVM CPU count and memory size."""
        payload = {
            "vcpu_count": vcpu_count,
            "mem_size_mib": mem_size_mib,
            "smt": smt,
            "track_dirty_pages": track_dirty_pages,
        }
        return self._request("PUT", "/machine-config", payload)

    def get_machine_config(self) -> dict[str, Any]:
        """Retrieve current microVM machine configuration."""
        return self._request("GET", "/machine-config")

    def set_boot_source(
        self,
        kernel_image_path: str,
        boot_args: str,
        initrd_path: str | None = None,
    ) -> dict[str, Any]:
        """Configure microVM uncompressed kernel image and command-line arguments."""
        payload: dict[str, Any] = {
            "kernel_image_path": str(kernel_image_path),
            "boot_args": boot_args,
        }
        if initrd_path:
            payload["initrd_path"] = str(initrd_path)
        return self._request("PUT", "/boot-source", payload)

    def attach_drive(
        self,
        drive_id: str,
        path_on_host: str,
        is_root_device: bool = True,
        is_read_only: bool = False,
        partuuid: str | None = None,
    ) -> dict[str, Any]:
        """Attach a block device drive (e.g. rootfs ext4 image) to the microVM."""
        payload: dict[str, Any] = {
            "drive_id": drive_id,
            "path_on_host": str(path_on_host),
            "is_root_device": is_root_device,
            "is_read_only": is_read_only,
        }
        if partuuid:
            payload["partuuid"] = partuuid
        return self._request("PUT", f"/drives/{drive_id}", payload)

    def attach_network_interface(
        self,
        iface_id: str,
        host_dev_name: str,
        guest_mac: str | None = None,
    ) -> dict[str, Any]:
        """Attach a host TAP network interface to the microVM."""
        payload: dict[str, Any] = {
            "iface_id": iface_id,
            "host_dev_name": host_dev_name,
        }
        if guest_mac:
            payload["guest_mac"] = guest_mac
        return self._request("PUT", f"/network-interfaces/{iface_id}", payload)

    def start_instance(self) -> dict[str, Any]:
        """Issue the InstanceStart command to boot the guest microVM."""
        return self._request("PUT", "/actions", {"action_type": "InstanceStart"})

    def send_ctrl_alt_del(self) -> dict[str, Any]:
        """Send graceful reboot/shutdown ACPI signal."""
        return self._request("PUT", "/actions", {"action_type": "SendCtrlAltDel"})

    def pause_instance(self) -> dict[str, Any]:
        """Pause execution of the microVM vCPUs."""
        return self._request("PATCH", "/vm", {"state": "Paused"})

    def resume_instance(self) -> dict[str, Any]:
        """Resume execution of a paused microVM."""
        return self._request("PATCH", "/vm", {"state": "Resumed"})

    def create_snapshot(
        self,
        mem_file_path: str,
        snapshot_path: str,
        snapshot_type: str = "Full",
    ) -> dict[str, Any]:
        """Create a full microVM state snapshot for sub-millisecond restoration."""
        payload = {
            "mem_file_path": str(mem_file_path),
            "snapshot_path": str(snapshot_path),
            "snapshot_type": snapshot_type,
        }
        return self._request("PUT", "/snapshot/create", payload)

    def load_snapshot(
        self,
        mem_file_path: str,
        snapshot_path: str,
        resume_vm: bool = True,
    ) -> dict[str, Any]:
        """Restore microVM memory and CPU state from snapshot."""
        payload = {
            "mem_backend": {"backend_path": str(mem_file_path), "backend_type": "File"},
            "snapshot_path": str(snapshot_path),
            "resume_vm": resume_vm,
        }
        return self._request("PUT", "/snapshot/load", payload)

    def is_alive(self) -> bool:
        """Check whether the microVM socket is responsive."""
        if not self.socket_path.exists():
            return False
        try:
            self.get_machine_config()
            return True
        except Exception:
            return False
