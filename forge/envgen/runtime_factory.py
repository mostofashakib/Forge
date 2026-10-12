"""Runtime factory for sandbox isolation backends.

Selects between hardware-isolated Firecracker microVMs and Docker container
sandboxes based on platform virtualization capabilities (/dev/kvm) and configuration.
"""
from __future__ import annotations

import logging
import os
import shutil
from typing import TYPE_CHECKING

from forge.contracts.sandbox import IsolationBackendType, SandboxIsolationBackend

if TYPE_CHECKING:
    from forge.envgen.container import ContainerRuntime
    from forge.envgen.firecracker.runtime import FirecrackerMicroVMRuntime

logger = logging.getLogger(__name__)


def is_kvm_available() -> bool:
    """Check whether Linux Kernel-based Virtual Machine (/dev/kvm) is accessible."""
    kvm_path = "/dev/kvm"
    return os.path.exists(kvm_path) and os.access(kvm_path, os.R_OK | os.W_OK)


def is_firecracker_available() -> bool:
    """Check whether the Firecracker binary is installed on the host system."""
    return shutil.which("firecracker") is not None or os.path.exists("/usr/bin/firecracker")


def get_sandbox_runtime(
    backend_type: IsolationBackendType | str | None = None,
    *,
    allow_fallback: bool = True,
) -> SandboxIsolationBackend:
    """Resolve and instantiate the appropriate sandbox isolation runtime.

    Precedence:
    1. Explicit `backend_type` argument
    2. Environment variable `FORGE_SANDBOX_RUNTIME` ("firecracker", "docker", "auto")
    3. Auto-detection: Docker. Firecracker runs only when requested explicitly.
    """
    configured = (
        str(backend_type).lower()
        if backend_type is not None
        else os.getenv("FORGE_SANDBOX_RUNTIME", "auto").lower()
    )

    if configured in ("firecracker", IsolationBackendType.FIRECRACKER):
        kvm = is_kvm_available()
        fc = is_firecracker_available()
        if not kvm and not allow_fallback:
            raise RuntimeError(
                "Firecracker microVM isolation requested, but /dev/kvm is not accessible on this system."
            )
        if kvm and fc:
            logger.info("Using hardware-isolated Firecracker microVM sandbox backend")
            from forge.envgen.firecracker.runtime import FirecrackerMicroVMRuntime
            return FirecrackerMicroVMRuntime()
        elif allow_fallback:
            logger.warning(
                "Firecracker requested but /dev/kvm or binary missing; falling back to Docker container runtime"
            )
            from forge.envgen.container import ContainerRuntime
            return ContainerRuntime()
        else:
            from forge.envgen.firecracker.runtime import FirecrackerMicroVMRuntime
            return FirecrackerMicroVMRuntime()

    if configured in ("docker", IsolationBackendType.DOCKER):
        from forge.envgen.container import ContainerRuntime
        return ContainerRuntime()

    # Auto-detection mode. The microVM backend has no guest networking or image
    # store yet, so it cannot run worker episodes and is never picked here.
    from forge.envgen.container import ContainerRuntime
    return ContainerRuntime()
