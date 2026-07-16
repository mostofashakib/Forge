from __future__ import annotations

import json
import os
import sys
from pathlib import Path


class SandboxViolation(PermissionError):
    pass


def _inside(path: Path, roots: tuple[Path, ...]) -> bool:
    return any(path.is_relative_to(root) for root in roots)


def _install_audit_hook() -> None:
    read_roots = tuple(
        Path(value).resolve()
        for value in json.loads(os.environ["FORGE_VALIDATION_READ_ROOTS"])
    )
    write_roots = tuple(
        Path(value).resolve()
        for value in json.loads(os.environ["FORGE_VALIDATION_WRITE_ROOTS"])
    )

    def audit(event: str, args: tuple[object, ...]) -> None:
        if event == "open" and args:
            target = args[0]
            if isinstance(target, int):
                return
            path = Path(os.fspath(target)).resolve()
            mode = args[1] if len(args) > 1 else "r"
            is_write = (
                isinstance(mode, str) and any(flag in mode for flag in "wax+")
            ) or (isinstance(mode, int) and mode & (os.O_WRONLY | os.O_RDWR))
            roots = write_roots if is_write else read_roots
            if path == Path("/dev/null") or _inside(path, roots):
                return
            raise SandboxViolation(f"Generated validation code cannot access {path}")
        if event.startswith("socket."):
            raise SandboxViolation("Generated validation code cannot use the network")
        if event in {
            "subprocess.Popen",
            "os.system",
            "os.posix_spawn",
            "os.spawn",
            "ctypes.dlopen",
        }:
            raise SandboxViolation(f"Generated validation code cannot perform {event}")

    sys.addaudithook(audit)


def main() -> int:
    _install_audit_hook()
    import pytest

    return pytest.main(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
