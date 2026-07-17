from __future__ import annotations

import errno


def is_docker_daemon_unavailable(error: object, seen: set[int] | None = None) -> bool:
    """Return whether a Docker SDK error wraps a missing/refused daemon socket."""
    if seen is None:
        seen = set()

    error_id = id(error)
    if error_id in seen:
        return False
    seen.add(error_id)

    if isinstance(error, OSError) and error.errno in {errno.ENOENT, errno.ECONNREFUSED}:
        return True
    if isinstance(error, BaseException):
        nested_errors = [error.__cause__, error.__context__, *error.args]
        return any(
            nested is not None and is_docker_daemon_unavailable(nested, seen)
            for nested in nested_errors
        )
    if isinstance(error, (tuple, list)):
        return any(is_docker_daemon_unavailable(item, seen) for item in error)
    return False
