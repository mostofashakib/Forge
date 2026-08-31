"""Bundle a generated environment's source tree into a standalone, runnable zip.

Produces a self-contained package a user can unzip, install, and run locally —
like cloning a repo — independent of the Forge platform. The bundle carries the
generated source plus a README and a docker-compose so the app runs with one
command.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

# Directories under the env root that are copied wholesale (filtered below).
_SOURCE_TREES = ("app", "custom")
# Individual source files at the env root.
_SOURCE_FILES = ("container_env.py", "reward_fn.py", "state_schema.json")
# At least these must exist for the bundle to be runnable.
_REQUIRED = ("app/main.py",)
# Never bundle runtime artifacts or build caches.
_SKIP_DIRS = {"episodes", "__pycache__"}
_SKIP_SUFFIXES = (".pyc",)


class SourceBundleError(RuntimeError):
    """Raised when an environment is missing or too incomplete to bundle."""


def build_source_bundle(env_dir: Path, env_name: str) -> bytes:
    """Return a ``.zip`` of the generated environment rooted at ``env_name/``.

    Raises :class:`SourceBundleError` when the environment does not exist or is
    missing files required to run it.
    """
    env_dir = Path(env_dir)
    if not env_dir.is_dir():
        raise SourceBundleError(f"Environment {env_name!r} not found")

    missing = [rel for rel in _REQUIRED if not (env_dir / rel).is_file()]
    if missing:
        raise SourceBundleError(
            f"Environment {env_name!r} is incomplete; missing: {', '.join(missing)}"
        )

    root = env_dir.resolve()
    members = _collect_members(env_dir, root)

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for relpath, path in members:
            archive.writestr(f"{env_name}/{relpath}", path.read_bytes())
        has_ui = any(relpath == "app/ui.html" for relpath, _ in members)
        archive.writestr(
            f"{env_name}/README.md", _render_readme(env_name, has_ui=has_ui)
        )
        archive.writestr(f"{env_name}/docker-compose.yml", _render_compose(env_name))
        _write_executable(archive, f"{env_name}/run.sh", _render_run_script(env_name))
    return buffer.getvalue()


def _write_executable(archive: zipfile.ZipFile, arcname: str, content: str) -> None:
    """Add ``content`` to the archive as a regular file with mode 0755."""
    info = zipfile.ZipInfo(arcname)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = (0o100755) << 16  # regular file, rwxr-xr-x
    archive.writestr(info, content)


def _collect_members(env_dir: Path, root: Path) -> list[tuple[str, Path]]:
    members: list[tuple[str, Path]] = []
    for name in _SOURCE_FILES:
        path = env_dir / name
        if _is_safe_file(path, root):
            members.append((name, path))
    for tree in _SOURCE_TREES:
        base = env_dir / tree
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if any(part in _SKIP_DIRS for part in path.relative_to(env_dir).parts):
                continue
            if not _is_safe_file(path, root):
                continue
            members.append((path.relative_to(env_dir).as_posix(), path))
    return members


def _is_safe_file(path: Path, root: Path) -> bool:
    """A regular file that stays within ``root`` once symlinks are resolved."""
    if not path.is_file():
        return False
    if path.suffix in _SKIP_SUFFIXES:
        return False
    try:
        resolved = path.resolve()
    except OSError:
        return False
    return resolved.is_relative_to(root)


def _render_readme(env_name: str, *, has_ui: bool) -> str:
    app_contents = (
        "the runnable FastAPI application and its UI (`main.py`, `ui.html`)"
        if has_ui
        else "the runnable FastAPI application (`main.py`) — this environment is "
             "headless, so it exposes the API only"
    )
    return f"""# {env_name}

A Forge-generated reinforcement-learning environment, exported as a standalone
package. Everything needed to run it locally is in this folder — no dependency on
the Forge platform.

## Prerequisites

- Python 3.12+
- Redis (the app streams telemetry to it) — or just use the bundled Docker setup
- Docker with Compose (optional, for the one-command path)

## Quick start (one command)

```bash
./run.sh
```

`run.sh` does everything: it uses Docker Compose if available (app + Redis),
otherwise it creates a virtualenv, installs dependencies, starts a local Redis,
and serves the app. The app comes up at http://localhost:8000
(health check: `/forge/health`).

## Run with Docker

```bash
REDIS_PASSWORD="$(openssl rand -hex 32)" docker compose up --build
```

## Run locally without Docker

1. Start an authenticated Redis server and point the app at it:

   ```bash
   export REDIS_PASSWORD="$(openssl rand -hex 32)"
   redis-server --daemonize yes --requirepass "$REDIS_PASSWORD"
   export REDIS_URL="redis://:${{REDIS_PASSWORD}}@localhost:6379/0"
   ```

2. Install dependencies:

   ```bash
   cd app
   pip install -r requirements.txt
   ```

3. Serve the app on port 8000:

   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```

## Forge control endpoints

- `GET  /forge/state` — current environment state
- `POST /forge/reset` — restore a fresh, reproducible initial state
- `POST /forge/snapshot` / `POST /forge/restore/{{slot}}` — save and restore state

## Contents

- `app/` — {app_contents}
- `container_env.py` — the gymnasium bridge exposing the app as an RL environment
- `reward_fn.py` — the reward function
- `custom/` — policies and configuration
- `state_schema.json` — the declared state contract (when present)
"""


def _render_run_script(env_name: str) -> str:
    return f"""#!/usr/bin/env bash
# One-command local runner for the {env_name} environment.
# Prefers Docker Compose; otherwise sets up a virtualenv, installs dependencies,
# starts a local Redis, and serves the app on port 8000.
set -euo pipefail
cd "$(dirname "$0")"

if [ -z "${{REDIS_PASSWORD:-}}" ]; then
  if command -v openssl >/dev/null 2>&1; then
    REDIS_PASSWORD="$(openssl rand -hex 32)"
  else
    REDIS_PASSWORD="$(python3 -c 'import secrets; print(secrets.token_hex(32))')"
  fi
  export REDIS_PASSWORD
fi
if [ "${{#REDIS_PASSWORD}}" -lt 32 ] || printf '%s' "$REDIS_PASSWORD" | grep -q '[^0-9A-Fa-f]'; then
  echo "[run] ERROR: REDIS_PASSWORD must contain at least 32 hexadecimal characters." >&2
  exit 1
fi

# --- Docker path: brings up the app + Redis together ----------------------
if command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
  echo "[run] Docker detected — starting app + Redis via Docker Compose…"
  exec docker compose up --build
fi

echo "[run] Docker not available — running locally."

# --- Redis (telemetry backend) --------------------------------------------
if [ -z "${{REDIS_URL:-}}" ]; then
  if command -v redis-server >/dev/null 2>&1; then
    echo "[run] Starting a local redis-server…"
    REDIS_CONFIG="$(mktemp "${{TMPDIR:-/tmp}}/{env_name}-redis.XXXXXX.conf")"
    chmod 600 "$REDIS_CONFIG"
    printf 'bind 127.0.0.1\nrequirepass %s\n' "$REDIS_PASSWORD" >"$REDIS_CONFIG"
    if ! redis-server "$REDIS_CONFIG" --daemonize yes >/dev/null 2>&1; then
      rm -f "$REDIS_CONFIG"
      echo "[run] ERROR: could not start the authenticated local Redis server." >&2
      exit 1
    fi
    rm -f "$REDIS_CONFIG"
    export REDIS_URL="redis://:$REDIS_PASSWORD@localhost:6379/0"
  else
    echo "[run] ERROR: REDIS_URL is not set and redis-server is not installed." >&2
    echo "       Install Redis or Docker, or set REDIS_URL to a running instance." >&2
    exit 1
  fi
fi

# --- Python virtualenv + dependencies -------------------------------------
PYTHON="${{PYTHON:-python3}}"
if [ ! -d .venv ]; then
  echo "[run] Creating virtualenv (.venv)…"
  "$PYTHON" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
echo "[run] Installing dependencies…"
pip install --quiet --upgrade pip
pip install --quiet -r app/requirements.txt

# --- Serve -----------------------------------------------------------------
echo "[run] Serving on http://localhost:8000 (health: /forge/health)…"
cd app
exec uvicorn main:app --host 0.0.0.0 --port 8000
"""


def _render_compose(env_name: str) -> str:
    return f"""# Standalone runner for the {env_name} environment (app + Redis).
# Start everything with: docker compose up --build
services:
  redis:
    image: redis:7-alpine
    command: ["sh", "-c", "exec redis-server --requirepass \"$$REDIS_PASSWORD\""]
    environment:
      REDIS_PASSWORD: ${{REDIS_PASSWORD:?set REDIS_PASSWORD or use ./run.sh}}

  app:
    build: ./app
    environment:
      REDIS_URL: redis://:${{REDIS_PASSWORD}}@redis:6379/0
    ports:
      - "8000:8000"
    depends_on:
      - redis
"""
