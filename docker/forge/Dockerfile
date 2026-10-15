# syntax=docker/dockerfile:1
# Forge backend: one image, two commands (API and worker).
#   API:    uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
#   worker: celery -A backend.app.worker.celery_app worker
# The image holds no secrets. Configuration comes from the environment at run time.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.8.22 /uv /usr/local/bin/uv

# The Docker CLI and buildx: Forge builds and runs sandboxes through the host's
# Docker daemon, reached over the mounted socket.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl gnupg \
    && install -m 0755 -d /etc/apt/keyrings \
    && curl -fsSL https://download.docker.com/linux/debian/gpg -o /etc/apt/keyrings/docker.asc \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/debian $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
        > /etc/apt/sources.list.d/docker.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends docker-ce-cli docker-buildx-plugin sqlite3 \
    && apt-get purge -y gnupg \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH=/app/.venv/bin:$PATH \
    HF_HOME=/models

WORKDIR /app

# Dependencies first, so code changes reuse this layer.
COPY pyproject.toml uv.lock README.md ./
# The uv download cache lives in a build cache mount, never in the image.
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --no-install-project

COPY forge ./forge
COPY backend ./backend
COPY docker/premade ./docker/premade
COPY forge.config.yaml ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev

# A fixed non-root user. Compose adds the host's docker group (DOCKER_GID) so it
# can use the socket. /data, /generated_envs, and /models are volume mount points.
RUN useradd --uid 10001 --create-home --shell /usr/sbin/nologin forge \
    && mkdir -p /data /generated_envs /models \
    && chown forge:forge /data /generated_envs /models
USER forge

EXPOSE 8000
CMD ["uvicorn", "backend.app.main:app", "--host", "127.0.0.1", "--port", "8000"]
