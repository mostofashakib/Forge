"""Live sandbox streams over WebSocket: build progress, event feed, shell, and activity logs."""
from __future__ import annotations

import asyncio
import json
import logging
import os

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool

from backend.app.api._pubsub_relay import stream_channel
from backend.app.api.sandbox_schemas import BUILD_IN_PROGRESS
from backend.app.database import get_session_factory
from backend.app.models import SandboxEnvironment
from forge.settings import redis_url

logger = logging.getLogger("backend.app.api.sandbox")
router = APIRouter(prefix="/api/sandbox")


def _finished_build_message(env_name: str) -> dict | None:
    """The terminal progress message for a build that already ended, else None."""
    with get_session_factory()() as db:
        sandbox = db.get(SandboxEnvironment, env_name)
        if sandbox is None:
            return {"done": True, "error": f"Sandbox '{env_name}' not found"}
        if sandbox.status in BUILD_IN_PROGRESS:
            return None
        if sandbox.status == "error":
            return {"done": True, "error": "Build failed — check the worker logs for details."}
        return {"done": True}


def _container_id(env_name: str) -> str | None:
    """Look up the container in a short session, not one held for a socket's lifetime."""
    with get_session_factory()() as db:
        sandbox = db.get(SandboxEnvironment, env_name)
        return sandbox.container_id if sandbox else None


@router.websocket("/ws/progress/{env_name}")
async def sandbox_progress(websocket: WebSocket, env_name: str):
    """Stream build progress from a Celery worker via Redis pub/sub."""
    await stream_channel(
        websocket,
        redis_url=redis_url(),
        channel=f"forge:progress:{env_name}",
        finished_message=lambda: _finished_build_message(env_name),
        is_final=lambda data: bool(data.get("done")),
        log_tag="progress",
    )


@router.websocket("/ws/feed/{env_name}")
async def sandbox_event_feed(websocket: WebSocket, env_name: str):
    """Tail forge:events:<env_name> Redis Stream and push to frontend."""
    try:
        await websocket.accept()
    except (WebSocketDisconnect, RuntimeError):
        # The React client can mount and immediately unmount the feed before
        # Uvicorn completes the handshake. That is a normal disconnect race,
        # not an application failure.
        logger.debug("[ws:feed] client disconnected before accept — env_name=%s", env_name)
        return

    redis_connection_url = redis_url()
    from forge.envgen.telemetry.stream import StreamConsumer
    consumer = StreamConsumer(redis_url=redis_connection_url, env_name=env_name)
    try:
        async for event in consumer.tail(last_id="$"):
            try:
                await websocket.send_json(event)
            except (WebSocketDisconnect, RuntimeError):
                logger.debug("[ws:feed] client disconnected — env_name=%s", env_name)
                break
    except WebSocketDisconnect:
        logger.debug("[ws:feed] client disconnected — env_name=%s", env_name)
    finally:
        await consumer.close()
        try:
            await websocket.close()
        except (WebSocketDisconnect, RuntimeError):
            pass


@router.websocket("/ws/exec/{env_name}")
async def sandbox_exec(websocket: WebSocket, env_name: str):
    """Bridge WebSocket to docker exec shell via PTY for full interactive terminal support."""
    await websocket.accept()
    container_id = await run_in_threadpool(_container_id, env_name)
    if not container_id:
        await websocket.send_text("Container not running\r\n")
        await websocket.close()
        return

    import fcntl
    import pty as _pty
    import struct
    import subprocess as _subprocess
    import termios

    master_fd, slave_fd = _pty.openpty()
    # Default terminal size: 80 cols × 24 rows
    fcntl.ioctl(slave_fd, termios.TIOCSWINSZ, struct.pack("HHHH", 24, 80, 0, 0))

    proc = _subprocess.Popen(
        ["docker", "exec", "-it", container_id, "/bin/bash"],
        stdin=slave_fd,
        stdout=slave_fd,
        stderr=slave_fd,
        close_fds=True,
    )
    os.close(slave_fd)

    loop = asyncio.get_running_loop()
    closed = asyncio.Event()

    def _read_pty() -> None:
        try:
            data = os.read(master_fd, 4096)
            loop.create_task(_forward(data))
        except OSError:
            closed.set()

    async def _forward(data: bytes) -> None:
        try:
            await websocket.send_text(data.decode(errors="replace"))
        except Exception:
            closed.set()

    loop.add_reader(master_fd, _read_pty)

    async def _ws_reader() -> None:
        try:
            while True:
                text = await websocket.receive_text()
                # Resize message: {"type":"resize","cols":N,"rows":N}
                try:
                    msg = json.loads(text)
                    if msg.get("type") == "resize":
                        cols, rows = int(msg["cols"]), int(msg["rows"])
                        if not (1 <= cols <= 500 and 1 <= rows <= 200):
                            continue
                        fcntl.ioctl(master_fd, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))
                        continue
                except (ValueError, KeyError, TypeError):
                    pass
                try:
                    os.write(master_fd, text.encode())
                except OSError:
                    break
        except WebSocketDisconnect:
            pass
        except Exception:
            logger.exception("[sandbox:exec] terminal input failed for %s", env_name)
        finally:
            closed.set()

    ws_task = asyncio.create_task(_ws_reader())
    try:
        await closed.wait()
    finally:
        ws_task.cancel()
        loop.remove_reader(master_fd)
        try:
            os.close(master_fd)
        except OSError:
            pass
        proc.terminate()
        # The reaping thread finishes even if this handler is cancelled mid-await.
        await asyncio.to_thread(proc.wait)
    try:
        await websocket.close()
    except RuntimeError:
        pass


@router.websocket("/ws/activity/{env_name}")
async def sandbox_activity(websocket: WebSocket, env_name: str):
    """Stream container logs to the Observability panel."""
    await websocket.accept()
    container_id = await run_in_threadpool(_container_id, env_name)
    closed = asyncio.Event()

    async def _docker_logs() -> None:
        if not container_id:
            return
        proc = None
        try:
            proc = await asyncio.create_subprocess_exec(
                "docker", "logs", "--follow", "--timestamps", container_id,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            while not closed.is_set():
                try:
                    raw = await asyncio.wait_for(proc.stdout.readline(), timeout=0.5)
                except asyncio.TimeoutError:
                    continue
                if not raw:
                    break
                line = raw.decode(errors="replace").rstrip()
                # docker --timestamps format: "2024-01-15T10:30:45.123456789Z content"
                ts, _, content = line.partition(" ")
                if not content:
                    content, ts = ts, ""
                try:
                    await websocket.send_json({"type": "log", "ts": ts[:19], "content": content})
                except (WebSocketDisconnect, RuntimeError):
                    break
        except (FileNotFoundError, OSError) as exc:
            logger.warning(
                "[sandbox:activity] Docker log stream failed for %s: %s",
                env_name,
                exc,
            )
        finally:
            if proc is not None:
                try:
                    proc.terminate()
                except ProcessLookupError:
                    logger.debug("[sandbox:activity] Docker log process already exited for %s", env_name)
                await proc.wait()

    async def _ws_watcher() -> None:
        try:
            while True:
                await websocket.receive_text()
        except (WebSocketDisconnect, Exception):
            closed.set()

    tasks = [
        asyncio.create_task(_docker_logs()),
        asyncio.create_task(_ws_watcher()),
    ]
    await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    closed.set()
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    try:
        await websocket.close()
    except Exception:
        pass
