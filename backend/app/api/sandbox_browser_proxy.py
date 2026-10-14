"""Reach a browser sandbox's desktop through the API, by sandbox name.

The desktop (Selkies) loads its assets and opens its WebSocket at paths relative
to the page, so this proxy carries both HTTP and WebSocket traffic under
/api/sandbox/{env_name}/browser/. It only ever reaches the port the database
records for that sandbox, so it cannot be pointed at another service.
"""
from __future__ import annotations

import asyncio
import logging

import httpx
import websockets
from fastapi import APIRouter, HTTPException, Request, WebSocket
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask

from backend.app.models import SandboxEnvironment
from forge.settings import sandbox_url

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/sandbox", tags=["sandbox-browser"])

# Headers that describe one hop of the connection, not the message.
_HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailer", "transfer-encoding", "upgrade", "host", "content-length",
}
_UPSTREAM_TIMEOUT = httpx.Timeout(30.0, connect=5.0)


def _browser_port(env_name: str) -> int:
    from backend.app.database import get_session_factory

    with get_session_factory()() as db:
        sandbox = db.get(SandboxEnvironment, env_name)
        if sandbox is None:
            raise HTTPException(status_code=404, detail=f"Sandbox {env_name!r} not found")
        if sandbox.env_type != "browser":
            raise HTTPException(status_code=409, detail=f"{env_name!r} is not a browser sandbox")
        if sandbox.status != "running" or sandbox.container_port is None:
            raise HTTPException(status_code=409, detail=f"{env_name!r} is not running")
        return sandbox.container_port


def _upstream_path(path: str, query: str) -> str:
    if any(segment in (".", "..") for segment in path.split("/")):
        raise HTTPException(status_code=400, detail="sandbox path segment may not be . or ..")
    return f"/{path}" + (f"?{query}" if query else "")


@router.api_route("/{env_name}/browser/{path:path}", methods=["GET", "HEAD", "POST", "PUT", "PATCH", "DELETE"])
async def proxy_browser_http(env_name: str, path: str, request: Request) -> StreamingResponse:
    port = await asyncio.to_thread(_browser_port, env_name)
    url = sandbox_url(port) + _upstream_path(path, request.url.query)
    headers = {key: value for key, value in request.headers.items() if key.lower() not in _HOP_BY_HOP}
    headers["x-forge-actor"] = "user"

    client = httpx.AsyncClient(timeout=_UPSTREAM_TIMEOUT)
    try:
        upstream = await client.send(
            client.build_request(request.method, url, headers=headers, content=await request.body()),
            stream=True,
        )
    except httpx.HTTPError as exc:
        await client.aclose()
        raise HTTPException(status_code=502, detail=f"browser sandbox {env_name!r} unreachable: {exc}") from exc

    async def close() -> None:
        await upstream.aclose()
        await client.aclose()

    return StreamingResponse(
        upstream.aiter_raw(),
        status_code=upstream.status_code,
        headers={key: value for key, value in upstream.headers.items() if key.lower() not in _HOP_BY_HOP - {"content-length"}},
        background=BackgroundTask(close),
    )


@router.websocket("/{env_name}/browser/{path:path}")
async def proxy_browser_websocket(websocket: WebSocket, env_name: str, path: str) -> None:
    try:
        port = await asyncio.to_thread(_browser_port, env_name)
        target = sandbox_url(port, scheme="ws") + _upstream_path(path, websocket.url.query)
    except HTTPException as exc:
        await websocket.close(code=4000 + exc.status_code, reason=str(exc.detail)[:120])
        return

    try:
        upstream = await websockets.connect(
            target, subprotocols=websocket.scope.get("subprotocols") or None, max_size=None, open_timeout=5,
        )
    except (OSError, websockets.WebSocketException) as exc:
        logger.warning("[browser-proxy] %s unreachable at %s: %s", env_name, target, exc)
        await websocket.close(code=1011, reason="browser sandbox unreachable")
        return

    await websocket.accept(subprotocol=upstream.subprotocol)

    async def client_to_upstream() -> None:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                return
            await upstream.send(message["bytes"] if message.get("bytes") is not None else message["text"])

    async def upstream_to_client() -> None:
        async for message in upstream:
            if isinstance(message, bytes):
                await websocket.send_bytes(message)
            else:
                await websocket.send_text(message)

    pumps = [asyncio.create_task(client_to_upstream()), asyncio.create_task(upstream_to_client())]
    try:
        await asyncio.wait(pumps, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for pump in pumps:
            pump.cancel()
        await upstream.close()
        try:
            await websocket.close()
        except RuntimeError:
            pass  # the browser already closed its side
