# forge/runtime/tools.py
"""The concrete `ToolProvider` implementations.

Three ways an environment can say what the agent may do:

  SpecToolProvider        a fixed list, for an environment that knows its tools
  CapabilityToolProvider  whatever the environment's capabilities expose
  OpenAPIToolProvider     discovered from a running container's /openapi.json

`OpenAPIToolProvider` holds the discovery that used to be private to
`ContainerEpisodeRunner`, so anything that can reach a container app can now
ask what it offers instead of that knowledge living inside one runner.
"""
from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence

import httpx

from forge.contracts import ToolParam, ToolProvider, ToolSpec

logger = logging.getLogger(__name__)

# Forge's own control plane is not part of the agent's action surface, and
# neither is the browsable app. Excluded from discovery for that reason.
_RESERVED_PREFIX = "/forge/"
_RESERVED_PATHS = {"/ui"}


class SpecToolProvider(ToolProvider):
    """A fixed tool surface, declared up front."""

    def __init__(self, tools: Iterable[ToolSpec]) -> None:
        # Copied, not aliased: the surface an environment advertises must not
        # change because a caller kept mutating the list it passed in.
        self._tools = tuple(tools)

    def tools(self) -> Sequence[ToolSpec]:
        return self._tools


class CapabilityToolProvider(ToolProvider):
    """The tool surface of one or more `Capability` objects.

    Every capability schema already knows how to describe itself as tool specs;
    this just concatenates them in the order given.
    """

    def __init__(self, capabilities: Iterable[object]) -> None:
        self._capabilities = tuple(capabilities)
        for capability in self._capabilities:
            if not hasattr(getattr(capability, "schema", None), "tool_specs"):
                # Better to fail here than to hand the agent an empty surface
                # and let it discover at rollout time that it can do nothing.
                raise TypeError(
                    f"{type(capability).__name__} has no schema.tool_specs(); a capability "
                    "must be able to describe its own tool surface"
                )

    def tools(self) -> Sequence[ToolSpec]:
        return [
            spec
            for capability in self._capabilities
            for spec in capability.schema.tool_specs()
        ]


class OpenAPIToolProvider(ToolProvider):
    """Discovers a container app's action surface from its OpenAPI schema.

    Exposes the same manifest in two shapes. `action_manifest()` returns the
    endpoint dicts the episode runner drives — it needs the endpoint path to
    POST to, which `ToolSpec` does not carry — and `tools()` returns the
    contract form for anything that only speaks `ToolProvider`.

    Discovery is best-effort by design: a container that is not up yet costs an
    empty manifest, never an exception out of the episode.
    """

    def __init__(self, client: httpx.Client, *, timeout: float = 10.0) -> None:
        self._client = client
        self._timeout = timeout
        self._manifest: list[dict] | None = None

    def action_manifest(self) -> list[dict]:
        """Build an action manifest from /openapi.json. Cached after first call."""
        if self._manifest is not None:
            return self._manifest
        try:
            schema = self._client.get("/openapi.json", timeout=self._timeout).json()
            self._manifest = self._actions_from(schema)
            logger.info("[tools] discovered %d action endpoints", len(self._manifest))
        except Exception as exc:
            logger.warning("[tools] could not discover actions: %s", exc)
            self._manifest = []
        return self._manifest

    def tools(self) -> Sequence[ToolSpec]:
        return [self._to_spec(action) for action in self.action_manifest()]

    # ------------------------------------------------------------------

    @staticmethod
    def _actions_from(schema: dict) -> list[dict]:
        components = schema.get("components", {}).get("schemas", {})
        actions: list[dict] = []
        for path, path_item in schema.get("paths", {}).items():
            if path.startswith(_RESERVED_PREFIX) or path in _RESERVED_PATHS:
                continue
            post_op = path_item.get("post")
            if post_op is None:
                continue
            body = post_op.get("requestBody", {})
            request_schema = body.get("content", {}).get("application/json", {}).get("schema", {})
            if "$ref" in request_schema:
                ref_name = request_schema["$ref"].split("/")[-1]
                request_schema = components.get(ref_name, {})
            actions.append({
                "endpoint": path,
                "description": post_op.get("summary") or post_op.get("operationId") or path,
                "request_schema": request_schema,
            })
        return actions

    @staticmethod
    def _to_spec(action: dict) -> ToolSpec:
        request_schema = action.get("request_schema") or {}
        properties = request_schema.get("properties", {})
        required = set(request_schema.get("required", []))
        return ToolSpec(
            name=action["endpoint"],
            description=action.get("description", ""),
            params=[
                ToolParam(
                    name=name,
                    type=spec.get("type", "string"),
                    description=spec.get("description", ""),
                    required=name in required,
                )
                for name, spec in properties.items()
            ],
        )
