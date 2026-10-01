"""The open-source models the local Ollama server has pulled, for the Settings page."""
from __future__ import annotations

import httpx

from forge.grading_provenance import model_family

_TIMEOUT_S = 3.0


class OllamaUnavailable(RuntimeError):
    """The Ollama server could not be reached or answered with an error."""


def list_ollama_models(base_url: str, *, client: httpx.Client | None = None) -> list[dict]:
    """Every pulled model, sorted by name, with its family, size, and whether it runs in Ollama's cloud."""
    url = f"{base_url.rstrip('/')}/api/tags"
    owned = client is None
    http = client or httpx.Client(timeout=_TIMEOUT_S)
    try:
        response = http.get(url)
        response.raise_for_status()
        models = response.json().get("models", [])
    except (httpx.HTTPError, ValueError) as exc:
        raise OllamaUnavailable(f"could not list models from {url}: {exc}") from exc
    finally:
        if owned:
            http.close()
    return sorted(
        (
            {
                "name": model["name"],
                "family": model_family(model["name"]),
                "parameters": (model.get("details") or {}).get("parameter_size"),
                # Ollama runs ":cloud" models on its hosted service, not on this machine.
                "cloud": model["name"].endswith(":cloud"),
            }
            for model in models
            if model.get("name")
        ),
        key=lambda model: model["name"],
    )
