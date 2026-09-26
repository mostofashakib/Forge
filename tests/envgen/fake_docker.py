"""A small in-memory Docker daemon for ContainerRuntime tests.

It models what the runtime relies on: containers found by id or name, name
conflicts, published ports assigned on run, start/stop/remove, label
filtering, and networks that containers can join.
"""
from __future__ import annotations

import itertools

import docker.errors


class FakeContainer:
    def __init__(self, daemon: "FakeDocker", cid: str, kwargs: dict) -> None:
        self._daemon = daemon
        self.id = cid
        self.name = kwargs["name"]
        self.labels = dict(kwargs.get("labels") or {})
        self.kwargs = kwargs
        self.status = "running"
        self.networks: list[str] = [kwargs.get("network") or "bridge"]
        self.ports = {
            key: [{"HostIp": "127.0.0.1", "HostPort": str(next(daemon.port_counter))}]
            for key in (kwargs.get("ports") or {})
        }
        self.attrs = {"RestartCount": 0}

    def reload(self) -> None:
        pass

    def start(self) -> None:
        if self.name in self._daemon.refuse_start:
            raise docker.errors.APIError("refuses to start")
        self.status = "running"

    def stop(self, timeout: int = 10) -> None:
        self.status = "exited"

    def rename(self, name: str) -> None:
        self.name = name

    def remove(self, force: bool = False) -> None:
        self._daemon.removed.append(self.name)
        self._daemon.containers._by_id.pop(self.id, None)


class FakeContainers:
    def __init__(self, daemon: "FakeDocker") -> None:
        self._daemon = daemon
        self._by_id: dict[str, FakeContainer] = {}

    def get(self, key: str) -> FakeContainer:
        for container in self._by_id.values():
            if key in (container.id, container.name):
                return container
        raise docker.errors.NotFound(f"no such container: {key}")

    def run(self, **kwargs) -> FakeContainer:
        if kwargs.get("image") in self._daemon.missing_images:
            raise docker.errors.ImageNotFound(kwargs["image"])
        if any(c.name == kwargs["name"] for c in self._by_id.values()):
            raise docker.errors.APIError(f"409 Conflict: {kwargs['name']}")
        container = FakeContainer(self._daemon, f"id-{kwargs['name']}-{next(self._daemon.id_counter)}", kwargs)
        self._by_id[container.id] = container
        return container

    def list(self, filters: dict | None = None) -> list[FakeContainer]:
        wanted = (filters or {}).get("label")
        key, _, value = (wanted or "").partition("=")
        return [
            c for c in self._by_id.values()
            if c.status == "running" and (not wanted or c.labels.get(key) == value)
        ]


class FakeNetwork:
    def __init__(self, daemon: "FakeDocker", name: str, kwargs: dict) -> None:
        self._daemon = daemon
        self.name = name
        self.kwargs = kwargs

    def connect(self, container) -> None:
        self._daemon.containers.get(container.id).networks.append(self.name)

    def remove(self) -> None:
        self._daemon.networks._by_name.pop(self.name, None)


class FakeNetworks:
    def __init__(self, daemon: "FakeDocker") -> None:
        self._daemon = daemon
        self._by_name: dict[str, FakeNetwork] = {}

    def get(self, name: str) -> FakeNetwork:
        if name not in self._by_name:
            raise docker.errors.NotFound(f"no such network: {name}")
        return self._by_name[name]

    def create(self, name: str, **kwargs) -> FakeNetwork:
        network = FakeNetwork(self._daemon, name, kwargs)
        self._by_name[name] = network
        return network


class FakeDocker:
    def __init__(self) -> None:
        self.port_counter = itertools.count(41000)
        self.id_counter = itertools.count(1)
        self.containers = FakeContainers(self)
        self.networks = FakeNetworks(self)
        self.images = type("Images", (), {"remove": staticmethod(lambda *a, **k: None)})()
        self.removed: list[str] = []
        self.refuse_start: set[str] = set()
        self.missing_images: set[str] = set()

    def named(self, name: str) -> FakeContainer:
        return self.containers.get(name)

    def close(self) -> None:
        pass
