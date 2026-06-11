import random
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class SimClock:
    _base: int = 1_700_000_000  # fixed epoch, not wall-clock
    _tick: int = 0

    def now(self) -> datetime:
        return datetime.fromtimestamp(self._base + self._tick, tz=timezone.utc)

    def advance(self, seconds: int = 1) -> None:
        self._tick += seconds


@dataclass
class IDGenerator:
    _counters: dict[str, int] = field(default_factory=dict)

    def next(self, prefix: str) -> str:
        count = self._counters.get(prefix, 0)
        self._counters[prefix] = count + 1
        return f"{prefix}_{count:04d}"


@dataclass
class SeededUUIDGenerator:
    """UUIDs drawn from a seeded RNG so identical seeds yield identical IDs."""

    seed: int

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def next(self) -> str:
        return str(uuid.UUID(int=self._rng.getrandbits(128), version=4))


@dataclass
class RuntimeContext:
    seed: int
    actor_id: str = "agent"
    clock: SimClock = field(default_factory=SimClock)
    id_generator: IDGenerator = field(default_factory=IDGenerator)
    rng: random.Random = field(init=False)
    uuid_generator: SeededUUIDGenerator = field(init=False)

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)
        self.uuid_generator = SeededUUIDGenerator(self.seed)
