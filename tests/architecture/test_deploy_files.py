"""The deploy files agree with each other: every variable has a documented home,
and wiring-check.sh tests exactly the connections deploy/README.md lists."""
from __future__ import annotations

import re
from pathlib import Path

DEPLOY = Path(__file__).resolve().parents[2] / "deploy"
# ${NAME} and ${NAME:-default}, but not $${NAME}, which Compose passes to the container's shell.
VARIABLE = re.compile(r"(?<!\$)\$\{([A-Z][A-Z0-9_]*)")
WIRING_ID = re.compile(r"\b([WX]\d+)\b")


def _example_names() -> set[str]:
    text = (DEPLOY / ".env.production.example").read_text()
    return {line.split("=", 1)[0] for line in text.splitlines() if re.match(r"^[A-Z][A-Z0-9_]*=", line)}


def _read_variables(*names: str) -> dict[str, set[str]]:
    return {name: set(VARIABLE.findall((DEPLOY / name).read_text())) for name in names}


def test_every_variable_the_stack_reads_is_in_the_example_env_file():
    read = _read_variables("docker-compose.yml", "litestream.yml", "backup.sh", "heartbeat.sh")
    documented = _example_names()

    missing = {name: sorted(found - documented) for name, found in read.items() if found - documented}

    assert missing == {}


def test_the_pattern_skips_escaped_and_lowercase_references():
    assert VARIABLE.findall("${FORGE_ENV} ${FORGE_TAILNET_HOST:-localhost} $${FORGE_REDIS_PASSWORD} ${1:-}") == [
        "FORGE_ENV", "FORGE_TAILNET_HOST",
    ]


def test_wiring_check_tests_exactly_the_connections_in_the_readme():
    readme_section = (DEPLOY / "README.md").read_text().split("## The wiring map", 1)[1].split("\n## ", 1)[0]
    in_readme = set(WIRING_ID.findall(readme_section))
    in_script = set(re.findall(r"\b(?:pass|fail|skip|check) ([WX]\d+)\b", (DEPLOY / "wiring-check.sh").read_text()))

    assert in_readme == {f"W{n}" for n in range(1, 20)} | {f"X{n}" for n in range(1, 6)}
    assert in_script == in_readme
