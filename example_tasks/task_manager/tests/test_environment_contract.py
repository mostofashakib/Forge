#!/usr/bin/env python3
"""The task as a package: what it declares, what it ships, and what it needs.

Most of these are one line in a config that a later edit could remove without
anyone noticing until a run failed in a confusing way.
"""

from __future__ import annotations

import ast
import re
import sys
import tomllib
from pathlib import Path

FAILURES: list[str] = []
SKIPPED = 0
ROOT = Path(__file__).resolve().parents[1]

#: Authoring-time guards. Harbor uploads only `tests/` to /tests, so task.toml,
#: the Dockerfile and the instruction are not in the graded image -- and a suite
#: that failed there would report the environment broken over files that were
#: never meant to ship. They run from a checkout, where they are the point, and
#: skip inside the image.
IN_REPOSITORY = (ROOT / "task.toml").is_file()


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  ok    {name}")
    else:
        FAILURES.append(f"{name}: {detail}")
        print(f"  FAIL  {name}  {detail}")


def needs_checkout() -> bool:
    """True when this check cannot run, having said so."""
    global SKIPPED
    if IN_REPOSITORY:
        return False
    SKIPPED += 1
    print("  skip  not a checkout; this guard runs at authoring time")
    return True


def config() -> dict:
    return tomllib.loads((ROOT / "task.toml").read_text(encoding="utf-8"))


def test_the_task_declares_itself() -> None:
    if needs_checkout():
        return
    data = config()
    check("schema_version is 1.4", data["schema_version"] == "1.4", str(data.get("schema_version")))
    check("the task is namespaced", data["task"]["name"] == "forge/task-manager-reassignment",
          data["task"]["name"])
    check("it has a description", len(data["task"]["description"]) > 40)
    check("and says how hard it is", bool(data["metadata"]["difficulty_explanation"]))


def test_the_workspace_is_handed_over_as_an_mcp_server() -> None:
    """The agent is given the tools; it is not told a command name."""
    if needs_checkout():
        return
    servers = config()["environment"]["mcp_servers"]
    check("exactly one server is registered", len(servers) == 1, str(len(servers)))
    server = servers[0]
    check("named for the workspace", server["name"] == "tasks")
    check("over stdio, because the environment has no network",
          server["transport"] == "stdio", server["transport"])
    check("pointing at the client wrapper",
          server["command"] == "/usr/local/bin/tasks-mcp", server["command"])


def test_the_instruction_names_no_tool() -> None:
    """Discovering the workspace is the harness's job, not the prompt's."""
    if needs_checkout():
        return
    instruction = (ROOT / "instruction.md").read_text(encoding="utf-8")
    for leak in ("tasks list_tasks", "python3", "agent.sock", "/app", "CLI", "MCP"):
        check(f"the prompt does not mention {leak!r}", leak not in instruction)
    check("but it does name the one tool that records the answer",
          "submit_handover_report" in instruction)


def test_the_environment_is_sealed() -> None:
    if needs_checkout():
        return
    data = config()
    check("the world has no network", data["environment"]["network_mode"] == "no-network")
    check("the agent phase is allowlisted",
          data["agent"]["network_mode"] == "allowlist", str(data["agent"].get("network_mode")))
    check("to one host", data["agent"]["allowed_hosts"] == ["openrouter.ai"])
    check("the agent runs unprivileged", data["agent"]["user"] == "agent")
    check("the verifier runs as root", data["verifier"]["user"] == "root")


def test_the_graded_state_is_collected_from_the_world() -> None:
    if needs_checkout():
        return
    collect = config()["verifier"]["collect"]
    check("a collection step is declared", len(collect) == 1)
    step = collect[0]
    check("it runs in the world's service", step["service"] == "tasks")
    check("through the privileged CLI", "admin_cli export_state" in step["command"])
    artifacts = config()["artifacts"]
    check("and the export is kept as an artifact",
          any(isinstance(a, dict) and "state-export.json" in a.get("source", "")
              for a in artifacts), str(artifacts))


def test_the_task_has_no_python_dependencies() -> None:
    """The graded container is built without pip.

    A verifier that dies importing its own config reports zero for reasons that
    have nothing to do with the agent, so this is a constraint rather than an
    accident.
    """
    if needs_checkout():
        return
    stdlib = set(sys.stdlib_module_names)
    # Modules of this task, resolved from PYTHONPATH rather than installed.
    local = {"task_sim", "verifiers", "agent", "workspace", "reward_episodes",
             "reward_matrix", "dynamic_scenario", "rl_env", "grader_audit"}
    # The one exception, and it is confined to one file. `agent/harbor_agent.py`
    # is the translation between Harbor's agent protocol and this package's own
    # loop, so it necessarily imports Harbor -- and it runs on the machine that
    # starts the run, never in the graded image. Every other module in `agent`
    # is held to the same stdlib-only rule as the rest, which is what lets the
    # loop be driven from a plain `python3 -m agent` with nothing installed.
    # The suite covering it gets the same exemption for the same reason, and
    # only inside a try/ImportError: it validates the trajectory against
    # Harbor's own ATIF schema when Harbor is installed, and falls back to its
    # structural checks when it is not.
    exempt = {
        "agent/harbor_agent.py": {"harbor"},
        "tests/test_agent_adapters.py": {"harbor"},
    }
    offenders: list[str] = []
    for directory in ("environment", "verifiers", "tests", "tools", "agent"):
        for path in sorted((ROOT / directory).rglob("*.py")):
            allowed = local | exempt.get(str(path.relative_to(ROOT)), set())
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                names: list[str] = []
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                    names = [node.module]
                for name in names:
                    root = name.split(".")[0]
                    if root not in stdlib and root not in allowed:
                        offenders.append(f"{path.relative_to(ROOT)}: {name}")
    check("nothing imports a third-party package", not offenders, str(sorted(set(offenders))))
    for absent in ("requirements.txt", "pyproject.toml", "uv.lock", "setup.py"):
        check(f"no {absent}", not (ROOT / absent).exists())


def test_the_expected_files_are_all_here() -> None:
    if needs_checkout():
        return
    required = [
        "README.md", "instruction.md", "task.toml",
        "environment/Dockerfile", "environment/docker-compose.yaml", "environment/rl_env.py",
        "environment/bin/tasks", "environment/bin/tasks-mcp",
        "environment/bin/tasks-admin", "environment/bin/tasksd-entrypoint",
        "solution/solve.sh", "tests/test.sh",
        "verifiers/__init__.py", "verifiers/experiment.yaml",
        "verifiers/contracts/reassignment.py",
    ]
    for relative in required:
        check(f"{relative} exists", (ROOT / relative).is_file())


def test_the_scripts_are_runnable() -> None:
    if needs_checkout():
        return
    for relative in ("solution/solve.sh", "tests/test.sh",
                     "environment/bin/tasks", "environment/bin/tasks-mcp",
                     "environment/bin/tasks-admin", "environment/bin/tasksd-entrypoint"):
        path = ROOT / relative
        check(f"{relative} starts with a shebang",
              path.read_text(encoding="utf-8").startswith("#!"))


def test_test_sh_runs_every_suite_and_always_writes_a_reward() -> None:
    if needs_checkout():
        return
    script = (ROOT / "tests" / "test.sh").read_text(encoding="utf-8")
    listed = re.search(r"for suite in (.+); do", script)
    check("the suite list is in the script", listed is not None)
    named = set(listed.group(1).split()) if listed else set()
    on_disk = {
        path.stem for path in (ROOT / "tests").glob("test_*.py")
        if path.stem != "test_reassignment_readiness"
    }
    check("every suite on disk is run", on_disk - named == set(), str(on_disk - named))
    check("and every suite it names exists", named - on_disk == set(), str(named - on_disk))
    # A missing reward file reaches Harbor as RewardFileNotFoundError, which says
    # nothing about what went wrong.
    check("it does not exit early on a failing self-test", "set -e\n" not in script)
    check("the graded outcome always runs", "test_reassignment_readiness.py" in script)
    check("with the self-test status passed through", "TASK_SELFTEST_STATUS" in script)


def test_the_experiment_config_is_readable_without_yaml() -> None:
    import verifiers as verifiers_package
    from verifiers.presets import REWARD_PRESETS, load_experiment

    # Resolved from the installed package, not from the repo root: in the graded
    # image the verifier stack lives at /opt/grading/verifiers, and this is the
    # same way the readiness entry point finds its own config.
    path = Path(verifiers_package.__file__).resolve().parent / "experiment.yaml"
    check("the config ships beside the package", path.is_file(), str(path))
    config_data = load_experiment(path)
    check("it names a preset", "reward_preset" in config_data, str(config_data))
    check("and the preset exists", config_data["reward_preset"] in REWARD_PRESETS,
          str(config_data.get("reward_preset")))


def test_the_build_context_excludes_what_must_not_ship() -> None:
    if needs_checkout():
        return
    ignored = (ROOT / ".dockerignore").read_text(encoding="utf-8").split()
    for entry in ("instruction.md", "solution", "tools", "jobs", "agent"):
        check(f"{entry} is kept out of the image", entry in ignored, str(ignored))


def main() -> int:
    print(__doc__)
    for test in (
        test_the_task_declares_itself,
        test_the_workspace_is_handed_over_as_an_mcp_server,
        test_the_instruction_names_no_tool,
        test_the_environment_is_sealed,
        test_the_graded_state_is_collected_from_the_world,
        test_the_task_has_no_python_dependencies,
        test_the_expected_files_are_all_here,
        test_the_scripts_are_runnable,
        test_test_sh_runs_every_suite_and_always_writes_a_reward,
        test_the_experiment_config_is_readable_without_yaml,
        test_the_build_context_excludes_what_must_not_ship,
    ):
        print(f"\n{test.__name__}")
        test()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} failure(s):")
        for failure in FAILURES:
            print(f"  - {failure}")
        return 1
    suffix = f" ({SKIPPED} authoring-time guard(s) skipped inside the image)" if SKIPPED else ""
    print(f"all environment-contract checks passed{suffix}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
