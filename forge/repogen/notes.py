"""What the task writer is told about a repo environment.

Checks run in Forge's grading sandbox, a clean BusyBox shell whose PATH holds
only the toolbox, so every check puts the image's /usr/local/bin back first.
"""
from __future__ import annotations

from dataclasses import dataclass

_SAMPLE = 25
_PATH = "export PATH=/usr/local/bin:$PATH"

_PYTHON = {
    "language": "Python",
    "run_one": "python -m pytest -q tests/test_app.py::test_ok",
    "select": "Stable ids read module.path::test_name. Dots in the module path are folders, "
              "so tests.test_app::test_ok is tests/test_app.py::test_ok. A final CamelCase part is a "
              "test class, so tests.test_signer.TestSigner::test_sign is "
              "tests/test_signer.py::TestSigner::test_sign.",
    "stub": "raise NotImplementedError",
}
_NODE = {
    "language": "TypeScript/JavaScript",
    "run_one": "npx jest src/math.test.ts -t 'add sums'",
    "select": "Stable ids read file::full test name. Select one with the runner's file argument "
              "and -t '<full test name>' (vitest: npx vitest run <file> -t '<name>').",
    "stub": "throw new Error('not implemented')",
}


@dataclass(frozen=True)
class RepoMeta:
    full_name: str
    sha: str
    toolchain: str
    test_command: str
    stable_tests: list[str]


def repo_notes(meta: RepoMeta) -> str:
    style = _PYTHON if meta.toolchain == "python" else _NODE
    sample = "\n".join(f"  {test}" for test in meta.stable_tests[:_SAMPLE])
    return f"""\
Repository {meta.full_name} at commit {meta.sha[:12]}, checked out at /repo ({style['language']}).
Shell as root, no network, clock frozen. Dependencies are already installed.
{len(meta.stable_tests)} tests pass reliably. Tasks may rely only on these. A sample:
{sample}
{style['select']}

Write coding tasks against this code:
  - Plant a bug: seed.setup edits one source file (sed or a heredoc) so some stable
    tests fail. The agent finds and fixes it.
  - Stub a function: seed.setup replaces a function body with `{style['stub']}`.
    The agent restores the behavior.
Never edit test files in seed.setup or in golden steps. Golden steps are shell
commands that repair the source.

Checks are kind "shell" and run in a clean shell. Start every check with
`cd /repo && {_PATH} && ` and then run the specific stable tests the change
breaks, for example: `cd /repo && {_PATH} && {style['run_one']}`.
Each check must fail after setup and pass after the golden steps."""
