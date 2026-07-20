"""Static audit for the generated-environment authoring contract (TASKS.md #9).

Two invariants are enforced against generated backend source:

  1. State is centralized in a single state-management class exposing both
     ``reset_state()`` and ``seed_state(seed)``.
  2. Every route handler returns a typed dict — never a bare string or f-string,
     on the success path or the error path.
"""

from __future__ import annotations

import pytest

from forge.envgen.agents.correctness import audit_authoring_contract
from forge.envgen.agents.reviewer import ReviewSeverity


def _categories(issues):
    return {i.category for i in issues}


# A state class with both contract methods and endpoints that return typed dicts.
_COMPLIANT = '''
from fastapi import FastAPI

app = FastAPI()


class EnvState:
    def __init__(self):
        self.rows = []

    def seed_state(self, seed: int) -> None:
        self.rows = [{"id": 1, "seed": seed}]

    def reset_state(self) -> None:
        self.rows = []
        self.seed_state(0)


STATE = EnvState()


@app.get("/forge/state")
def get_state():
    return {"rows": STATE.rows}


@app.post("/close_ticket")
def close_ticket(ticket_id: int):
    if ticket_id < 0:
        return {"ok": False, "error": "unknown ticket"}
    return {"ok": True, "ticket_id": ticket_id}
'''


# ── State-management class contract ───────────────────────────────────────


def test_compliant_app_has_no_findings():
    assert audit_authoring_contract({"main.py": _COMPLIANT}) == []


def test_missing_state_class_is_flagged():
    src = _COMPLIANT.replace("class EnvState:", "class Unrelated:").replace(
        "def seed_state(self, seed: int) -> None:", "def build(self, seed: int) -> None:"
    ).replace("def reset_state(self) -> None:", "def wipe(self) -> None:")
    assert "state_class_missing" in _categories(audit_authoring_contract({"main.py": src}))


def test_class_with_only_reset_state_is_insufficient():
    # A class exposing reset_state but not seed_state does not satisfy the contract.
    src = _COMPLIANT.replace(
        "    def seed_state(self, seed: int) -> None:\n"
        "        self.rows = [{\"id\": 1, \"seed\": seed}]\n\n",
        "",
    )
    assert "state_class_missing" in _categories(audit_authoring_contract({"main.py": src}))


def test_seed_state_without_a_seed_parameter_is_flagged():
    # seed_state must be seed-driven; a no-arg seed_state cannot be reproducible.
    src = _COMPLIANT.replace(
        "def seed_state(self, seed: int) -> None:", "def seed_state(self) -> None:"
    ).replace("self.rows = [{\"id\": 1, \"seed\": seed}]", "self.rows = [{\"id\": 1}]")
    assert "seed_state_signature" in _categories(audit_authoring_contract({"main.py": src}))


def test_state_class_may_live_in_any_backend_file():
    other = _COMPLIANT.replace(
        "@app.get(\"/forge/state\")\n"
        "def get_state():\n"
        "    return {\"rows\": STATE.rows}\n\n\n"
        "@app.post(\"/close_ticket\")\n"
        "def close_ticket(ticket_id: int):\n"
        "    if ticket_id < 0:\n"
        "        return {\"ok\": False, \"error\": \"unknown ticket\"}\n"
        "    return {\"ok\": True, \"ticket_id\": ticket_id}\n",
        "",
    )
    main = "from state import STATE\n"
    assert audit_authoring_contract({"main.py": main, "state.py": other}) == []


# ── Typed-dict return contract ────────────────────────────────────────────


def test_endpoint_returning_a_bare_string_is_flagged():
    src = _COMPLIANT.replace(
        "    return {\"ok\": True, \"ticket_id\": ticket_id}",
        "    return \"closed\"",
    )
    issues = audit_authoring_contract({"main.py": src})
    assert "untyped_return" in _categories(issues)


def test_endpoint_returning_an_fstring_is_flagged():
    src = _COMPLIANT.replace(
        "    return {\"ok\": True, \"ticket_id\": ticket_id}",
        "    return f\"closed {ticket_id}\"",
    )
    assert "untyped_return" in _categories(audit_authoring_contract({"main.py": src}))


def test_error_path_returning_a_string_is_flagged():
    src = _COMPLIANT.replace(
        "        return {\"ok\": False, \"error\": \"unknown ticket\"}",
        "        return \"unknown ticket\"",
    )
    assert "untyped_return" in _categories(audit_authoring_contract({"main.py": src}))


def test_non_endpoint_helper_returning_a_string_is_not_flagged():
    # Only route handlers are subject to the typed-dict rule; a plain helper
    # returning a string must NOT trip the audit (false-positive guard).
    src = _COMPLIANT + '\n\ndef format_label(row) -> str:\n    return f"row-{row}"\n'
    assert "untyped_return" not in _categories(audit_authoring_contract({"main.py": src}))


def test_endpoint_returning_a_file_response_is_not_flagged():
    src = _COMPLIANT + (
        "\n\n@app.get(\"/ui\")\n"
        "def ui():\n"
        "    return FileResponse(\"ui.html\")\n"
    )
    assert "untyped_return" not in _categories(audit_authoring_contract({"main.py": src}))


def test_nested_helper_string_inside_endpoint_is_not_flagged():
    src = _COMPLIANT.replace(
        "    return {\"ok\": True, \"ticket_id\": ticket_id}",
        "    def label():\n"
        "        return \"inner\"\n"
        "    return {\"ok\": True, \"label\": label()}",
    )
    assert "untyped_return" not in _categories(audit_authoring_contract({"main.py": src}))


def test_all_findings_are_errors():
    src = _COMPLIANT.replace("class EnvState:", "class Unrelated:").replace(
        "    return {\"ok\": True, \"ticket_id\": ticket_id}", "    return \"closed\""
    )
    issues = audit_authoring_contract({"main.py": src})
    assert issues
    assert all(i.severity == ReviewSeverity.ERROR for i in issues)


def test_syntactically_invalid_file_is_skipped():
    assert audit_authoring_contract({"main.py": "def (:::"}) != []  # missing state class
    # ...but a broken file must not raise.
    audit_authoring_contract({"broken.py": "def (:::", "main.py": _COMPLIANT})
