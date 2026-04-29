from __future__ import annotations
from forge.extraction.schemas import CompilerInput, SuccessCondition


class AdversarialTestGenerator:
    def __init__(self, llm_client=None) -> None:
        self._llm_client = llm_client

    def generate(self, compiler_input: CompilerInput) -> dict[str, str]:
        return {
            f"{task.name}_adversarial": self._generate_task(task, compiler_input.project_name)
            for task in compiler_input.tasks
        }

    def _generate_task(self, task, project_name: str) -> str:
        lines = [
            "from __future__ import annotations",
            "import pytest",
            f"from {project_name}.verifiers.{task.name} import verify_{task.name}",
            "",
            "",
            "class _FakeStep:",
            "    def __init__(self, action_type: str):",
            "        self.action = {'type': action_type}",
            "",
            "",
            "class _FakeTraj:",
            "    def __init__(self, steps=None, events=None):",
            "        self.steps = steps or []",
            "        self.events = events or []",
        ]
        for i, cond in enumerate(task.success_conditions, 1):
            lines.extend(["", ""])
            lines.extend(self._generate_condition_test(task.name, cond, i).splitlines())
        return "\n".join(lines) + "\n"

    def _generate_condition_test(self, task_name: str, cond: SuccessCondition, idx: int) -> str:
        if cond.type == "state_check":
            return (
                f"def test_{task_name}_adversarial_state_{idx}():\n"
                f"    traj = _FakeTraj()\n"
                f"    result = verify_{task_name}({{}}, traj, {{}})\n"
                f"    assert not result.passed\n"
            )
        if cond.type == "event_check":
            return (
                f"def test_{task_name}_adversarial_event_{idx}():\n"
                f"    traj = _FakeTraj(events=[])\n"
                f"    result = verify_{task_name}({{}}, traj, {{}})\n"
                f"    assert not result.passed\n"
            )
        if cond.type == "temporal_check":
            parts = cond.expression.split(" before ", 1)
            if len(parts) == 2:
                a, b = parts[0].strip(), parts[1].strip()
                return (
                    f"def test_{task_name}_adversarial_temporal_{idx}():\n"
                    f"    traj = _FakeTraj(events=[\n"
                    f"        {{'type': {b!r}}},\n"
                    f"        {{'type': {a!r}}},\n"
                    f"    ])\n"
                    f"    result = verify_{task_name}({{}}, traj, {{}})\n"
                    f"    assert not result.passed\n"
                )
            return (
                f"def test_{task_name}_adversarial_temporal_{idx}():\n"
                f"    pytest.skip({f'Cannot parse temporal expression: {cond.expression}'!r})\n"
            )
        if cond.type == "policy_check":
            return (
                f"def test_{task_name}_adversarial_policy_{idx}():\n"
                f"    traj = _FakeTraj(steps=[_FakeStep({cond.expression!r})])\n"
                f"    result = verify_{task_name}({{}}, traj, {{}})\n"
                f"    assert not result.passed\n"
            )
        if cond.type == "negative_check":
            return (
                f"def test_{task_name}_adversarial_negative_{idx}():\n"
                f"    traj = _FakeTraj(steps=[_FakeStep({cond.expression!r})])\n"
                f"    result = verify_{task_name}({{}}, traj, {{}})\n"
                f"    assert not result.passed\n"
            )
        if cond.type == "semantic_check":
            return (
                f"@pytest.mark.skip(reason='Semantic adversarial test requires LLM at compile time')\n"
                f"def test_{task_name}_adversarial_semantic_{idx}():\n"
                f"    pass\n"
            )
        return (
            f"def test_{task_name}_adversarial_unknown_{idx}():\n"
            f"    pytest.skip({f'Unknown condition type: {cond.type}'!r})\n"
        )
