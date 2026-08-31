"""A generated package must satisfy the contracts it is built against."""
from __future__ import annotations

import importlib
import sys

from forge.compiler.generators.gym_wrapper import GymWrapperGenerator
from forge.compiler.generators.initial_state import InitialStateGenerator
from forge.compiler.generators.reward import RewardGenerator
from forge.compiler.generators.transition import TransitionGenerator
from forge.compiler.generators.verifier import VerifierGenerator
from forge.compiler.package_builder import PackageBuilder
from forge.extraction.schemas import (
    ActionDef,
    ActionParam,
    CompilerInput,
    EntityDef,
    FieldDef,
    SuccessCondition,
    TaskTemplate,
)


def _compiler_input() -> CompilerInput:
    return CompilerInput(
        project_name="ticket_env",
        domain="support",
        entities=[
            EntityDef(name="ticket", fields=[FieldDef(name="id", type="string")])
        ],
        actions=[ActionDef(name="close_ticket", params=[])],
        tasks=[],
    )


def _compiler_input_with_task() -> CompilerInput:
    # The brief's own fixture declares zero tasks, so it never renders the
    # {% for task in tasks %} branches of gym_wrapper.py.j2 / verifier.py.j2 /
    # reward.py.j2. Those branches are exactly where a second bare-function
    # registration bug lives (see test_generated_wrapper_registers_verifier_
    # and_rubric_instances_not_functions below), so this fixture exists to
    # actually render them.
    return CompilerInput(
        project_name="counter_env_contract_check",
        domain="counter",
        entities=[
            EntityDef(
                name="counter",
                fields=[
                    FieldDef(name="id", type="string"),
                    FieldDef(name="value", type="integer", default=0),
                ],
            )
        ],
        actions=[
            ActionDef(
                name="increment",
                params=[ActionParam(name="counter_id", type="string")],
                mutates=["counter.value"],
            )
        ],
        tasks=[
            TaskTemplate(
                name="reach_target",
                description="Reach target value",
                success_conditions=[
                    SuccessCondition(
                        type="state_check", expression="counter.value >= target"
                    )
                ],
            )
        ],
    )


# ---------------------------------------------------------------------------
# Brief's tests, verbatim
# ---------------------------------------------------------------------------


def test_generated_transitions_expose_a_handler_class():
    # generate() returns {action_name: source}, one entry per declared action.
    sources = TransitionGenerator().generate(_compiler_input())
    source = sources["close_ticket"]
    assert "class CloseTicketHandler(TransitionHandler):" in source
    assert "def apply(self, state: dict, action: Action, ctx" in source


def test_generated_transitions_keep_the_plain_function():
    # False-positive guard: the function form is what customization overrides
    # and hand-written callers use. Wrapping must not remove it.
    sources = TransitionGenerator().generate(_compiler_input())
    assert "def apply_close_ticket(" in sources["close_ticket"]


def test_generated_wrapper_registers_handler_instances_not_functions():
    source = GymWrapperGenerator().generate(_compiler_input())
    # Negative: registering a bare function would now raise at build time.
    assert 'te.register("close_ticket", CloseTicketHandler())' in source
    assert "apply_close_ticket)" not in source


def test_generated_wrapper_uses_the_provider_keyword():
    source = GymWrapperGenerator().generate(_compiler_input())
    assert "initial_state_provider=" in source
    assert "initial_state_factory=" not in source


def test_generated_initial_state_implements_the_provider_contract():
    source = InitialStateGenerator().generate(_compiler_input())
    assert "InitialStateProvider" in source
    assert (
        "def reset(self, ctx: RuntimeContext, *, seed: int | None, options: dict)"
        in source
    )
    assert "def create(" not in source


# ---------------------------------------------------------------------------
# Additional coverage: verifier / reward wrapper classes (mirrors the
# transition tests above; the brief describes the same additive shape for
# verifier.py.j2 and reward.py.j2 but its zero-task fixture can't exercise it)
# ---------------------------------------------------------------------------


def test_generated_verifiers_expose_a_verifier_class():
    sources = VerifierGenerator().generate(_compiler_input_with_task())
    source = sources["reach_target"]
    assert "class ReachTargetVerifier(Verifier):" in source
    assert (
        "def verify(self, state: dict, trajectory, task) -> VerificationResult:"
        in source
    )


def test_generated_verifiers_keep_the_plain_function():
    sources = VerifierGenerator().generate(_compiler_input_with_task())
    assert "def verify_reach_target(" in sources["reach_target"]


def test_generated_rewards_expose_a_rubric_class():
    sources = RewardGenerator().generate(_compiler_input_with_task())
    source = sources["reach_target"]
    assert "class ReachTargetRubric(Rubric):" in source
    assert (
        "def score(self, state, trajectory, verifier_results, task) -> RewardBreakdown:"
        in source
    )


def test_generated_rewards_keep_the_plain_function():
    sources = RewardGenerator().generate(_compiler_input_with_task())
    assert "def compute_reach_target_reward(" in sources["reach_target"]


def test_generated_wrapper_registers_verifier_and_rubric_instances_not_functions():
    # VerifierEngine.register and RewardEngine.set_default apply the exact
    # same isinstance(..., Verifier/Rubric) gate that TransitionEngine.register
    # does (forge/runtime/verifier.py, forge/runtime/reward.py) — a bare
    # function there raises TypeError at build time exactly like the
    # transition case the brief documents. The brief's Step 4 diff shows only
    # the transition registration line, but leaving ve.register/re.set_default
    # on bare functions would leave this exact defect open for every
    # generated environment that declares a task.
    source = GymWrapperGenerator().generate(_compiler_input_with_task())
    assert 've.register("reach_target", ReachTargetVerifier())' in source
    assert "re.set_default(ReachTargetRubric())" in source
    # Negative: the old bare-function registration lines must be gone.
    assert "verify_reach_target)" not in source
    assert "compute_reach_target_reward)" not in source


# ---------------------------------------------------------------------------
# Real import-and-execute verification: rendering-and-parsing (ast.parse, as
# the other generator tests do) only proves the output is syntactically valid
# Python. The registries this task closes a window on only reject a bad shape
# at *runtime*, when te.register/ve.register/re.set_default actually execute
# — so this builds the package for real, imports it, and constructs the env.
# ---------------------------------------------------------------------------


def _clear_generated_envs_modules() -> None:
    for name in [m for m in sys.modules if m.startswith("generated_envs")]:
        del sys.modules[name]


def test_generated_package_actually_imports_and_builds(tmp_path, monkeypatch):
    ci = _compiler_input_with_task()
    envs_root = tmp_path / "generated_envs"
    PackageBuilder(envs_root).build(ci)
    (envs_root / "__init__.py").touch()

    monkeypatch.syspath_prepend(str(tmp_path))
    _clear_generated_envs_modules()
    try:
        module = importlib.import_module(
            f"generated_envs.{ci.project_name}.gym_wrapper"
        )
        build_fn = getattr(module, f"build_{ci.project_name}_env")

        # This call is exactly where a bare-function registration used to
        # raise TypeError (te.register / ve.register / re.set_default all
        # execute inside it). Constructing the env for real, rather than
        # only rendering-and-parsing the template source, is the "does it
        # actually import and work" check.
        env = build_fn()
        obs, info = env.reset(seed=0)
        assert "episode_id" in info
        assert isinstance(obs, dict)
    finally:
        _clear_generated_envs_modules()


def test_generated_verifier_class_does_not_shadow_imported_verifier_of_same_name(
    tmp_path, monkeypatch
):
    # A task named "semantic" (or "event", "policy", "temporal", "negative",
    # "exact_state") pascal_cases to a generated class name that collides
    # with the same-named class imported from forge.runtime.verifiers (e.g.
    # `class SemanticVerifier(Verifier)` rebinding the imported
    # `SemanticVerifier`). Rendering and ast.parse'ing the template can't
    # catch this — the module still imports and type-checks fine, it only
    # breaks at call time. So this builds a real package and CALLS the
    # generated verify_semantic function.
    ci = CompilerInput(
        project_name="semantic_shadow_env",
        domain="support",
        entities=[
            EntityDef(name="ticket", fields=[FieldDef(name="id", type="string")])
        ],
        actions=[ActionDef(name="close_ticket", params=[])],
        tasks=[
            TaskTemplate(
                name="semantic",
                description="Reply satisfies the rubric",
                success_conditions=[
                    SuccessCondition(
                        type="semantic_check",
                        expression="reply_text",
                        rubric="Politely acknowledges the customer's issue",
                    )
                ],
            )
        ],
    )
    envs_root = tmp_path / "generated_envs"
    PackageBuilder(envs_root).build(ci)
    (envs_root / "__init__.py").touch()

    monkeypatch.syspath_prepend(str(tmp_path))
    _clear_generated_envs_modules()
    try:
        module = importlib.import_module(
            f"generated_envs.{ci.project_name}.verifiers.semantic"
        )
        from forge.contracts import VerificationResult

        result = module.verify_semantic({"reply_text": "hi"}, [], {})
        assert isinstance(result, VerificationResult)
    finally:
        _clear_generated_envs_modules()


def test_generated_test_suite_modules_actually_run(tmp_path, monkeypatch):
    # The generated tests/test_transitions.py and tests/test_verifiers.py
    # build initial state via `...InitialStateFactory().create(ctx, {})`.
    # Renaming create -> reset in initial_state.py.j2 (this task's one
    # non-additive edit) breaks that helper unless the two test templates are
    # updated to match — a regression this test would catch directly.
    ci = _compiler_input_with_task()
    envs_root = tmp_path / "generated_envs"
    PackageBuilder(envs_root).build(ci)
    (envs_root / "__init__.py").touch()

    monkeypatch.syspath_prepend(str(tmp_path))
    _clear_generated_envs_modules()
    try:
        transitions_mod = importlib.import_module(
            f"generated_envs.{ci.project_name}.tests.test_transitions"
        )
        transitions_mod.test_increment_returns_transition_result()
        transitions_mod.test_increment_does_not_mutate_original()

        verifiers_mod = importlib.import_module(
            f"generated_envs.{ci.project_name}.tests.test_verifiers"
        )
        verifiers_mod.test_reach_target_verifier_returns_result()
    finally:
        _clear_generated_envs_modules()
