"""Unit tests for synthetic data generation types (Feature Request 7).

Supported formats:
- preference pairs (reusing passed as preferred, rejected as dispreferred)
- RL tasks (task prompt, verifiers, golden solution patch)
- SFT data (task prompt and golden demonstration for imitation learning)
- Target environment selection
"""
from __future__ import annotations

from contextlib import contextmanager
import pytest
from forge.taskfactory.pipeline import PipelineResult, TaskFactoryPipeline
from forge.taskfactory.profile import EnvironmentProfile, ToolInfo
from forge.taskfactory.schemas import (
    Check,
    GoldenStep,
    PreferencePair,
    ReviewVerdict,
    SFTItem,
    SyntheticTask,
    TaskDraft,
    TaskRejection,
    TaskSeed,
    Taxonomy,
    TaxonomyCategory,
)
from forge.taskfactory.pass_k import PassKResult


def _sample_task(task_id: str = "task_1", category: str = "core") -> SyntheticTask:
    return SyntheticTask(
        id=task_id,
        category=category,
        difficulty=1,
        title="Sample Task",
        objective="Sort the inbox emails by date",
        seed=TaskSeed(),
        golden=[GoldenStep(tool="sort", args={"by": "date"})],
        checks=[Check(kind="value", path="sorted", value=True)],
        reflection_points=[],
        step_budget=10,
        round=1,
        review=ReviewVerdict(
            slot=0,
            realistic=True,
            realistic_reason="good",
            fair=True,
            fair_reason="good",
            sensible=True,
            sensible_reason="good",
        ),
        fingerprint="fp123",
    )


def _sample_rejection(slot: int = 1, category: str = "core") -> TaskRejection:
    return TaskRejection(
        slot=slot,
        category=category,
        difficulty=1,
        round=1,
        stage="pass_k",
        reason="Nondeterministic output between runs",
        draft=TaskDraft(
            slot=slot,
            category=category,
            difficulty=1,
            title="Bad Task",
            objective="Sort emails unpredictably",
            seed=TaskSeed(),
            golden=[GoldenStep(tool="random_sort", args={})],
            checks=[Check(kind="value", path="sorted", value=True)],
        ),
    )


def test_rl_task_properties():
    task = _sample_task()
    assert task.task_prompt == "Sort the inbox emails by date"
    assert len(task.verifiers) == 1
    assert task.verifiers[0].kind == "value"
    assert len(task.golden_solution_patch) == 1
    assert task.golden_solution_patch[0]["tool"] == "sort"


def test_pipeline_result_to_preference_pairs():
    task = _sample_task()
    rejection = _sample_rejection()
    taxonomy = Taxonomy(categories=[TaxonomyCategory(name="core", description="", difficulties=[1])])
    result = PipelineResult(
        requested=1,
        taxonomy=taxonomy,
        tasks=[task],
        rejections=[rejection],
        data_type="preference_pairs",
    )
    pairs = result.to_preference_pairs()
    assert len(pairs) == 1
    pair = pairs[0]
    assert isinstance(pair, PreferencePair)
    assert pair.task_prompt == task.objective
    assert pair.preferred["id"] == "task_1"
    assert "Sort emails unpredictably" in str(pair.dispreferred)
    assert pair.rejection_reason == "Nondeterministic output between runs"


def test_pipeline_result_to_sft_items():
    task = _sample_task()
    taxonomy = Taxonomy(categories=[TaxonomyCategory(name="core", description="", difficulties=[1])])
    result = PipelineResult(
        requested=1,
        taxonomy=taxonomy,
        tasks=[task],
        data_type="sft_data",
    )
    sft = result.to_sft_items()
    assert len(sft) == 1
    item = sft[0]
    assert isinstance(item, SFTItem)
    assert item.task_prompt == task.objective
    assert len(item.golden_solution) == 1
    assert item.golden_solution[0]["tool"] == "sort"
    assert item.golden_solution_patch == item.golden_solution


def test_pipeline_run_generates_preference_pairs_byproduct():
    class FakeTaxonomyBuilder:
        def build(self, profile, count):
            return Taxonomy(categories=[TaxonomyCategory(name="core", description="", difficulties=[1])])

    profile = EnvironmentProfile(
        env_name="mail",
        env_type="premade:gmail",
        family="state",
        tools=(ToolInfo(name="/archive_email"),),
        state_sample={"emails": []},
    )

    class FakeWriter:
        def write(self, prof, taxonomy, slots, *, avoid_titles=(), feedback=None):
            result = {}
            for s in slots:
                title = "Good" if s.index == 0 else "Bad"
                result[s.index] = TaskDraft(
                    slot=s.index,
                    category="core",
                    difficulty=1,
                    title=title,
                    objective=f"{title} objective",
                    seed=TaskSeed(),
                    golden=[GoldenStep(tool="/archive_email", args={"email_id": "e1"})],
                    checks=[
                        Check(
                            kind="record_exists",
                            collection="emails",
                            match={"id": "e1"},
                            expect={"folder": "archive"},
                        )
                    ],
                )
            return result

        def revise(self, prof, taxonomy, items):
            return {}

    class FakeReviewer:
        def review(self, prof, tasks):
            return {
                slot.index: ReviewVerdict(
                    slot=slot.index,
                    realistic=True,
                    realistic_reason="ok",
                    fair=True,
                    fair_reason="ok",
                    sensible=True,
                    sensible_reason="ok",
                )
                for slot, draft in tasks
            }

    class FakeRunner:
        @contextmanager
        def batch(self):
            yield

        def run(self, draft):
            pass

    def fake_pass_k(runner, draft, k):
        if "Bad" in draft.title:
            return PassKResult(False, reason="flaky check")
        return PassKResult(True, fingerprint="fp1")

    pipeline = TaskFactoryPipeline(
        profile=profile,
        taxonomy_builder=FakeTaxonomyBuilder(),
        writer=FakeWriter(),
        reviewer=FakeReviewer(),
        runner=FakeRunner(),
        pass_k=fake_pass_k,
    )

    result = pipeline.run(count=2, k=2, data_type="preference_pairs")
    assert result.data_type == "preference_pairs"
    assert len(result.tasks) == 1
    assert len(result.rejections) >= 1
    assert len(result.preference_pairs) == 1
    pair = result.preference_pairs[0]
    assert pair.preferred["title"] == "Good"
    assert "Bad" in str(pair.dispreferred)


def test_rejects_invalid_task_rejection_stage():
    with pytest.raises(Exception):
        # Validation test for malformed rejection
        TaskRejection(slot="invalid_type", category="core", difficulty=1, round=1, stage="unknown", reason="err")  # type: ignore
