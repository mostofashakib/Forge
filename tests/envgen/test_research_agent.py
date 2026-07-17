from __future__ import annotations

import pytest

from forge.envgen.artifact_bus import ArtifactBus
from forge.envgen.context import EnvGenContext
from forge.envgen.research import (
    ApplicationResearchBrief,
    ContextPruner,
    ResearchSource,
    UserResearchAgent,
    WebDocument,
)
from forge.extraction.llm_client import MockLLMClient
from forge.extraction.schemas import ActionDef, CompilerInput, EntityDef, FieldDef


def _ctx(
    *,
    reference_urls: list[str] | None = None,
    source_product_name: str = "",
    source_product_url: str = "",
) -> EnvGenContext:
    return EnvGenContext(
        env_name="mail_env",
        description="A faithful prototype of Acme Mail",
        compiler_input=CompilerInput(
            project_name="mail_env",
            domain="acme_mail",
            entities=[EntityDef(name="message", fields=[FieldDef(name="id", type="string")])],
            actions=[ActionDef(name="archive_message", params=[])],
            tasks=[],
        ),
        reference_urls=reference_urls or [],
        source_product_name=source_product_name,
        source_product_url=source_product_url,
    )


class _Web:
    def __init__(self) -> None:
        self.searched = False
        self.fetched: list[str] = []

    async def search(self, query: str, *, limit: int) -> list[str]:
        self.searched = True
        return ["https://docs.example.test/mail"][:limit]

    async def fetch(self, url: str) -> WebDocument:
        self.fetched.append(url)
        return WebDocument(
            title="Acme Mail guide",
            url=url,
            text="Inbox supports unread, selected, archived, and empty states.",
        )


def _brief() -> ApplicationResearchBrief:
    return ApplicationResearchBrief(
        product_summary="A mail application",
        workflows=["Open inbox", "Archive a message"],
        functional_requirements=["Archive messages"],
        ui_states=["Unread", "Selected", "Empty inbox"],
        data_requirements=["Message id and archived status"],
        business_rules=["Archived messages leave the inbox"],
        rl_observations=["Expose the selected message and inbox contents"],
        edge_cases=["Archiving an already archived message is idempotent"],
        sources=[ResearchSource(title="Guide", url="https://docs.example.test/mail")],
    )


@pytest.mark.asyncio
async def test_research_agent_reads_explicit_sources_and_publishes_pruned_contexts():
    web = _Web()
    agent = UserResearchAgent(
        client=MockLLMClient({"ApplicationResearchBrief": _brief()}),
        web=web,
    )
    bus = ArtifactBus()

    await agent.run(_ctx(reference_urls=["https://docs.example.test/mail"]), bus)

    assert web.searched is False
    assert web.fetched == ["https://docs.example.test/mail"]
    assert "ui_states" in bus.get("ui_research").sections
    assert "ui_states" not in bus.get("backend_research").sections
    assert "rl_observations" in bus.get("rl_research").sections
    assert all("raw" not in key for key in bus.snapshot())


@pytest.mark.asyncio
async def test_research_agent_prioritizes_original_product_url():
    web = _Web()
    agent = UserResearchAgent(
        client=MockLLMClient({"ApplicationResearchBrief": _brief()}),
        web=web,
    )

    await agent.run(_ctx(
        source_product_name="Acme Mail",
        source_product_url="https://acme.example.test",
        reference_urls=["https://docs.example.test/mail"],
    ), ArtifactBus())

    assert web.fetched == [
        "https://acme.example.test",
        "https://docs.example.test/mail",
    ]


@pytest.mark.asyncio
async def test_research_agent_falls_back_to_structured_spec_when_web_and_llm_fail():
    class _FailingWeb(_Web):
        async def search(self, query: str, *, limit: int) -> list[str]:
            raise RuntimeError("offline")

    class _FailingClient:
        def extract(self, system, user, schema):
            raise RuntimeError("model unavailable")

    bus = ArtifactBus()
    await UserResearchAgent(client=_FailingClient(), web=_FailingWeb()).run(_ctx(), bus)

    backend = bus.get("backend_research")
    assert backend.product_summary == _ctx().description
    assert backend.sections["functional_requirements"] == ["Support the archive_message action"]


def test_context_pruner_enforces_role_relevance_and_size_budget():
    brief = _brief().model_copy(update={
        "workflows": [f"Workflow {index} " + ("x" * 80) for index in range(30)]
    })
    context = ContextPruner(max_chars=300, max_items_per_section=2).for_role(brief, "ui")

    assert len(context.sections["workflows"]) <= 2
    assert "data_requirements" not in context.sections
    assert len(context.as_prompt()) <= 300


def test_research_prompt_caps_raw_web_content():
    document = WebDocument(
        title="Large guide",
        url="https://docs.example.test/large",
        text="x" * 100_000,
    )

    prompt = UserResearchAgent._build_prompt(_ctx(), [document])

    assert prompt.count("x") < 25_000
