from __future__ import annotations

import asyncio
import html
import re
from dataclasses import dataclass
from typing import Protocol
from urllib.parse import parse_qs, quote_plus, unquote, urlparse

import httpx
from pydantic import BaseModel, Field

from forge.envgen.agents.base import EnvGenAgent
from forge.envgen.artifact_bus import ArtifactBus
from forge.envgen.context import EnvGenContext
from forge.envgen.config import EnvGenConfig, envgen_config
from forge.extraction.llm_client import LLMClient, get_client


class ResearchSource(BaseModel):
    title: str
    url: str
    summary: str = ""


class ApplicationResearchBrief(BaseModel):
    """Compact product understanding derived from specs and external evidence."""

    product_summary: str
    workflows: list[str] = Field(default_factory=list)
    functional_requirements: list[str] = Field(default_factory=list)
    ui_states: list[str] = Field(default_factory=list)
    data_requirements: list[str] = Field(default_factory=list)
    business_rules: list[str] = Field(default_factory=list)
    rl_observations: list[str] = Field(default_factory=list)
    edge_cases: list[str] = Field(default_factory=list)
    sources: list[ResearchSource] = Field(default_factory=list)


class SpecialistResearchContext(BaseModel):
    role: str
    product_summary: str
    sections: dict[str, list[str]] = Field(default_factory=dict)
    sources: list[ResearchSource] = Field(default_factory=list)
    prompt_budget: int = 12_000

    def as_prompt(self) -> str:
        lines = [f"Product understanding: {self.product_summary}"]
        for heading, items in self.sections.items():
            if items:
                lines.append(f"\n{heading.replace('_', ' ').title()}:")
                lines.extend(f"- {item}" for item in items)
        if self.sources:
            lines.append("\nEvidence sources:")
            lines.extend(f"- {source.title}: {source.url}" for source in self.sources)
        return "\n".join(lines)[: self.prompt_budget]


class ContextPruner:
    """Selects role-relevant research and enforces a hard prompt budget."""

    _ROLE_SECTIONS = {
        "backend": ("workflows", "functional_requirements", "data_requirements", "business_rules", "edge_cases"),
        "ui": ("workflows", "functional_requirements", "ui_states", "edge_cases"),
        "rl": ("workflows", "functional_requirements", "data_requirements", "business_rules", "rl_observations", "edge_cases"),
        "reviewer": ("workflows", "functional_requirements", "ui_states", "data_requirements", "business_rules", "rl_observations", "edge_cases"),
    }

    def __init__(self, *, max_chars: int = 12_000, max_items_per_section: int = 8) -> None:
        if max_chars < 200:
            raise ValueError("max_chars must be at least 200")
        if max_items_per_section < 1:
            raise ValueError("max_items_per_section must be positive")
        self.max_chars = max_chars
        self.max_items_per_section = max_items_per_section

    def for_role(self, brief: ApplicationResearchBrief, role: str) -> SpecialistResearchContext:
        if role not in self._ROLE_SECTIONS:
            raise ValueError(f"Unknown research context role: {role!r}")
        remaining = self.max_chars - len(brief.product_summary)
        sections: dict[str, list[str]] = {}
        for section_name in self._ROLE_SECTIONS[role]:
            selected: list[str] = []
            for raw_item in getattr(brief, section_name)[: self.max_items_per_section]:
                item = " ".join(raw_item.split())
                cost = len(item) + 3
                if cost > remaining:
                    break
                selected.append(item)
                remaining -= cost
            sections[section_name] = selected
        sources: list[ResearchSource] = []
        for source in brief.sources[:5]:
            cost = len(source.title) + len(source.url) + 4
            if cost > remaining:
                break
            sources.append(source.model_copy(update={"summary": ""}))
            remaining -= cost
        return SpecialistResearchContext(
            role=role,
            product_summary=brief.product_summary[: self.max_chars],
            sections=sections,
            sources=sources,
            prompt_budget=self.max_chars,
        )


@dataclass(frozen=True)
class WebDocument:
    title: str
    url: str
    text: str


class WebResearchProvider(Protocol):
    async def search(self, query: str, *, limit: int) -> list[str]: ...
    async def fetch(self, url: str) -> WebDocument: ...


class HttpWebResearchProvider:
    """Small dependency-free web reader used only by the research specialist."""

    _RESULT_RE = re.compile(
        r'<a[^>]+class="[^"]*result__a[^"]*"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
        re.IGNORECASE | re.DOTALL,
    )
    _TAG_RE = re.compile(r"<[^>]+>")

    def __init__(self, *, timeout: float = 8.0, max_document_chars: int = 20_000) -> None:
        self.timeout = timeout
        self.max_document_chars = max_document_chars

    async def search(self, query: str, *, limit: int) -> list[str]:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(
                f"https://html.duckduckgo.com/html/?q={quote_plus(query)}",
                headers={"User-Agent": "Forge research agent/1.0"},
            )
            response.raise_for_status()
        urls: list[str] = []
        for href, _title in self._RESULT_RE.findall(response.text):
            parsed = urlparse(html.unescape(href))
            target = unquote(parse_qs(parsed.query).get("uddg", [href])[0])
            if target.startswith(("https://", "http://")) and target not in urls:
                urls.append(target)
            if len(urls) >= limit:
                break
        return urls

    async def fetch(self, url: str) -> WebDocument:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError(f"Unsupported research URL: {url!r}")
        async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=True) as client:
            response = await client.get(url, headers={"User-Agent": "Forge research agent/1.0"})
            response.raise_for_status()
        content_type = response.headers.get("content-type", "")
        if "text/html" not in content_type and "text/plain" not in content_type:
            raise ValueError(f"Research URL did not return text: {url!r}")
        title_match = re.search(r"<title[^>]*>(.*?)</title>", response.text, re.I | re.S)
        title = html.unescape(self._TAG_RE.sub(" ", title_match.group(1))).strip() if title_match else parsed.hostname
        text = self._TAG_RE.sub(" ", response.text)
        text = " ".join(html.unescape(text).split())[: self.max_document_chars]
        return WebDocument(title=title, url=str(response.url), text=text)


_RESEARCH_SYSTEM = """You are a product research specialist for an RL environment builder.
Infer how the referenced application actually works from the user request, structured app spec,
and supplied web excerpts. Identify concrete workflows, functionality, observable UI states,
persistent data, business rules, RL-relevant observations, and edge cases needed for a faithful
prototype. Prefer supplied evidence over assumptions. Keep every list item concise. Do not copy
page prose. Sources must include only URLs present in the supplied excerpts."""


class ResearchPrompts:
    SYSTEM = _RESEARCH_SYSTEM


class UserResearchAgent(EnvGenAgent):
    """Researches the target product, then publishes only pruned specialist briefs."""

    agent_id = "user_researcher"
    depends_on: list[str] = []
    produces = ["backend_research", "ui_research", "rl_research", "reviewer_research"]

    def __init__(
        self,
        client: LLMClient | None = None,
        web: WebResearchProvider | None = None,
        pruner: ContextPruner | None = None,
        *,
        search_results: int | None = None,
        config: EnvGenConfig | None = None,
    ) -> None:
        self._config = config or envgen_config()
        self._client = client or get_client(
            max_tokens=self._config.standard_llm_tokens, capable=True
        )
        self._web = web or HttpWebResearchProvider(
            timeout=self._config.research_http_timeout,
            max_document_chars=self._config.research_document_chars,
        )
        self._pruner = pruner or ContextPruner(
            max_chars=self._config.specialist_context_chars,
            max_items_per_section=self._config.specialist_items_per_section,
        )
        self._search_results = search_results or self._config.research_search_results

    async def run(self, ctx: EnvGenContext, bus: ArtifactBus) -> None:
        await bus.log("[user-researcher] Reading application spec and external references…")
        urls = list(dict.fromkeys(ctx.reference_urls + self._urls_in(ctx.description)))
        if not urls and ctx.compiler_input.domain.lower() not in {"localhost", "test", ""}:
            query = (
                f"{ctx.compiler_input.domain} {ctx.description[:300]} "
                "application user guide workflows features data states"
            )
            try:
                urls = await self._web.search(query, limit=self._search_results)
            except Exception as exc:
                await bus.log(f"[user-researcher] Web search unavailable; using app spec ({exc})")
        documents = await self._fetch_documents(urls[: self._search_results], bus)
        prompt = self._build_prompt(
            ctx,
            documents,
            source_char_budget=self._config.research_total_source_chars,
        )
        loop = asyncio.get_running_loop()
        try:
            brief: ApplicationResearchBrief = await loop.run_in_executor(
                None,
                lambda: self._client.extract(
                    system=ResearchPrompts.SYSTEM,
                    user=prompt,
                    schema=ApplicationResearchBrief,
                ),
            )
        except Exception as exc:
            await bus.log(f"[user-researcher] Research synthesis failed; using structured spec ({exc})")
            brief = self._fallback_brief(ctx, documents)
        for artifact, role in (
            ("backend_research", "backend"),
            ("ui_research", "ui"),
            ("rl_research", "rl"),
            ("reviewer_research", "reviewer"),
        ):
            await bus.publish(artifact, self._pruner.for_role(brief, role))
        await bus.log("[user-researcher] Published pruned specialist context; raw pages discarded")

    async def _fetch_documents(self, urls: list[str], bus: ArtifactBus) -> list[WebDocument]:
        results = await asyncio.gather(
            *(self._web.fetch(url) for url in urls), return_exceptions=True
        )
        documents: list[WebDocument] = []
        for url, result in zip(urls, results):
            if isinstance(result, Exception):
                await bus.log(f"[user-researcher] Could not read {url}: {result}")
            else:
                documents.append(result)
        return documents

    @staticmethod
    def _urls_in(text: str) -> list[str]:
        return re.findall(r"https?://[^\s<>\]\[()]+", text)

    @staticmethod
    def _build_prompt(
        ctx: EnvGenContext,
        documents: list[WebDocument],
        *,
        source_char_budget: int = 24_000,
    ) -> str:
        remaining_source_chars = source_char_budget
        bounded_excerpts: list[str] = []
        for doc in documents:
            if remaining_source_chars <= 0:
                break
            excerpt = doc.text[:remaining_source_chars]
            bounded_excerpts.append(
                f"SOURCE: {doc.title}\nURL: {doc.url}\nEXCERPT: {excerpt}"
            )
            remaining_source_chars -= len(excerpt)
        excerpts = "\n\n".join(bounded_excerpts) or (
            "No web sources were available; rely on the structured spec and label no assumptions."
        )
        return (
            f"USER REQUEST:\n{ctx.description}\n\n"
            f"STRUCTURED APP SPEC:\n{ctx.compiler_input.model_dump_json(indent=2)}\n\n"
            f"EXTERNAL REFERENCES:\n{excerpts}"
        )

    @staticmethod
    def _fallback_brief(ctx: EnvGenContext, documents: list[WebDocument]) -> ApplicationResearchBrief:
        actions = [action.name for action in ctx.compiler_input.actions]
        entities = [entity.name for entity in ctx.compiler_input.entities]
        return ApplicationResearchBrief(
            product_summary=ctx.description,
            workflows=[f"User performs {action}" for action in actions],
            functional_requirements=[f"Support the {action} action" for action in actions],
            data_requirements=[f"Persist {entity} state" for entity in entities],
            rl_observations=[f"Expose {entity} state to the RL agent" for entity in entities],
            sources=[ResearchSource(title=doc.title, url=doc.url) for doc in documents],
        )
