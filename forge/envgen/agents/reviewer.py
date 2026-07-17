from __future__ import annotations

import ast
import asyncio
from enum import StrEnum

from pydantic import BaseModel, Field

from forge.envgen.agents.base import EnvGenAgent
from forge.envgen.artifact_bus import ArtifactBus
from forge.envgen.context import EnvGenContext
from forge.extraction.llm_client import LLMClient, get_client
from forge.envgen.config import envgen_config


class ReviewSeverity(StrEnum):
    ERROR = "error"
    WARNING = "warning"


class ReviewIssue(BaseModel):
    severity: ReviewSeverity
    category: str
    message: str
    artifact: str | None = None


class GenerationReview(BaseModel):
    approved: bool
    requirements_checked: list[str] = Field(default_factory=list)
    issues: list[ReviewIssue] = Field(default_factory=list)


class GenerationReviewError(RuntimeError):
    def __init__(self, review: GenerationReview) -> None:
        errors = [issue.message for issue in review.issues if issue.severity == ReviewSeverity.ERROR]
        super().__init__("Generated environment failed review: " + "; ".join(errors))
        self.review = review


class RequirementAssessment(BaseModel):
    requirements_met: bool
    findings: list[str] = Field(default_factory=list)


_REVIEW_SYSTEM = (
    "You are the final reviewer for a generated reinforcement-learning environment. "
    "Compare the user's request and structured domain requirements to the supplied artifacts. "
    "Check functional coverage, UI-to-API action coverage, RL state/reward suitability, and "
    "clear code responsibilities. Report only concrete unmet requirements or code-quality "
    "problems; do not reject for subjective style preferences."
)


class ReviewerPrompts:
    SYSTEM = _REVIEW_SYSTEM


class ReviewerAgent(EnvGenAgent):
    """Static and semantic quality gate for generated code and requirements."""

    agent_id = "reviewer"
    depends_on = [
        "app_code",
        "instrumented_code",
        "state_bridge_code",
        "state_schema_manifest",
        "policy_dsl",
        "reward_fn_code",
    ]
    optional_depends_on = ["reviewer_research"]
    produces = ["review_report"]

    def __init__(
        self,
        client: LLMClient | None = None,
        *,
        semantic_review: bool = True,
    ) -> None:
        self._semantic_review = semantic_review
        self._client = client or (
            get_client(max_tokens=envgen_config().standard_llm_tokens, capable=True)
            if semantic_review else None
        )

    async def run(self, ctx: EnvGenContext, bus: ArtifactBus) -> None:
        artifacts = {name: await bus.wait_for(name) for name in self.depends_on}
        artifacts.update({name: bus.get(name) for name in self.optional_depends_on})
        issues: list[ReviewIssue] = []
        app_code: dict[str, str] = artifacts["app_code"] or {}

        required_files = {"main.py", "ui.html", "requirements.txt", "Dockerfile"}
        for path in sorted(required_files - set(app_code)):
            issues.append(self._error("structure", f"Required file {path!r} is missing", path))

        for path, content in app_code.items():
            if not content.strip():
                issues.append(self._error("completeness", "Generated file is empty", path))
                continue
            if path.endswith(".py"):
                try:
                    ast.parse(content, filename=path)
                except SyntaxError as exc:
                    issues.append(self._error(
                        "syntax", f"Python does not parse: {exc.msg} at line {exc.lineno}", path
                    ))
            lowered = content.lower()
            if "todo" in lowered or "fixme" in lowered or "lorem ipsum" in lowered:
                issues.append(ReviewIssue(
                    severity=ReviewSeverity.WARNING,
                    category="quality",
                    message="Generated file contains placeholder text",
                    artifact=path,
                ))

        generated_python = {
            **{
                f"instrumented:{path}": content
                for path, content in (artifacts["instrumented_code"] or {}).items()
                if path.endswith(".py")
            },
            "state_bridge_code": artifacts["state_bridge_code"] or "",
            "reward_fn_code": artifacts["reward_fn_code"] or "",
        }
        for artifact_name, content in generated_python.items():
            if not content:
                continue
            try:
                ast.parse(content, filename=artifact_name)
            except SyntaxError as exc:
                issues.append(self._error(
                    "syntax",
                    f"Python does not parse: {exc.msg} at line {exc.lineno}",
                    artifact_name,
                ))

        backend_text = "\n".join(
            content for path, content in app_code.items() if path.endswith(".py")
        )
        for endpoint in (
            "/forge/health", "/forge/state", "/forge/reset",
            "/forge/snapshot", "/forge/restore", "/forge/restore-state",
        ):
            if endpoint not in backend_text:
                issues.append(self._error(
                    "requirements", f"Required Forge endpoint {endpoint!r} is missing", "main.py"
                ))
        for action in ctx.compiler_input.actions:
            if action.name not in backend_text:
                issues.append(self._error(
                    "requirements", f"Declared action {action.name!r} is not implemented"
                ))

        ui = app_code.get("ui.html", "").lower()
        if ui and not all(token in ui for token in ("<html", "<script", "</html>")):
            issues.append(self._error(
                "ui", "ui.html must contain a complete HTML document with client behavior", "ui.html"
            ))
        for action in ctx.compiler_input.actions:
            if ui and action.name.lower() not in ui:
                issues.append(self._error(
                    "requirements",
                    f"Declared action {action.name!r} is not exposed by the UI",
                    "ui.html",
                ))

        for artifact_name in self.depends_on[1:]:
            value = artifacts[artifact_name]
            if value is None or value == "" or value == {}:
                issues.append(self._error(
                    "artifact", f"Specialist output {artifact_name!r} is empty", artifact_name
                ))

        requirements = [
            ctx.description,
            f"Domain: {ctx.compiler_input.domain}",
            f"Actions: {', '.join(action.name for action in ctx.compiler_input.actions) or 'none'}",
            f"Policy requirements: {ctx.policy_requirements or 'default safety policy'}",
            f"Reward requirements: {ctx.reward_requirements or 'default task reward'}",
        ]
        if self._semantic_review and self._client is not None:
            review_chars = envgen_config().generated_file_review_chars
            artifact_excerpt = "\n\n".join(
                f"=== {path} ===\n{content[:review_chars]}"
                for path, content in app_code.items()
            )
            researched_context = artifacts["reviewer_research"]
            research_section = (
                f"Researched product context:\n{researched_context.as_prompt()}\n\n"
                if researched_context is not None
                else ""
            )
            semantic_input = (
                f"User request: {ctx.description}\n"
                f"Domain: {ctx.compiler_input.domain}\n"
                f"Entities: {[entity.model_dump() for entity in ctx.compiler_input.entities]}\n"
                f"Actions: {[action.model_dump() for action in ctx.compiler_input.actions]}\n"
                f"Policy requirements: {ctx.policy_requirements or 'default'}\n"
                f"Reward requirements: {ctx.reward_requirements or 'default'}\n\n"
                f"{research_section}"
                f"Generated application:\n{artifact_excerpt}\n\n"
                f"State bridge:\n{str(artifacts['state_bridge_code'])[:8000]}\n\n"
                f"Policy:\n{str(artifacts['policy_dsl'])[:4000]}\n\n"
                f"Reward:\n{str(artifacts['reward_fn_code'])[:8000]}"
            )
            loop = asyncio.get_running_loop()
            assessment: RequirementAssessment = await loop.run_in_executor(
                None,
                lambda: self._client.extract(
                    system=ReviewerPrompts.SYSTEM,
                    user=semantic_input,
                    schema=RequirementAssessment,
                ),
            )
            severity = (
                ReviewSeverity.WARNING if assessment.requirements_met else ReviewSeverity.ERROR
            )
            findings = assessment.findings or (
                ["Semantic reviewer found unmet user requirements"]
                if not assessment.requirements_met else []
            )
            issues.extend(
                ReviewIssue(
                    severity=severity,
                    category="semantic_review",
                    message=finding,
                )
                for finding in findings
            )
        review = GenerationReview(
            approved=not any(issue.severity == ReviewSeverity.ERROR for issue in issues),
            requirements_checked=requirements,
            issues=issues,
        )
        await bus.publish("review_report", review)

    @staticmethod
    def _error(category: str, message: str, artifact: str | None = None) -> ReviewIssue:
        return ReviewIssue(
            severity=ReviewSeverity.ERROR,
            category=category,
            message=message,
            artifact=artifact,
        )
