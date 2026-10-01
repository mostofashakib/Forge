from __future__ import annotations

import ast
import asyncio
from enum import StrEnum

from pydantic import BaseModel, Field

from forge.envgen.agents.base import EnvGenAgent
from forge.envgen.artifact_bus import ArtifactBus
from forge.envgen.context import EnvGenContext
from forge.envgen.agents.semantic_review import (
    SemanticReviewPanel,
    SemanticReviewPrompts,
)
from forge.extraction.llm_client import (
    LLMClient,
    generation_models,
    get_judge_client,
)
from forge.grading_provenance import model_family
from forge.validation.quorum import configured_quorum
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


# The review prompt and its output schema live with the panel that uses them.
ReviewerPrompts = SemanticReviewPrompts


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
        self._client: LLMClient | None = None
        self._panel: SemanticReviewPanel | None = None
        if not semantic_review:
            return

        # The semantic reviewer judges artifacts an LLM wrote, so it must not be
        # that LLM. It resolves through the judge client, and through an
        # independent quorum when FORGE_QUORUM_MODELS declares one.
        quorum_specs = configured_quorum()
        # Only build the single judge when it will actually be consulted:
        # get_client constructs the provider SDK eagerly, so an unused judge
        # would make its credentials mandatory for a quorum-only setup.
        if client is not None or not quorum_specs:
            self._client = client or get_judge_client(
                max_tokens=envgen_config().standard_llm_tokens
            )
        families = tuple(model_family(model) for model in generation_models())
        self._panel = SemanticReviewPanel(
            generator_families=families,
            client=self._client,
            quorum_specs=quorum_specs,
        )

    async def run(self, ctx: EnvGenContext, bus: ArtifactBus) -> None:
        artifacts = {name: await bus.wait_for(name) for name in self.depends_on}
        artifacts.update({name: bus.get(name) for name in self.optional_depends_on})
        app_code: dict[str, str] = artifacts["app_code"] or {}
        actions = [action.name for action in ctx.compiler_input.actions]

        issues: list[ReviewIssue] = [
            *self._structure_issues(app_code, with_ui=ctx.with_ui),
            *self._file_issues(app_code),
            *self._generated_python_issues(artifacts),
            *self._endpoint_issues(app_code, actions),
            # Contract conformance. Static, like the determinism gate — it asks
            # whether the generated code declares the shape the runtime requires,
            # which the semantic reviewer cannot check reliably.
            *self._contract_issues(
                artifacts["state_bridge_code"] or "",
                artifacts["reward_fn_code"] or "",
            ),
            *self._ui_issues(app_code.get("ui.html", "").lower(), actions),
            *self._empty_artifact_issues(artifacts),
        ]
        if self._semantic_review and self._panel is not None:
            issues.extend(await self._semantic_issues(ctx, artifacts, app_code))

        requirements = [
            ctx.description,
            f"Domain: {ctx.compiler_input.domain}",
            f"Actions: {', '.join(actions) or 'none'}",
            f"Policy requirements: {ctx.policy_requirements or 'default safety policy'}",
            f"Reward requirements: {ctx.reward_requirements or 'default task reward'}",
        ]
        review = GenerationReview(
            approved=not any(issue.severity == ReviewSeverity.ERROR for issue in issues),
            requirements_checked=requirements,
            issues=issues,
        )
        await bus.publish("review_report", review)

    def _structure_issues(self, app_code: dict[str, str], *, with_ui: bool) -> list[ReviewIssue]:
        required_files = {"main.py", "requirements.txt", "Dockerfile"}
        # A headless environment has no UI specialist, so ui.html is never
        # generated and must not be demanded of it.
        if with_ui:
            required_files.add("ui.html")
        return [
            self._error("structure", f"Required file {path!r} is missing", path)
            for path in sorted(required_files - set(app_code))
        ]

    def _file_issues(self, app_code: dict[str, str]) -> list[ReviewIssue]:
        issues: list[ReviewIssue] = []
        for path, content in app_code.items():
            if not content.strip():
                issues.append(self._error("completeness", "Generated file is empty", path))
                continue
            if path.endswith(".py"):
                issues.extend(self._syntax_issues(content, path))
            lowered = content.lower()
            if "todo" in lowered or "fixme" in lowered or "lorem ipsum" in lowered:
                issues.append(ReviewIssue(
                    severity=ReviewSeverity.WARNING,
                    category="quality",
                    message="Generated file contains placeholder text",
                    artifact=path,
                ))
        return issues

    def _generated_python_issues(self, artifacts: dict) -> list[ReviewIssue]:
        generated_python = {
            **{
                f"instrumented:{path}": content
                for path, content in (artifacts["instrumented_code"] or {}).items()
                if path.endswith(".py")
            },
            "state_bridge_code": artifacts["state_bridge_code"] or "",
            "reward_fn_code": artifacts["reward_fn_code"] or "",
        }
        return [
            issue
            for name, content in generated_python.items() if content
            for issue in self._syntax_issues(content, name)
        ]

    def _syntax_issues(self, content: str, artifact: str) -> list[ReviewIssue]:
        try:
            ast.parse(content, filename=artifact)
        except SyntaxError as exc:
            return [self._error("syntax", f"Python does not parse: {exc.msg} at line {exc.lineno}", artifact)]
        return []

    def _endpoint_issues(self, app_code: dict[str, str], actions: list[str]) -> list[ReviewIssue]:
        backend_text = "\n".join(
            content for path, content in app_code.items() if path.endswith(".py")
        )
        issues = [
            self._error("requirements", f"Required Forge endpoint {endpoint!r} is missing", "main.py")
            for endpoint in (
                "/forge/health", "/forge/state", "/forge/reset",
                "/forge/snapshot", "/forge/restore", "/forge/restore-state",
            )
            if endpoint not in backend_text
        ]
        issues.extend(
            self._error("requirements", f"Declared action {action!r} is not implemented")
            for action in actions
            if action not in backend_text
        )
        return issues

    def _ui_issues(self, ui: str, actions: list[str]) -> list[ReviewIssue]:
        if not ui:
            return []
        issues = []
        if not all(token in ui for token in ("<html", "<script", "</html>")):
            issues.append(self._error(
                "ui", "ui.html must contain a complete HTML document with client behavior", "ui.html"
            ))
        issues.extend(
            self._error("requirements", f"Declared action {action!r} is not exposed by the UI", "ui.html")
            for action in actions
            if action.lower() not in ui
        )
        return issues

    def _empty_artifact_issues(self, artifacts: dict) -> list[ReviewIssue]:
        return [
            self._error("artifact", f"Specialist output {name!r} is empty", name)
            for name in self.depends_on[1:]
            if artifacts[name] is None or artifacts[name] == "" or artifacts[name] == {}
        ]

    async def _semantic_issues(self, ctx: EnvGenContext, artifacts: dict, app_code: dict[str, str]) -> list[ReviewIssue]:
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
        result = await loop.run_in_executor(None, lambda: self._panel.assess(semantic_input))
        if result.contested:
            # The panel disagreed. That is not approval: a contested gate
            # means the artifacts are not established as correct, and the
            # dissenting findings are exactly what the repair loop needs.
            # Labelled separately so a reader can tell "reviewers
            # disagreed" from "reviewers rejected".
            findings = result.findings or ["Semantic reviewers could not reach agreement"]
            return [
                ReviewIssue(severity=ReviewSeverity.ERROR, category="semantic_review_contested", message=finding)
                for finding in findings
            ]
        severity = ReviewSeverity.WARNING if result.requirements_met else ReviewSeverity.ERROR
        findings = result.findings or (
            ["Semantic reviewer found unmet user requirements"] if not result.requirements_met else []
        )
        return [
            ReviewIssue(severity=severity, category="semantic_review", message=finding)
            for finding in findings
        ]

    @staticmethod
    def _error(category: str, message: str, artifact: str | None = None) -> ReviewIssue:
        return ReviewIssue(
            severity=ReviewSeverity.ERROR,
            category=category,
            message=message,
            artifact=artifact,
        )

    @staticmethod
    def _subclasses(source: str, base: str) -> list[ast.ClassDef]:
        tree = ast.parse(source)
        return [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef)
            and any(
                (isinstance(b, ast.Name) and b.id == base)
                or (isinstance(b, ast.Attribute) and b.attr == base)
                for b in node.bases
            )
        ]

    def _contract_issues(
        self, state_bridge_code: str, reward_fn_code: str
    ) -> list[ReviewIssue]:
        # Unparseable source is already reported by the syntax loop above;
        # do not also report a spurious "no contract subclass found" here.
        issues: list[ReviewIssue] = []

        if state_bridge_code:
            try:
                bridges = self._subclasses(state_bridge_code, "Environment")
            except SyntaxError:
                bridges = None
            if bridges is not None and not bridges:
                issues.append(self._error(
                    "contract",
                    "The state bridge must subclass forge.contracts.Environment",
                    "state_bridge_code",
                ))

        if reward_fn_code:
            try:
                rubrics = self._subclasses(reward_fn_code, "Rubric")
            except SyntaxError:
                rubrics = None
            if rubrics is not None:
                if not rubrics:
                    issues.append(self._error(
                        "contract",
                        "The reward must subclass forge.contracts.Rubric",
                        "reward_fn_code",
                    ))
                elif not any(
                    isinstance(item, ast.FunctionDef) and item.name == "score"
                    for rubric in rubrics
                    for item in rubric.body
                ):
                    # `score` may be entirely absent, or present but defined
                    # `async def` (which fails the `ast.FunctionDef` check
                    # above, since `ast.AsyncFunctionDef` is a distinct node
                    # type). An automated repair specialist reading "must
                    # define score()" while `score` is visibly present could
                    # plausibly "fix" this by adding a second, sync `score`
                    # rather than removing `async` from the existing one —
                    # so the two cases get distinct messages.
                    if any(
                        isinstance(item, ast.AsyncFunctionDef) and item.name == "score"
                        for rubric in rubrics
                        for item in rubric.body
                    ):
                        issues.append(self._error(
                            "contract",
                            "score() is async; Rubric.score is synchronous — "
                            "remove `async` from the existing score(), do not "
                            "add a second one",
                            "reward_fn_code",
                        ))
                    else:
                        issues.append(self._error(
                            "contract",
                            "The Rubric subclass must define score(); without it the "
                            "reward cannot be registered",
                            "reward_fn_code",
                        ))

        return issues
