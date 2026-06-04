import pytest
import httpx
import respx
from forge.envgen.post_generation_validator import PostGenerationValidator, ValidationResult
from forge.schema.state_schema import StateSchemaManifest, FieldSpec


def _manifest_with_derived() -> StateSchemaManifest:
    return StateSchemaManifest(
        env_name="email_env",
        fields={
            "inbox_count": FieldSpec(type="integer"),
            "search_results": FieldSpec(type="array", derived_from=["search"]),
        },
    )


def _manifest_simple() -> StateSchemaManifest:
    return StateSchemaManifest(
        env_name="email_env",
        fields={
            "inbox_count": FieldSpec(type="integer"),
            "selected_email": FieldSpec(type="object"),
        },
    )


@respx.mock
def test_validate_passes_when_all_fields_present():
    respx.post("http://container/forge/reset").respond(200, json={})
    respx.get("http://container/forge/state").respond(
        200, json={"inbox_count": 5, "selected_email": {}}
    )
    validator = PostGenerationValidator(base_url="http://container")
    result = validator.validate(_manifest_simple())
    assert result.passed is True
    assert result.missing_fields == []
    assert result.coverage_score == 1.0


@respx.mock
def test_validate_fails_when_fields_missing():
    respx.post("http://container/forge/reset").respond(200, json={})
    respx.get("http://container/forge/state").respond(
        200, json={"inbox_count": 5}
    )
    validator = PostGenerationValidator(base_url="http://container")
    result = validator.validate(_manifest_simple())
    assert result.passed is False
    assert "selected_email" in result.missing_fields


@respx.mock
def test_validate_checks_derived_fields_after_action():
    # State after reset: search_results missing
    # State after /search action: search_results present
    state_calls = iter([
        {"inbox_count": 5},                       # after reset
        {"inbox_count": 5, "search_results": []}, # after /search
    ])

    def state_side_effect(request):
        return httpx.Response(200, json=next(state_calls))

    respx.post("http://container/forge/reset").respond(200, json={})
    respx.post("http://container/search").respond(200, json={})
    respx.get("http://container/forge/state").mock(side_effect=state_side_effect)

    validator = PostGenerationValidator(base_url="http://container")
    result = validator.validate(_manifest_with_derived())
    assert result.passed is True
    assert result.missing_fields == []


@respx.mock
def test_validate_fails_when_derived_field_never_populated():
    # search_results never appears even after /search
    respx.post("http://container/forge/reset").respond(200, json={})
    respx.post("http://container/search").respond(200, json={})
    respx.get("http://container/forge/state").respond(
        200, json={"inbox_count": 5}
    )
    validator = PostGenerationValidator(base_url="http://container")
    result = validator.validate(_manifest_with_derived())
    assert result.passed is False
    assert "search_results" in result.missing_fields


@respx.mock
def test_validate_returns_coverage_score():
    respx.post("http://container/forge/reset").respond(200, json={})
    respx.get("http://container/forge/state").respond(
        200, json={"inbox_count": 5}  # selected_email missing → 0.5 coverage
    )
    validator = PostGenerationValidator(base_url="http://container")
    result = validator.validate(_manifest_simple())
    assert result.coverage_score == pytest.approx(0.5)
