import pytest
from forge.schema.state_schema import FieldSpec, StateSchemaManifest


def _manifest() -> StateSchemaManifest:
    return StateSchemaManifest(
        env_name="email_env",
        fields={
            "inbox_count": FieldSpec(type="integer"),
            "selected_email": FieldSpec(type="object"),
            "last_updated": FieldSpec(type="datetime", volatile=True),
            "search_results": FieldSpec(type="array", derived_from=["search"]),
        },
    )


def test_stable_fields_excludes_volatile():
    m = _manifest()
    stable = m.stable_fields()
    assert "inbox_count" in stable
    assert "selected_email" in stable
    assert "search_results" in stable
    assert "last_updated" not in stable


def test_coverage_score_all_present():
    m = _manifest()
    state = {"inbox_count": 3, "selected_email": {}, "last_updated": "t", "search_results": []}
    assert m.coverage_score(state) == 1.0


def test_coverage_score_partial():
    m = _manifest()
    state = {"inbox_count": 3}
    # All 4 fields are required=True by default (volatile fields still count). 1 of 4 present → 0.25
    score = m.coverage_score(state)
    assert score == pytest.approx(0.25)


def test_coverage_score_empty_state():
    m = _manifest()
    assert m.coverage_score({}) == 0.0


def test_missing_fields_returns_absent_required():
    m = _manifest()
    state = {"inbox_count": 5, "last_updated": "t", "search_results": []}
    missing = m.missing_fields(state)
    assert "selected_email" in missing
    assert "inbox_count" not in missing


def test_missing_fields_empty_when_all_present():
    m = _manifest()
    state = {"inbox_count": 3, "selected_email": {}, "last_updated": "t", "search_results": []}
    assert m.missing_fields(state) == []


def test_state_changed_detects_stable_field_change():
    m = _manifest()
    before = {"inbox_count": 3, "selected_email": {}, "last_updated": "2024-01-01T00:00:00"}
    after  = {"inbox_count": 4, "selected_email": {}, "last_updated": "2024-01-01T00:01:00"}
    assert m.state_changed(before, after) is True


def test_state_changed_ignores_volatile_only_change():
    m = _manifest()
    before = {"inbox_count": 3, "selected_email": {}, "last_updated": "2024-01-01T00:00:00"}
    after  = {"inbox_count": 3, "selected_email": {}, "last_updated": "2024-01-01T00:01:00"}
    assert m.state_changed(before, after) is False


def test_state_changed_false_when_nothing_changes():
    m = _manifest()
    state = {"inbox_count": 3, "selected_email": {}}
    assert m.state_changed(state, state.copy()) is False


def test_manifest_serializes_and_round_trips():
    m = _manifest()
    json_str = m.model_dump_json()
    m2 = StateSchemaManifest.model_validate_json(json_str)
    assert m2.env_name == m.env_name
    assert m2.fields.keys() == m.fields.keys()
    assert m2.fields["last_updated"].volatile is True
    assert m2.fields["search_results"].derived_from == ["search"]
