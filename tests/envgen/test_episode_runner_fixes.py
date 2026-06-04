import pytest
from forge.envgen.episode_runner import HashNormalizer, EpisodeConfig
from forge.schema.state_schema import StateSchemaManifest, FieldSpec


def _manifest() -> StateSchemaManifest:
    return StateSchemaManifest(
        env_name="email_env",
        fields={
            "inbox_count": FieldSpec(type="integer"),
            "last_sync": FieldSpec(type="datetime", volatile=True),
        },
    )


def test_hash_normalizer_strips_volatile():
    m = _manifest()
    normalizer = HashNormalizer(manifest=m)
    state_a = {"inbox_count": 3, "last_sync": "2024-01-01T00:00:00"}
    state_b = {"inbox_count": 3, "last_sync": "2024-01-01T00:01:00"}
    # Only timestamp changed — hashes should be equal
    assert normalizer.hash(state_a) == normalizer.hash(state_b)


def test_hash_normalizer_detects_real_change():
    m = _manifest()
    normalizer = HashNormalizer(manifest=m)
    state_a = {"inbox_count": 3, "last_sync": "2024-01-01T00:00:00"}
    state_b = {"inbox_count": 4, "last_sync": "2024-01-01T00:00:00"}
    assert normalizer.hash(state_a) != normalizer.hash(state_b)


def test_hash_normalizer_fallback_without_manifest():
    normalizer = HashNormalizer(manifest=None)
    state = {"inbox_count": 3, "ts": "2024-01-01"}
    h = normalizer.hash(state)
    assert isinstance(h, str) and len(h) > 0


def test_episode_config_new_defaults():
    cfg = EpisodeConfig(base_url="http://localhost:8080", objective="test")
    assert cfg.consecutive_below_threshold == 8
    assert cfg.diff_floor == 0.1


def test_episode_config_diff_floor_configurable():
    cfg = EpisodeConfig(base_url="http://localhost:8080", objective="test", diff_floor=0.0)
    assert cfg.diff_floor == 0.0
