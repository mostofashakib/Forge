from pathlib import Path
from forge.customization.config import load_config, EnvConfig, RewardConfig, ObservationConfig


def test_load_config_returns_defaults_when_no_file(tmp_path):
    config = load_config(tmp_path)
    assert isinstance(config, EnvConfig)
    assert config.reward.base_success == 1.0
    assert config.reward.step_penalty == 0.01
    assert config.reward.max_reward == 1.0
    assert config.reward.min_reward == -1.0


def test_load_config_reads_reward_values(tmp_path):
    custom_dir = tmp_path / "custom"
    custom_dir.mkdir()
    (custom_dir / "config.yaml").write_text(
        "reward:\n  base_success: 2.0\n  step_penalty: 0.05\n"
    )
    config = load_config(tmp_path)
    assert config.reward.base_success == 2.0
    assert config.reward.step_penalty == 0.05


def test_load_config_partial_override_keeps_defaults(tmp_path):
    custom_dir = tmp_path / "custom"
    custom_dir.mkdir()
    (custom_dir / "config.yaml").write_text("reward:\n  base_success: 0.5\n")
    config = load_config(tmp_path)
    assert config.reward.base_success == 0.5
    assert config.reward.step_penalty == 0.01  # default kept


def test_load_config_observation_section(tmp_path):
    custom_dir = tmp_path / "custom"
    custom_dir.mkdir()
    (custom_dir / "config.yaml").write_text(
        "observation:\n"
        "  mode: role_based\n"
        "  actor_role: support_agent\n"
        "  visible_entities:\n"
        "    - tickets\n"
        "  hidden_entities:\n"
        "    - billing_notes\n"
    )
    config = load_config(tmp_path)
    assert config.observation.mode == "role_based"
    assert config.observation.actor_role == "support_agent"
    assert "tickets" in config.observation.visible_entities
    assert "billing_notes" in config.observation.hidden_entities


def test_load_config_empty_yaml_returns_defaults(tmp_path):
    custom_dir = tmp_path / "custom"
    custom_dir.mkdir()
    (custom_dir / "config.yaml").write_text("")
    config = load_config(tmp_path)
    assert config.reward.base_success == 1.0
