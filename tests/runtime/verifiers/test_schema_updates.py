from forge.extraction.schemas import SuccessCondition
from forge.customization.config import RewardConfig, load_config


def test_success_condition_has_rubric_field():
    cond = SuccessCondition(type="semantic_check", expression="reply_text", rubric="Must be polite")
    assert cond.rubric == "Must be polite"


def test_success_condition_rubric_defaults_to_empty():
    cond = SuccessCondition(type="state_check", expression="x > 0")
    assert cond.rubric == ""


def test_reward_config_has_semantic_weight():
    cfg = RewardConfig()
    assert cfg.semantic_weight == 0.0


def test_reward_config_has_invalid_action_penalty():
    cfg = RewardConfig()
    assert cfg.invalid_action_penalty == 0.5


def test_load_config_reads_semantic_weight(tmp_path):
    custom_dir = tmp_path / "custom"
    custom_dir.mkdir()
    (custom_dir / "config.yaml").write_text(
        "reward:\n  semantic_weight: 0.3\n  invalid_action_penalty: 0.2\n"
    )
    cfg = load_config(tmp_path)
    assert cfg.reward.semantic_weight == 0.3
    assert cfg.reward.invalid_action_penalty == 0.2
