import json
import tempfile
from pathlib import Path
import pytest
from typer.testing import CliRunner
from forge.cli.main import app
from forge.extraction.schemas import (
    CompilerInput, EntityDef, FieldDef, ActionDef, ActionParam,
    TaskTemplate, SuccessCondition,
)

runner = CliRunner()


def _counter_input() -> CompilerInput:
    return CompilerInput(
        project_name="counter_env",
        domain="counter",
        entities=[EntityDef(name="counter", fields=[
            FieldDef(name="id", type="string"),
            FieldDef(name="value", type="integer", default=0),
        ])],
        actions=[ActionDef(
            name="increment",
            params=[ActionParam(name="counter_id", type="string")],
            mutates=["counter.value"],
        )],
        tasks=[TaskTemplate(
            name="reach_target",
            description="Reach target",
            success_conditions=[SuccessCondition(type="state_check", expression="done")],
        )],
    )


def test_cli_has_compile_command():
    result = runner.invoke(app, ["compile", "--help"])
    assert result.exit_code == 0
    assert "compile" in result.output.lower() or "input" in result.output.lower()


def test_cli_has_validate_command():
    result = runner.invoke(app, ["validate", "--help"])
    assert result.exit_code == 0


def test_cli_has_run_command():
    result = runner.invoke(app, ["run", "--help"])
    assert result.exit_code == 0


def test_cli_has_export_command():
    result = runner.invoke(app, ["export", "--help"])
    assert result.exit_code == 0


def test_compile_command_builds_package(tmp_path):
    ci = _counter_input()
    input_file = tmp_path / "compiler_input.json"
    input_file.write_text(ci.model_dump_json())
    output_dir = tmp_path / "out"

    result = runner.invoke(app, [
        "compile",
        "--input", str(input_file),
        "--output", str(output_dir),
    ])
    assert result.exit_code == 0, result.output
    assert (output_dir / "counter_env" / "gym_wrapper.py").exists()


def test_compile_command_fails_on_invalid_json(tmp_path):
    bad_file = tmp_path / "bad.json"
    bad_file.write_text("not valid json {{{")
    result = runner.invoke(app, ["compile", "--input", str(bad_file), "--output", str(tmp_path)])
    assert result.exit_code != 0


def test_validate_command_on_built_package(tmp_path):
    from forge.compiler.package_builder import PackageBuilder
    ci = _counter_input()
    pkg_dir = PackageBuilder(tmp_path).build(ci)

    result = runner.invoke(app, ["validate", str(pkg_dir)])
    assert result.exit_code == 0, result.output
    assert "passed" in result.output.lower() or "valid" in result.output.lower()


def test_validate_command_fails_on_nonexistent_dir(tmp_path):
    result = runner.invoke(app, ["validate", str(tmp_path / "does_not_exist")])
    assert result.exit_code != 0


def test_cli_has_replay_command():
    result = runner.invoke(app, ["replay", "--help"])
    assert result.exit_code == 0
    assert "episode" in result.output.lower()


def test_replay_missing_episode(tmp_path):
    db_path = tmp_path / "test.db"
    db_url = f"sqlite:///{db_path}"

    # Create the schema so the DB exists but is empty
    from sqlalchemy import create_engine
    from backend.app.models import Base
    engine = create_engine(db_url, connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)

    result = runner.invoke(app, ["replay", "ep_does_not_exist", "--db", db_url])
    assert result.exit_code != 0
    assert "not found" in result.output.lower() or "not found" in (result.stderr or "").lower()


def test_replay_renders_episode(tmp_path):
    import datetime
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from backend.app.models import Base, Episode, EpisodeStep

    db_url = f"sqlite:///{tmp_path / 'test.db'}"
    engine = create_engine(db_url, connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)

    with Session() as db:
        ep = Episode(
            id="ep_test_001",
            env_name="my_env",
            task_name="do_thing",
            seed=7,
            agent_id="random",
            status="completed",
            total_steps=2,
            total_reward=1.5,
            passed=True,
            started_at=datetime.datetime.now(datetime.timezone.utc),
        )
        db.add(ep)
        db.add(EpisodeStep(
            episode_id="ep_test_001",
            step_index=0,
            action='{"type": "increment"}',
            reward=0.5,
            verifier_results='[{"verifier_id": "v1", "passed": false, "score": 0.5}]',
            diff='{"counter": [0, 1]}',
            events='[]',
            state_hash_before="aaa",
            state_hash_after="bbb",
            terminated=False,
            truncated=False,
        ))
        db.add(EpisodeStep(
            episode_id="ep_test_001",
            step_index=1,
            action='{"type": "submit"}',
            reward=1.0,
            verifier_results='[{"verifier_id": "v1", "passed": true, "score": 1.0}]',
            diff='{}',
            events='[{"type": "policy_violation", "rule_id": "no_spam"}]',
            state_hash_before="bbb",
            state_hash_after="ccc",
            terminated=True,
            truncated=False,
        ))
        db.commit()

    result = runner.invoke(app, ["replay", "ep_test_001", "--db", db_url])
    assert result.exit_code == 0, result.output
    assert "ep_test_001" in result.output
    assert "my_env" in result.output
    assert "increment" in result.output
    assert "submit" in result.output
    assert "counter" in result.output
    assert "no_spam" in result.output
    assert "[DONE]" in result.output


def test_replay_container_episode(tmp_path):
    """forge replay on a cep_* episode reads from JSONL."""
    import datetime
    import json as _json
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from backend.app.models import Base, AgentRun, AgentEpisode

    db_url = f"sqlite:///{tmp_path / 'test.db'}"
    engine = create_engine(db_url, connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)

    # Write a minimal JSONL file
    jsonl_path = tmp_path / "cep_abc_001.jsonl"
    steps = [
        {"step_index": 0, "action": {"endpoint": "/do_thing", "payload": {"x": 1}}, "reward": 0.5,
         "objective_score": 0.5, "state_hash_before": "aaa", "state_hash_after": "bbb",
         "terminated": False, "truncated": False, "termination_reason": None},
        {"step_index": 1, "action": {"endpoint": "/search", "payload": {"q": "foo"}}, "reward": 0.0,
         "objective_score": 0.0, "state_hash_before": "bbb", "state_hash_after": "bbb",
         "terminated": False, "truncated": False, "termination_reason": "diverged"},
        {"type": "episode_summary", "episode_id": "cep_abc_001", "total_steps": 2,
         "total_reward": 0.125, "final_objective_score": 0.0, "termination_reason": "diverged"},
    ]
    jsonl_path.write_text("\n".join(_json.dumps(s) for s in steps))

    with Session() as db:
        run = AgentRun(
            id="run_abc",
            env_name="test_env",
            agent_id="random",
            objective="do things",
            num_episodes=1,
            created_at=datetime.datetime.now(datetime.timezone.utc),
        )
        db.add(run)
        ep = AgentEpisode(
            id="cep_abc_001",
            run_id="run_abc",
            episode_index=0,
            seed=0,
            status="completed",
            total_steps=2,
            total_reward=0.125,
            final_objective_score=0.0,
            termination_reason="diverged",
            jsonl_path=str(jsonl_path),
            started_at=datetime.datetime.now(datetime.timezone.utc),
        )
        db.add(ep)
        db.commit()

    result = runner.invoke(app, ["replay", "cep_abc_001", "--db", db_url])
    assert result.exit_code == 0, result.output
    assert "cep_abc_001" in result.output
    assert "/do_thing" in result.output
    assert "/search" in result.output
    assert "state unchanged" in result.output
    assert "diverged" in result.output


def test_diagnose_command(tmp_path):
    """forge diagnose surfaces dead-end and reward issues."""
    import datetime
    import json as _json
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from backend.app.models import Base, AgentRun, AgentEpisode

    db_url = f"sqlite:///{tmp_path / 'test.db'}"
    engine = create_engine(db_url, connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)

    jsonl_path = tmp_path / "cep_diag_001.jsonl"
    steps = [
        {"step_index": 0, "action": {"endpoint": "/noop"}, "reward": 0.0,
         "objective_score": 0.0, "state_hash_before": "aaa", "state_hash_after": "aaa",
         "terminated": False, "truncated": False, "termination_reason": None},
        {"type": "episode_summary", "episode_id": "cep_diag_001", "total_steps": 1,
         "total_reward": 0.0, "final_objective_score": 0.0, "termination_reason": "diverged"},
    ]
    jsonl_path.write_text("\n".join(_json.dumps(s) for s in steps))

    with Session() as db:
        run = AgentRun(
            id="run_diag",
            env_name="diag_env",
            agent_id="random",
            objective="do something",
            num_episodes=1,
            created_at=datetime.datetime.now(datetime.timezone.utc),
        )
        db.add(run)
        ep = AgentEpisode(
            id="cep_diag_001",
            run_id="run_diag",
            episode_index=0,
            seed=0,
            status="completed",
            total_steps=1,
            total_reward=0.0,
            final_objective_score=0.0,
            termination_reason="diverged",
            jsonl_path=str(jsonl_path),
            started_at=datetime.datetime.now(datetime.timezone.utc),
        )
        db.add(ep)
        db.commit()

    result = runner.invoke(app, ["diagnose", "diag_env", "--db", db_url])
    assert result.exit_code == 0, result.output
    assert "diag_env" in result.output
    assert "Dead-end rate" in result.output
    assert "Issues found" in result.output


def test_replay_json_output(tmp_path):
    import datetime
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from backend.app.models import Base, Episode, EpisodeStep

    db_url = f"sqlite:///{tmp_path / 'test.db'}"
    engine = create_engine(db_url, connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)

    with Session() as db:
        ep = Episode(
            id="ep_json_001",
            env_name="json_env",
            task_name="task",
            seed=1,
            agent_id="random",
            status="completed",
            total_steps=1,
            total_reward=1.0,
            passed=True,
            started_at=datetime.datetime.now(datetime.timezone.utc),
        )
        db.add(ep)
        db.add(EpisodeStep(
            episode_id="ep_json_001",
            step_index=0,
            action='{"type": "act"}',
            reward=1.0,
            verifier_results='[]',
            diff='{}',
            events='[]',
            state_hash_before="x",
            state_hash_after="y",
            terminated=True,
            truncated=False,
        ))
        db.commit()

    result = runner.invoke(app, ["replay", "ep_json_001", "--db", db_url, "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["episode_id"] == "ep_json_001"
    assert data["passed"] is True
    assert len(data["steps"]) == 1
    assert data["steps"][0]["action"] == {"type": "act"}


def test_benchmark_run_help():
    result = runner.invoke(app, ["benchmark", "run", "--help"])
    assert result.exit_code == 0
    assert "--domains" in result.output


def test_benchmark_report_help():
    result = runner.invoke(app, ["benchmark", "report", "--help"])
    assert result.exit_code == 0
    assert "--output" in result.output


def test_benchmark_eval_help():
    result = runner.invoke(app, ["benchmark", "eval", "--help"])
    assert result.exit_code == 0
    assert "--suite" in result.output
