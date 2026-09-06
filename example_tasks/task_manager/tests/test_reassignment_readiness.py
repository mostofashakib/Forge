#!/usr/bin/env python3
"""Harbor verifier entry point for the task-manager handover benchmark.

Runs as root in the agent's container after the agent phase. It never touches
workspace state directly -- there is no database in this container. It asks the
environment for a state export over ``admin.sock``, which is mode 0600 and owned
by the environment's own user, so the unprivileged agent could not have produced
or tampered with the answer this verifier reads.

The grade is the layered verifier's: ``verifiers`` supplies the framework,
``verifiers.contracts.reassignment`` states what this task requires, and
``TieredRewardEngine`` turns the two into a reward under a named preset. It is
the only grader: there is no second scorer and no diagnostic key stream beside
it, because two answers to "how did this run do" is one too many.

The grader and the answer key live in the grading package, which the agent's
image does not contain.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
# The full world ships root-only at /opt/grading so the verifier can derive
# ground truth from the seed. Added here rather than relying on the caller's
# PYTHONPATH: a verifier that dies on an import writes no reward at all.
if os.path.isdir("/opt/grading"):
    sys.path.insert(0, "/opt/grading")

from task_sim.progress import IntegrityError, check_workspace_integrity  # noqa: E402
from task_sim.protocol import ADMIN_SOCKET, request  # noqa: E402
from verifiers import Episode, TieredRewardEngine, blank_reward_dict, reward_dict  # noqa: E402
from verifiers.contracts.reassignment import build_contract  # noqa: E402
from verifiers.presets import DEFAULT_PRESET, load_experiment  # noqa: E402

REWARD_DIR = Path(os.environ.get("TASK_REWARD_DIR", "/logs/verifier"))
ADMIN = os.environ.get("TASKS_ADMIN_SOCKET", ADMIN_SOCKET)
STATE_FILE = os.environ.get("TASK_STATE_FILE", "")


def experiment_config() -> dict:
    """The shipped `verifiers/experiment.yaml`, or an empty config."""
    import verifiers as verifiers_package

    path = Path(verifiers_package.__file__).resolve().parent / "experiment.yaml"
    if not path.is_file():
        return {}
    try:
        return load_experiment(path)
    except Exception:  # noqa: BLE001 - an unreadable config falls back to defaults
        return {}


def chosen_preset() -> str:
    """Env first, then the shipped experiment file, then the default.

    An ablation is meant to be a configuration change rather than a fork, so the
    preset that graded a run is read from one place and written back into that
    run's own details.
    """
    from_env = os.environ.get("TASK_REWARD_PRESET", "").strip()
    if from_env:
        return from_env
    return str(experiment_config().get("reward_preset") or DEFAULT_PRESET)


def blank(valid: float) -> dict[str, float]:
    return blank_reward_dict(valid)


def evaluate(state: dict | None, preset: str | None = None) -> tuple[dict[str, float], dict]:
    """Grade one episode. The only function in this module that decides a score."""
    preset = preset or chosen_preset()
    if state is None:
        return blank(0.0), {"integrity_error": "workspace state unavailable", "preset": preset}

    try:
        # Seeded tasks destroyed, the event ledger rewritten: the episode did
        # not happen in the world this task specifies, so no grader's number
        # from it is evidence about the agent.
        check_workspace_integrity(state)
    except IntegrityError as error:
        # The exception is the whole record of why this run was discarded, so it
        # is written down as data and not only as a sentence: `kind` says which
        # of four unrelated things went wrong, and the rest names the sections,
        # tasks or events to go and look at.
        return blank(0.0), {"preset": preset, "integrity_error": str(error),
                            "integrity": error.as_dict()}
    except Exception as error:  # noqa: BLE001 - a broken check is not a bad episode
        # A crash in the check is a defect in the grader, and must not be filed
        # under any of the kinds above -- those send someone to inspect a
        # workspace that may be perfectly intact.
        message = f"integrity check failed: {type(error).__name__}: {error}"
        return blank(0.0), {"preset": preset, "integrity_error": message,
                            "integrity": {"kind": "check_failed", "message": message}}

    contract = build_contract()
    evaluation = TieredRewardEngine.for_preset(contract, preset).evaluate(
        Episode.from_state(state)
    )

    rewards = blank(1.0)
    rewards.update(reward_dict(evaluation))
    return rewards, {
        "preset": preset,
        "evaluation": evaluation.as_dict(),
        "summary": evaluation.summary(),
    }


def load_state() -> tuple[dict | None, str | None]:
    """Prefer a collected state snapshot; otherwise ask the environment."""
    if STATE_FILE:
        try:
            payload = json.loads(Path(STATE_FILE).read_text(encoding="utf-8"))
            return payload.get("result", payload), None
        except Exception as exc:  # noqa: BLE001
            return None, f"could not read state snapshot: {type(exc).__name__}: {exc}"
    try:
        response = request(ADMIN, {"op": "export_state"}, timeout=120.0)
    except Exception as exc:  # noqa: BLE001
        return None, f"could not reach the workspace over {ADMIN}: {type(exc).__name__}: {exc}"
    if not response.get("ok"):
        return None, f"workspace refused the state export: {response.get('error')}"
    return response["result"], None


def main() -> None:
    REWARD_DIR.mkdir(parents=True, exist_ok=True)
    state, state_error = load_state()
    rewards, details = evaluate(state)

    # A failed environment self-test invalidates the episode: the world the
    # agent acted in was not the world this task specifies, so its score is not
    # evidence about the agent.
    if os.environ.get("TASK_SELFTEST_STATUS", "0") != "0":
        # Blank every key, not just `valid`. Harbor's pass criterion is
        # rewards["reward"] == 1.0, so leaving a score standing would let a
        # broken environment be counted as a passing run.
        rewards = blank(0.0)
        details["integrity_error"] = "environment self-tests failed; see verifier stdout"

    if state_error:
        details["integrity_error"] = state_error

    (REWARD_DIR / "reward.json").write_text(
        json.dumps(rewards, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    (REWARD_DIR / "details.json").write_text(
        json.dumps(details, sort_keys=True, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(details.get("summary", ""))
    print(json.dumps({"rewards": rewards, "details": details},
                     sort_keys=True, indent=2, default=str))

    # A verifier fault must be loud. A low-scoring agent must not be.
    if rewards["valid"] == 0.0:
        sys.exit(1)


if __name__ == "__main__":
    main()
