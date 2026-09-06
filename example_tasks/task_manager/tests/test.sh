#!/usr/bin/env bash
# Harbor uploads this directory to /tests and runs this script as the verifier
# user (root). The workspace lives in the `tasks` service; nothing here opens a
# database directly.
#
# Deliberately not `set -e`: a failing environment self-test must still produce a
# reward file. A missing reward file reaches Harbor as RewardFileNotFoundError,
# which says nothing about what went wrong; `valid=0` in the reward stream says
# the environment was broken and the episode should not be read as evidence
# about the agent.
set -uo pipefail

# /opt/grading holds the full world and the verifier package, readable only by
# root; /opt holds the agent's client-only package. Order matters: the verifier
# needs the full one.
export PYTHONPATH=/opt/grading:/opt:/tests
mkdir -p /logs/verifier

selftest_status=0
for suite in test_virtual_clock test_scenario_engine test_task_surface test_mcp_server test_tool_contract test_rl_contract test_verifiers test_harbor_reward test_reward_validation test_reward_matrix test_integrity_violations test_agent_adapters test_environment_contract; do
  echo "--- ${suite} ---"
  if ! python3 "/tests/${suite}.py"; then
    echo "SELF-TEST FAILED: ${suite}" >&2
    selftest_status=1
  fi
done

# The graded outcome always runs, and always writes a reward.
export TASK_SELFTEST_STATUS="${selftest_status}"
python3 /tests/test_reassignment_readiness.py
verifier_status=$?

if [ "${selftest_status}" -ne 0 ]; then
  echo "Environment self-tests failed; reward reports valid=0." >&2
fi
exit "${verifier_status}"
