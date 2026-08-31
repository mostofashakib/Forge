from __future__ import annotations

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.simulation_driver import legacy_events, reset_determinism_check, run_incident_reference_simulation


class SimulationDriverTests(unittest.TestCase):
    def test_driver_exports_harbor_trajectory_and_reset_check(self) -> None:
        trajectory = run_incident_reference_simulation(seed=1, output_path="/tmp/fleet_test/trajectory.json")
        event_types = [event["event_type"] for event in legacy_events(trajectory)]

        self.assertEqual("ATIF-v1.7", trajectory["schema_version"])
        self.assertEqual("ollama", trajectory["agent"]["name"])
        self.assertEqual("gemma4:26b", trajectory["agent"]["model_name"])
        self.assertEqual("user", trajectory["steps"][0]["source"])
        self.assertTrue(any(step.get("tool_calls") for step in trajectory["steps"]))
        self.assertIn("setup", event_types)
        self.assertIn("agent_thought", event_types)
        self.assertIn("tool_call", event_types)
        self.assertIn("tool_result", event_types)
        self.assertIn("agent_answer", event_types)
        self.assertIn("eval", event_types)
        self.assertTrue(reset_determinism_check(trajectory)["passed"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
