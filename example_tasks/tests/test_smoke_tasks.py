from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tests.slack.test_smoke import SlackSmokeTaskTests, SlackToolSmokeTests
from tests.task_manager.test_smoke import TaskManagerSmokeTaskTests, TaskManagerToolSmokeTests

__all__ = [
    "SlackSmokeTaskTests",
    "SlackToolSmokeTests",
    "TaskManagerSmokeTaskTests",
    "TaskManagerToolSmokeTests",
]


if __name__ == "__main__":
    unittest.main(verbosity=2)
