from __future__ import annotations

import random
from dataclasses import dataclass

from tcrb.models import TaskSpec, Workload


@dataclass
class CustomMinimalPlanner:
    planner_id: str = "custom_minimal"

    def choose_tool(
        self,
        *,
        task: TaskSpec,
        workload: Workload,
        policy: str,
        attempt_number: int,
        attempted_tools: set[str],
        last_status: str | None,
        rng: random.Random,
    ) -> str:
        del workload, policy, attempt_number, attempted_tools, last_status, rng
        return task.primary_tool


if __name__ == "__main__":
    print(
        "Minimal planner example. Register this class in src/tcrb/planner.py and add a planner config to benchmark it through the CLI."
    )