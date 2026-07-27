from __future__ import annotations

import random
from dataclasses import dataclass, field

from .agent import Agent
from .types import (
    AgentAction,
    Observation,
    ToolCall,
    ToolDef,
    FinalAnswer,
)

# Forward reference to v0.1 ToolPlanner protocol
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..planner import ToolPlanner


_MARKER = "LEGACY_ROUTING"


@dataclass
class LegacyRoutingAgent:
    agent_id: str = "legacy_routing"
    planner: "ToolPlanner | None" = None
    policy: str = "naive_retry"
    attempt_number: int = field(default=1, init=False)
    attempted_tools: set[str] = field(default_factory=set, init=False)
    last_status: str | None = field(default=None, init=False)

    def next_action(
        self,
        *,
        task_query: str,
        available_tools: list[ToolDef],
        history: list[tuple[AgentAction | None, Observation | None]],
        rng: random.Random,
    ) -> AgentAction:
        if self.planner is None:
            return FinalAnswer(text="")

        if self.attempt_number > 1:
            last_obs = None
            for _, obs in reversed(history):
                if obs is not None:
                    last_obs = obs
                    break
            if last_obs is not None and last_obs.status != "success":
                self.last_status = last_obs.status

        from ..models import TaskSpec, Workload, ToolSpec

        task = TaskSpec(
            task_id=_MARKER,
            primary_tool=available_tools[0].name if available_tools else "",
            fallback_tools=[t.name for t in available_tools[1:]],
            required_schema=[],
            user_query=task_query,
        )

        tools: dict[str, ToolSpec] = {}
        for td in available_tools:
            props = td.input_schema.get("properties", {})
            tools[td.name] = ToolSpec(
                name=td.name,
                base_latency_ms=td.base_latency_ms,
                jitter_ms=td.jitter_ms,
                schema_fields=list(props.keys()),
                description=td.description,
                timeout_ms=td.timeout_ms,
                fault_multipliers=dict(td.fault_multipliers),
            )

        workload = Workload(tools=tools, tasks=[task])

        tool_name = self.planner.choose_tool(
            task=task,
            workload=workload,
            policy=self.policy,
            attempt_number=self.attempt_number,
            attempted_tools=self.attempted_tools,
            last_status=self.last_status,
            rng=rng,
        )

        if not tool_name or tool_name not in tools:
            return FinalAnswer(text="")

        self.attempted_tools.add(tool_name)
        self.attempt_number += 1

        return ToolCall(name=tool_name, arguments={})


def wrap_legacy_planner(planner: "ToolPlanner", policy: str = "naive_retry") -> LegacyRoutingAgent:
    return LegacyRoutingAgent(
        agent_id="legacy_routing",
        planner=planner,
        policy=policy,
    )
