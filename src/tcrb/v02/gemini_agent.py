from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from typing import Any

from google import genai
from google.genai import types

from .agent import Agent, LogProbAgent, parse_action
from .types import (
    Abort,
    AgentAction,
    Clarify,
    FinalAnswer,
    Observation,
    ToolCall,
    ToolDef,
)


REVIEW_SYSTEM_PROMPT = """You are a tool-calling recovery expert. An agent just made a tool call and received an observation. Your job is to decide what the agent should do next.

Available actions:
1. RETRY: Try the same tool again (for transient errors)
2. FALLBACK: Use a different tool (if current tool failed)
3. CLARIFY: Ask user for more information
4. ABORT: Give up if task is impossible
5. ANSWER: Provide final answer based on observation

Rules:
- For timeout/rate_limit/execution_error: RETRY once, then FALLBACK or ABORT
- For schema_drift: Try to extract useful info, then ANSWER or FALLBACK
- For partial_output: Use what's available, then ANSWER
- For silent_corruption: If suspicious, CLARIFY or ABORT
- For cross_source_conflict: CLARIFY with user
- For success: ANSWER with the data

Output EXACTLY one JSON object:
{"action": "RETRY"|"FALLBACK"|"CLARIFY"|"ABORT"|"ANSWER", "tool": "tool_name", "text": "message"}

Examples:
- {"action": "RETRY", "tool": "customer_lookup", "text": ""}
- {"action": "FALLBACK", "tool": "order_lookup", "text": ""}
- {"action": "CLARIFY", "tool": "", "text": "Which customer ID?"}
- {"action": "ABORT", "tool": "", "text": "Task impossible: missing data"}
- {"action": "ANSWER", "tool": "", "text": "Customer Alice Chen is premium tier"}
"""


@dataclass
class GeminiReviewerAgent:
    agent_id: str = "gemini_reviewer"
    base_agent: LogProbAgent | None = None
    model: Any = None
    tokenizer: Any = None
    system_prompt: str = ""
    gemini_client: Any = field(default=None, init=False)
    gemini_model: str = field(default="gemini-3.0-flash-lite", init=False)
    review_threshold: float = 0.5
    max_reviews: int = 3
    _review_count: int = field(default=0, init=False)

    def __post_init__(self):
        if self.base_agent is None and self.model is not None and self.tokenizer is not None:
            self.base_agent = LogProbAgent(
                agent_id="base",
                model=self.model,
                tokenizer=self.tokenizer,
                system_prompt=self.system_prompt,
            )
        
        api_key = os.environ.get("GEMINI_API_KEY")
        if api_key:
            self.gemini_client = genai.Client(api_key=api_key)

    def _needs_review(self, observation: Observation | None) -> bool:
        if observation is None:
            return False
        
        error_statuses = {"timeout", "rate_limit", "execution_error", "invalid_arguments", "unknown_tool"}
        if observation.status in error_statuses:
            return True
        
        warning_statuses = {"schema_drift", "partial_output", "silent_corruption", "cross_source_conflict"}
        if observation.status in warning_statuses:
            return True
        
        if self._review_count >= self.max_reviews:
            return False
        
        return False

    def _review_with_gemini(
        self,
        task_query: str,
        action: AgentAction,
        observation: Observation,
        available_tools: list[ToolDef],
        history: list[tuple[AgentAction | None, Observation | None]],
    ) -> AgentAction | None:
        if not self.gemini_client:
            return None
        
        try:
            tools_text = "\n".join(f"- {t.name}: {t.description}" for t in available_tools)
            
            if isinstance(action, ToolCall):
                action_text = f"Tool call: {action.name}({json.dumps(action.arguments)})"
            else:
                action_text = f"Action: {type(action).__name__}"
            
            obs_text = f"Observation: status={observation.status}, payload={json.dumps(observation.payload)}"
            
            history_text = ""
            for a, o in history[-3:]:
                if isinstance(a, ToolCall):
                    history_text += f"  - Called {a.name}, got {o.status if o else 'None'}\n"
            
            prompt = f"""{REVIEW_SYSTEM_PROMPT}

Task: {task_query}

Available tools:
{tools_text}

Recent history:
{history_text}

Last action: {action_text}
Last observation: {obs_text}

What should the agent do next? Output JSON:"""
            
            response = self.gemini_client.models.generate_content(
                model=self.gemini_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=256,
                ),
            )
            result_text = response.text.strip()
            
            if result_text.startswith("```"):
                result_text = result_text.split("```")[1]
                if result_text.startswith("json"):
                    result_text = result_text[4:]
                result_text = result_text.strip()
            
            result = json.loads(result_text)
            action_type = result.get("action", "ANSWER")
            
            if action_type == "RETRY":
                tool_name = result.get("tool", "")
                if isinstance(action, ToolCall) and tool_name:
                    return ToolCall(name=tool_name, arguments=action.arguments)
                return action
            
            elif action_type == "FALLBACK":
                tool_name = result.get("tool", "")
                if tool_name and any(t.name == tool_name for t in available_tools):
                    args = self._extract_fallback_args(tool_name, task_query, history)
                    return ToolCall(name=tool_name, arguments=args)
                return action
            
            elif action_type == "CLARIFY":
                text = result.get("text", "Need more information")
                return Clarify(text=text)
            
            elif action_type == "ABORT":
                reason = result.get("text", "Task impossible")
                return Abort(reason=reason)
            
            elif action_type == "ANSWER":
                text = result.get("text", "")
                if not text and isinstance(action, ToolCall) and observation.payload:
                    text = json.dumps(observation.payload)
                return FinalAnswer(text=text if text else "Task completed")
            
            return None
        
        except Exception as e:
            print(f"[GeminiReviewer] Error: {e}")
            return None

    def _extract_fallback_args(self, tool_name: str, task_query: str, history: list) -> dict[str, Any]:
        import re
        args: dict[str, Any] = {}
        
        m = re.search(r'\b([A-Z][0-9]{2,4})\b', task_query)
        if m:
            args["customer_id"] = m.group(1)
            args["order_id"] = m.group(1)
            args["ticket_id"] = m.group(1)
        
        m = re.search(r'\b([A-Z]+-\d+)\b', task_query)
        if m:
            for key in ["customer_id", "order_id", "ticket_id", "code"]:
                args[key] = m.group(1)
        
        return args

    def next_action(
        self,
        *,
        task_query: str,
        available_tools: list[ToolDef],
        history: list[tuple[AgentAction | None, Observation | None]],
        rng: random.Random,
    ) -> AgentAction:
        if not self.base_agent:
            return FinalAnswer(text="No base agent configured")
        
        action = self.base_agent.next_action(
            task_query=task_query,
            available_tools=available_tools,
            history=history,
            rng=rng,
        )
        
        if history:
            last_action, last_obs = history[-1]
            if last_action and last_obs and self._needs_review(last_obs):
                self._review_count += 1
                reviewed_action = self._review_with_gemini(
                    task_query=task_query,
                    action=last_action,
                    observation=last_obs,
                    available_tools=available_tools,
                    history=history[:-1],
                )
                if reviewed_action is not None:
                    return reviewed_action
        
        return action
