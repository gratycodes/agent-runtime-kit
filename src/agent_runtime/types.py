"""Core types for the agent runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Message:
    """A single chat message in the agent transcript."""

    role: str  # "system" | "user" | "assistant" | "tool"
    content: str
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[ToolCall] | None = None


@dataclass
class ToolCall:
    """A request from the LLM to invoke a registered tool."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class Step:
    """One iteration of the agent loop."""

    index: int
    assistant_message: Message
    tool_results: list[Message] = field(default_factory=list)
    latency_ms: float = 0.0


@dataclass
class RunResult:
    """Outcome of an agent.run(...) call."""

    final_text: str
    steps: list[Step] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    total_latency_ms: float = 0.0
    stopped_reason: str = "completed"  # completed | max_steps | error

    @property
    def step_count(self) -> int:
        return len(self.steps)
