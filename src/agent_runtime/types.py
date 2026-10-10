"""Core types for the agent runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Usage:
    """Token usage reported for one model call.

    Field meanings follow the OpenTelemetry GenAI semantic conventions:
    cached and reasoning tokens are subsets of input and output tokens,
    not extra tokens on top.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    reasoning_output_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_input_tokens=self.cache_read_input_tokens
            + other.cache_read_input_tokens,
            reasoning_output_tokens=self.reasoning_output_tokens
            + other.reasoning_output_tokens,
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_input_tokens": self.cache_read_input_tokens,
            "reasoning_output_tokens": self.reasoning_output_tokens,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> Usage | None:
        if data is None:
            return None
        return cls(
            input_tokens=int(data.get("input_tokens", 0)),
            output_tokens=int(data.get("output_tokens", 0)),
            cache_read_input_tokens=int(data.get("cache_read_input_tokens", 0)),
            reasoning_output_tokens=int(data.get("reasoning_output_tokens", 0)),
        )


@dataclass
class Message:
    """A single chat message in the agent transcript."""

    role: str  # "system" | "user" | "assistant" | "tool"
    content: str
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[ToolCall] | None = None
    # Set on assistant messages when the LLM reports them.
    usage: Usage | None = None
    model: str | None = None  # model that actually served the response


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
    usage: Usage = field(default_factory=Usage)  # summed over all model calls

    @property
    def step_count(self) -> int:
        return len(self.steps)
