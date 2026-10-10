"""agent-runtime-kit: multi-step agent loop with tools, pluggable LLM, and memory."""

from agent_runtime.agent import Agent
from agent_runtime.llm import LLM, FakeLLM, FakeTurn, OpenAIAdapter
from agent_runtime.memory import TranscriptMemory
from agent_runtime.tools import (
    ToolRegistry,
    built_in_registry,
    calculator,
    http_get,
    tool,
)
from agent_runtime.trajectory import (
    ReplayLLM,
    ReplayToolRegistry,
    Trajectory,
    TrajectoryMismatch,
    diff_trajectories,
)
from agent_runtime.types import Message, RunResult, Step, ToolCall, Usage

__version__ = "0.2.0"

__all__ = [
    "LLM",
    "Agent",
    "FakeLLM",
    "FakeTurn",
    "Message",
    "OpenAIAdapter",
    "ReplayLLM",
    "ReplayToolRegistry",
    "RunResult",
    "Step",
    "ToolCall",
    "ToolRegistry",
    "Trajectory",
    "TrajectoryMismatch",
    "TranscriptMemory",
    "Usage",
    "__version__",
    "built_in_registry",
    "calculator",
    "diff_trajectories",
    "http_get",
    "tool",
]
