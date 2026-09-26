"""agent-runtime-kit: multi-step agent loop with tools, pluggable LLM, and memory."""

from agent_runtime.agent import Agent
from agent_runtime.llm import FakeLLM, FakeTurn, LLM, OpenAIAdapter
from agent_runtime.memory import TranscriptMemory
from agent_runtime.tools import ToolRegistry, built_in_registry, calculator, http_get, tool
from agent_runtime.types import Message, RunResult, Step, ToolCall

__version__ = "0.1.0"

__all__ = [
    "Agent",
    "FakeLLM",
    "FakeTurn",
    "LLM",
    "Message",
    "OpenAIAdapter",
    "RunResult",
    "Step",
    "ToolCall",
    "ToolRegistry",
    "TranscriptMemory",
    "built_in_registry",
    "calculator",
    "http_get",
    "tool",
    "__version__",
]
