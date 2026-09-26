"""Multi-step agent loop with tools and pluggable LLM."""

from __future__ import annotations

import time
import uuid
from typing import Any

from agent_runtime.llm import LLM
from agent_runtime.memory import TranscriptMemory
from agent_runtime.tools import ToolRegistry, built_in_registry
from agent_runtime.types import Message, RunResult, Step, ToolCall


class Agent:
    """Run a tool-using conversation until the model stops calling tools.

    This is an honest MVP loop — not a graph framework. One registry, one
    memory buffer, one LLM, max_steps guardrail.
    """

    def __init__(
        self,
        llm: LLM,
        tools: ToolRegistry | None = None,
        *,
        system_prompt: str | None = None,
        max_steps: int = 8,
    ) -> None:
        self.llm = llm
        self.tools = tools if tools is not None else built_in_registry()
        self.max_steps = max_steps
        self.memory = TranscriptMemory(
            system_prompt
            or (
                "You are a helpful agent. Use tools when they improve accuracy. "
                "Prefer calculator for arithmetic."
            )
        )

    def run(self, user_message: str, *, max_steps: int | None = None) -> RunResult:
        """Execute the agent loop for a single user turn."""
        limit = max_steps if max_steps is not None else self.max_steps
        self.memory.add(Message(role="user", content=user_message))

        steps: list[Step] = []
        all_tool_calls: list[ToolCall] = []
        total_latency = 0.0
        final_text = ""
        stopped = "completed"

        for i in range(limit):
            t0 = time.perf_counter()
            assistant = self.llm.complete(
                self.memory.get(),
                tools=self.tools.list_specs(),
            )
            latency = (time.perf_counter() - t0) * 1000.0
            total_latency += latency

            # Normalize tool call ids
            if assistant.tool_calls:
                for tc in assistant.tool_calls:
                    if not tc.id:
                        tc.id = f"call_{uuid.uuid4().hex[:8]}"

            self.memory.add(assistant)
            tool_results: list[Message] = []

            if assistant.tool_calls:
                all_tool_calls.extend(assistant.tool_calls)
                for tc in assistant.tool_calls:
                    result_text = self._invoke_tool(tc)
                    tool_msg = Message(
                        role="tool",
                        content=result_text,
                        name=tc.name,
                        tool_call_id=tc.id,
                    )
                    tool_results.append(tool_msg)
                    self.memory.add(tool_msg)

                steps.append(
                    Step(
                        index=i,
                        assistant_message=assistant,
                        tool_results=tool_results,
                        latency_ms=latency,
                    )
                )
                continue

            # No tool calls → final answer
            final_text = assistant.content or ""
            steps.append(
                Step(
                    index=i,
                    assistant_message=assistant,
                    tool_results=[],
                    latency_ms=latency,
                )
            )
            break
        else:
            stopped = "max_steps"
            # Prefer last assistant content if we never got a clean final turn
            if steps and not final_text:
                final_text = steps[-1].assistant_message.content or ""

        return RunResult(
            final_text=final_text,
            steps=steps,
            tool_calls=all_tool_calls,
            total_latency_ms=total_latency,
            stopped_reason=stopped,
        )

    def _invoke_tool(self, tc: ToolCall) -> str:
        if tc.name not in self.tools:
            return f'{{"error": "Unknown tool: {tc.name}"}}'
        args: dict[str, Any] = tc.arguments or {}
        return self.tools.call(tc.name, args)

    def reset(self) -> None:
        self.memory.clear(keep_system=True)
