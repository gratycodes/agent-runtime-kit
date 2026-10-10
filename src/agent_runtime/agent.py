"""Multi-step agent loop with tools and pluggable LLM."""

from __future__ import annotations

import time
import uuid
from collections.abc import Mapping
from typing import Any

from agent_runtime.llm import LLM
from agent_runtime.memory import TranscriptMemory
from agent_runtime.tools import ToolRegistry, built_in_registry
from agent_runtime.tracing import AttributeValue, NoopTracing, Tracing, tool_error_type
from agent_runtime.types import Message, RunResult, Step, ToolCall, Usage


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
        name: str = "agent",
        tracing: Tracing | None = None,
    ) -> None:
        self.llm = llm
        self.name = name
        self.tracing: Tracing = tracing if tracing is not None else NoopTracing()
        self.tools = tools if tools is not None else built_in_registry()
        self.max_steps = max_steps
        self.memory = TranscriptMemory(
            system_prompt
            or (
                "You are a helpful agent. Use tools when they improve accuracy. "
                "Prefer calculator for arithmetic."
            )
        )

    def run(
        self,
        user_message: str,
        *,
        max_steps: int | None = None,
        trace_attributes: Mapping[str, AttributeValue] | None = None,
    ) -> RunResult:
        """Execute the agent loop for a single user turn.

        ``trace_attributes`` are added to every span of this run when tracing
        is enabled (for example ``{"feature": "checkout", "tenant.id": "t-1"}``).
        """
        attrs = dict(trace_attributes or {})
        provider = getattr(self.llm, "provider_name", None)
        request_model = getattr(self.llm, "model", None)
        with self.tracing.invoke_agent(
            agent_name=self.name,
            provider=provider,
            request_model=request_model,
            attributes=attrs,
        ) as agent_span:
            result = self._run(user_message, max_steps, attrs, provider, request_model)
            agent_span.set_usage(result.usage)
            if result.stopped_reason == "max_steps":
                agent_span.set_error("max_steps", "agent stopped at max_steps")
            return result

    def _run(
        self,
        user_message: str,
        max_steps: int | None,
        attrs: dict[str, AttributeValue],
        provider: str | None,
        request_model: str | None,
    ) -> RunResult:
        limit = max_steps if max_steps is not None else self.max_steps
        self.memory.add(Message(role="user", content=user_message))

        steps: list[Step] = []
        all_tool_calls: list[ToolCall] = []
        total_latency = 0.0
        usage = Usage()
        final_text = ""
        stopped = "completed"

        for i in range(limit):
            with self.tracing.chat(
                provider=provider, request_model=request_model, attributes=attrs
            ) as chat_span:
                t0 = time.perf_counter()
                assistant = self.llm.complete(
                    self.memory.get(),
                    tools=self.tools.list_specs(),
                )
                latency = (time.perf_counter() - t0) * 1000.0
                chat_span.set_usage(assistant.usage)
                if assistant.model:
                    chat_span.set_attribute("gen_ai.response.model", assistant.model)
                chat_span.set_attribute(
                    "gen_ai.response.finish_reasons",
                    ["tool_calls"] if assistant.tool_calls else ["stop"],
                )
            total_latency += latency
            if assistant.usage is not None:
                usage = usage + assistant.usage

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
                    result_text = self._traced_tool_call(tc, attrs)
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
            usage=usage,
        )

    def _traced_tool_call(self, tc: ToolCall, attrs: dict[str, AttributeValue]) -> str:
        with self.tracing.execute_tool(
            tool_name=tc.name,
            call_id=tc.id,
            agent_name=self.name,
            arguments=tc.arguments or {},
            attributes=attrs,
        ) as tool_span:
            result = self._invoke_tool(tc)
            self.tracing.record_tool_result(tool_span, result)
            if tc.name not in self.tools:
                tool_span.set_error("unknown_tool", f"unknown tool {tc.name}")
            elif (err := tool_error_type(result)) is not None:
                tool_span.set_error(err, "tool returned an error")
            return result

    def _invoke_tool(self, tc: ToolCall) -> str:
        if tc.name not in self.tools:
            return f'{{"error": "Unknown tool: {tc.name}"}}'
        args: dict[str, Any] = tc.arguments or {}
        return self.tools.call(tc.name, args)

    def reset(self) -> None:
        self.memory.clear(keep_system=True)
