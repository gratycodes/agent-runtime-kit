"""GenAI semantic-convention spans emitted by the agent loop."""

from __future__ import annotations

import pytest

from agent_runtime import Agent, FakeLLM, FakeTurn, Usage, built_in_registry

otel = pytest.importorskip("opentelemetry.sdk.trace", reason="needs the otel extra")

from opentelemetry.sdk.trace import TracerProvider  # noqa: E402
from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (  # noqa: E402
    InMemorySpanExporter,
)
from opentelemetry.trace import SpanKind, StatusCode  # noqa: E402

from agent_runtime.tracing import GenAITracing  # noqa: E402


@pytest.fixture
def exporter() -> InMemorySpanExporter:
    return InMemorySpanExporter()


def _tracing(exporter: InMemorySpanExporter, **kwargs) -> GenAITracing:
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return GenAITracing(provider, **kwargs)


def _script() -> list[FakeTurn]:
    return [
        FakeTurn(
            tool_calls=[{"id": "call_1", "name": "calculator", "arguments": {"expression": "6*7"}}],
            usage=Usage(input_tokens=900, output_tokens=40, cache_read_input_tokens=512),
        ),
        FakeTurn(
            content="42",
            usage=Usage(input_tokens=1000, output_tokens=20, reasoning_output_tokens=8),
            model="example-small-2026-09-01",
        ),
    ]


def test_spans_follow_genai_conventions(exporter: InMemorySpanExporter) -> None:
    agent = Agent(
        llm=FakeLLM(script=_script(), model="example-small"),
        tools=built_in_registry(),
        name="math-helper",
        tracing=_tracing(exporter, attributes={"feature": "calc"}),
    )
    result = agent.run("6*7?", trace_attributes={"tenant.id": "tenant-a"})
    assert result.final_text == "42"

    spans = exporter.get_finished_spans()
    names = [s.name for s in spans]
    assert names == [
        "chat example-small",
        "execute_tool calculator",
        "chat example-small",
        "invoke_agent math-helper",
    ]
    chat1, tool, chat2, root = spans

    # Parent/child structure: everything under invoke_agent, same trace.
    assert root.parent is None
    for child in (chat1, tool, chat2):
        assert child.parent.span_id == root.context.span_id
        assert child.context.trace_id == root.context.trace_id

    assert root.kind == SpanKind.INTERNAL
    assert root.attributes["gen_ai.operation.name"] == "invoke_agent"
    assert root.attributes["gen_ai.agent.name"] == "math-helper"
    assert root.attributes["gen_ai.provider.name"] == "fake"
    assert root.attributes["gen_ai.usage.input_tokens"] == 1900
    assert root.attributes["gen_ai.usage.output_tokens"] == 60

    assert chat1.kind == SpanKind.CLIENT
    assert chat1.attributes["gen_ai.operation.name"] == "chat"
    assert chat1.attributes["gen_ai.request.model"] == "example-small"
    assert chat1.attributes["gen_ai.response.model"] == "example-small"
    assert chat1.attributes["gen_ai.usage.input_tokens"] == 900
    assert chat1.attributes["gen_ai.usage.cache_read.input_tokens"] == 512
    assert chat1.attributes["gen_ai.response.finish_reasons"] == ("tool_calls",)
    assert chat2.attributes["gen_ai.response.model"] == "example-small-2026-09-01"
    assert chat2.attributes["gen_ai.usage.reasoning.output_tokens"] == 8
    assert chat2.attributes["gen_ai.response.finish_reasons"] == ("stop",)
    assert "gen_ai.usage.reasoning.output_tokens" not in chat1.attributes

    assert tool.attributes["gen_ai.operation.name"] == "execute_tool"
    assert tool.attributes["gen_ai.tool.name"] == "calculator"
    assert tool.attributes["gen_ai.tool.call.id"] == "call_1"
    assert tool.attributes["gen_ai.tool.type"] == "function"
    assert "gen_ai.tool.call.arguments" not in tool.attributes, "content capture is opt-in"

    for span in spans:
        assert span.attributes["feature"] == "calc"
        assert span.attributes["tenant.id"] == "tenant-a"


def test_tool_content_capture_is_opt_in(exporter: InMemorySpanExporter) -> None:
    agent = Agent(
        llm=FakeLLM(script=_script()),
        tracing=_tracing(exporter, capture_tool_content=True),
    )
    agent.run("6*7?")
    tool = next(s for s in exporter.get_finished_spans() if s.name.startswith("execute_tool"))
    assert tool.attributes["gen_ai.tool.call.arguments"] == '{"expression": "6*7"}'
    assert '"result": 42.0' in tool.attributes["gen_ai.tool.call.result"]


def test_tool_failures_and_unknown_tools_are_marked(exporter: InMemorySpanExporter) -> None:
    script = [
        FakeTurn(
            tool_calls=[
                {"id": "a", "name": "calculator", "arguments": {"expression": "1/0"}},
                {"id": "b", "name": "no_such_tool", "arguments": {}},
            ]
        ),
        FakeTurn(content="sorry"),
    ]
    Agent(llm=FakeLLM(script=script), tracing=_tracing(exporter)).run("x")
    tools = {s.attributes["gen_ai.tool.call.id"]: s for s in exporter.get_finished_spans()
             if s.attributes["gen_ai.operation.name"] == "execute_tool"}
    assert tools["a"].attributes["error.type"] == "tool_error"
    assert tools["a"].status.status_code == StatusCode.ERROR
    assert tools["b"].attributes["error.type"] == "unknown_tool"


def test_llm_exception_is_recorded_and_reraised(exporter: InMemorySpanExporter) -> None:
    class Boom:
        model = "example-small"
        provider_name = "example"

        def complete(self, messages, tools=None):
            raise TimeoutError("upstream timed out")

    with pytest.raises(TimeoutError):
        Agent(llm=Boom(), tracing=_tracing(exporter)).run("x")
    chat, root = exporter.get_finished_spans()
    for span in (chat, root):
        assert span.status.status_code == StatusCode.ERROR
        assert span.attributes["error.type"] == "TimeoutError"
    assert any(e.name == "exception" for e in chat.events)


def test_max_steps_marks_agent_span(exporter: InMemorySpanExporter) -> None:
    looping = [FakeTurn(tool_calls=[{"name": "calculator", "arguments": {"expression": "1+1"}}])] * 5
    result = Agent(llm=FakeLLM(script=looping), tracing=_tracing(exporter), max_steps=2).run("x")
    assert result.stopped_reason == "max_steps"
    root = exporter.get_finished_spans()[-1]
    assert root.attributes["error.type"] == "max_steps"
