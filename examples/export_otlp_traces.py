"""Run two scripted support-agent conversations and write their traces as OTLP/JSON.

The output file is a fixture for cost tooling such as otel-genai-cost
(https://github.com/gratycodes/otel-genai-cost). Everything is synthetic: the
model is FakeLLM with scripted token usage, model names are the made-up
``example-*`` names used in that project's example price book, and the tools
return canned data.

    pip install -e ".[otel]"
    python examples/export_otlp_traces.py examples/fixtures/agent_runs.otlp.json
"""

from __future__ import annotations

import sys
from pathlib import Path

from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from agent_runtime import Agent, FakeLLM, FakeTurn, ToolRegistry, Usage
from agent_runtime.otlp_json import spans_to_otlp_json
from agent_runtime.tracing import GenAITracing

ORDERS = {
    "A-1001": {"status": "delivered", "days_since_delivery": 3, "amount": 42.50},
    "B-2002": {"status": "in_transit", "eta_days": 2},
}


def support_tools() -> ToolRegistry:
    reg = ToolRegistry()

    @reg.register(name="lookup_order", description="Look up a (synthetic) order by id")
    def lookup_order(order_id: str) -> dict:
        return ORDERS.get(order_id, {"error": f"order {order_id} not found"})

    @reg.register(name="refund_policy", description="Return the (synthetic) refund window")
    def refund_policy() -> dict:
        return {"refund_window_days": 30}

    return reg


def refund_run() -> tuple[Agent, str, dict]:
    script = [
        FakeTurn(
            tool_calls=[
                {"id": "call_1", "name": "lookup_order", "arguments": {"order_id": "A-1001"}},
                {"id": "call_2", "name": "refund_policy", "arguments": {}},
            ],
            usage=Usage(input_tokens=2400, output_tokens=60, cache_read_input_tokens=1800),
        ),
        FakeTurn(
            content="Order A-1001 was delivered 3 days ago, inside the 30-day window, so it qualifies for a refund of $42.50.",
            usage=Usage(input_tokens=2650, output_tokens=48, cache_read_input_tokens=2300),
            model="example-large-2026-09-01",
        ),
    ]
    llm = FakeLLM(script=script, model="example-large")
    return (
        Agent(llm=llm, tools=support_tools(), name="support-bot"),
        "Can I get a refund for order A-1001?",
        {"feature": "refund-assistant", "tenant.id": "tenant-a"},
    )


def status_run() -> tuple[Agent, str, dict]:
    script = [
        FakeTurn(
            tool_calls=[{"id": "call_1", "name": "lookup_order", "arguments": {"order_id": "B-2002"}}],
            usage=Usage(input_tokens=900, output_tokens=25),
        ),
        FakeTurn(content="Order B-2002 is in transit and should arrive in about 2 days.",
                 usage=Usage(input_tokens=1010, output_tokens=19)),
    ]
    llm = FakeLLM(script=script, model="example-small")
    return (
        Agent(llm=llm, tools=support_tools(), name="support-bot"),
        "Where is order B-2002?",
        {"feature": "order-status", "tenant.id": "tenant-b"},
    )


def export(path: Path) -> int:
    exporter = InMemorySpanExporter()
    provider = TracerProvider(resource=Resource.create({"service.name": "support-bot-demo"}))
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracing = GenAITracing(provider)

    for build in (refund_run, status_run):
        agent, question, attrs = build()
        agent.tracing = tracing
        agent.run(question, trace_attributes=attrs)

    spans = exporter.get_finished_spans()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(spans_to_otlp_json(spans) + "\n", encoding="utf-8")
    return len(spans)


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "agent_runs.otlp.json")
    n = export(out)
    print(f"wrote {n} spans to {out}")


if __name__ == "__main__":
    main()
