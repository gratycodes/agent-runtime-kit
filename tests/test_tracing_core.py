"""Tracing behaviour that must hold without OpenTelemetry installed."""

from __future__ import annotations

import subprocess
import sys

from agent_runtime import Agent, FakeLLM
from agent_runtime.tracing import NoopTracing, tool_error_type


def test_core_import_does_not_load_opentelemetry() -> None:
    code = (
        "import sys, agent_runtime; "
        "from agent_runtime import Agent, FakeLLM; "
        "Agent(llm=FakeLLM()).run('hi'); "
        "print(any(m.startswith('opentelemetry') for m in sys.modules))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def test_noop_tracing_is_the_default() -> None:
    agent = Agent(llm=FakeLLM())
    assert isinstance(agent.tracing, NoopTracing)


def test_tool_error_type() -> None:
    assert tool_error_type('{"error": "boom"}') == "tool_error"
    assert tool_error_type('{"result": 4}') is None
    assert tool_error_type("plain text") is None
