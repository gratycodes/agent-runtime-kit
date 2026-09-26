"""Tests for the agent loop and tool selection."""

from __future__ import annotations

import json

from agent_runtime.agent import Agent
from agent_runtime.llm import FakeLLM, FakeTurn
from agent_runtime.tools import built_in_registry


def test_agent_final_answer_no_tools() -> None:
    llm = FakeLLM(script=[FakeTurn(content="Hello!")])
    agent = Agent(llm=llm, tools=built_in_registry(), max_steps=3)
    result = agent.run("Say hello")
    assert result.final_text == "Hello!"
    assert result.step_count == 1
    assert result.tool_calls == []
    assert result.stopped_reason == "completed"


def test_agent_uses_calculator_then_answers() -> None:
    llm = FakeLLM(
        script=[
            FakeTurn(
                content="",
                tool_calls=[
                    {
                        "id": "c1",
                        "name": "calculator",
                        "arguments": {"expression": "6 * 7"},
                    }
                ],
            ),
            FakeTurn(content="The answer is 42."),
        ]
    )
    agent = Agent(llm=llm, tools=built_in_registry(), max_steps=5)
    result = agent.run("What is 6*7?")

    assert result.final_text == "The answer is 42."
    assert result.step_count == 2
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "calculator"
    # Tool result should have been fed back into memory / steps
    tool_payload = json.loads(result.steps[0].tool_results[0].content)
    assert tool_payload["result"] == 42.0
    assert result.total_latency_ms >= 0.0


def test_agent_tool_selection_prefers_named_tool() -> None:
    """FakeLLM scripted to pick calculator — verifies registry dispatch."""
    llm = FakeLLM(
        script=[
            FakeTurn(
                tool_calls=[
                    {
                        "name": "calculator",
                        "arguments": {"expression": "100 - 1"},
                    }
                ]
            ),
            FakeTurn(content="99"),
        ]
    )
    agent = Agent(llm=llm, tools=built_in_registry())
    result = agent.run("100 minus 1")
    assert result.tool_calls[0].name == "calculator"
    assert "99" in result.final_text


def test_agent_unknown_tool_surfaces_error() -> None:
    llm = FakeLLM(
        script=[
            FakeTurn(
                tool_calls=[
                    {"id": "x", "name": "not_a_real_tool", "arguments": {}}
                ]
            ),
            FakeTurn(content="I could not use that tool."),
        ]
    )
    agent = Agent(llm=llm, tools=built_in_registry())
    result = agent.run("Use a missing tool")
    err = result.steps[0].tool_results[0].content
    assert "Unknown tool" in err
    assert result.final_text.startswith("I could not")


def test_agent_max_steps() -> None:
    # Always request another tool call — should stop at max_steps
    turns = [
        FakeTurn(
            tool_calls=[
                {
                    "name": "calculator",
                    "arguments": {"expression": "1+1"},
                }
            ]
        )
        for _ in range(10)
    ]
    llm = FakeLLM(script=turns, fallback="should-not-reach")
    agent = Agent(llm=llm, tools=built_in_registry(), max_steps=3)
    result = agent.run("loop forever")
    assert result.stopped_reason == "max_steps"
    assert result.step_count == 3


def test_memory_accumulates_across_steps() -> None:
    llm = FakeLLM(
        script=[
            FakeTurn(
                tool_calls=[
                    {
                        "name": "calculator",
                        "arguments": {"expression": "2+2"},
                    }
                ]
            ),
            FakeTurn(content="4"),
        ]
    )
    agent = Agent(llm=llm, tools=built_in_registry())
    agent.run("2+2?")
    # system + user + assistant(tool) + tool + assistant(final)
    assert len(agent.memory) >= 5
    roles = [m.role for m in agent.memory.get()]
    assert roles[0] == "system"
    assert "tool" in roles
