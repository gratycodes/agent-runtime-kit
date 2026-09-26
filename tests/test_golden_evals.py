"""Golden evaluation cases — fixed FakeLLM scenarios with expected outcomes."""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from agent_runtime.agent import Agent
from agent_runtime.llm import FakeLLM, FakeTurn
from agent_runtime.tools import built_in_registry


@dataclass
class GoldenCase:
    name: str
    user: str
    script: list[FakeTurn]
    expect_final_contains: str
    expect_tool_names: list[str]
    expect_calc_result: float | None = None


GOLDEN_CASES = [
    GoldenCase(
        name="simple_arithmetic",
        user="Compute 12 * 11",
        script=[
            FakeTurn(
                tool_calls=[
                    {
                        "id": "g1",
                        "name": "calculator",
                        "arguments": {"expression": "12 * 11"},
                    }
                ]
            ),
            FakeTurn(content="12 * 11 = 132."),
        ],
        expect_final_contains="132",
        expect_tool_names=["calculator"],
        expect_calc_result=132.0,
    ),
    GoldenCase(
        name="nested_expression",
        user="What is (3 + 5) * sqrt(9)?",
        script=[
            FakeTurn(
                tool_calls=[
                    {
                        "id": "g2",
                        "name": "calculator",
                        "arguments": {"expression": "(3 + 5) * sqrt(9)"},
                    }
                ]
            ),
            FakeTurn(content="The result is 24."),
        ],
        expect_final_contains="24",
        expect_tool_names=["calculator"],
        expect_calc_result=24.0,
    ),
    GoldenCase(
        name="no_tool_needed",
        user="What is the capital of France?",
        script=[FakeTurn(content="Paris is the capital of France.")],
        expect_final_contains="Paris",
        expect_tool_names=[],
        expect_calc_result=None,
    ),
]


@pytest.mark.parametrize("case", GOLDEN_CASES, ids=lambda c: c.name)
def test_golden_case(case: GoldenCase) -> None:
    agent = Agent(
        llm=FakeLLM(script=case.script),
        tools=built_in_registry(),
        max_steps=5,
    )
    result = agent.run(case.user)

    assert case.expect_final_contains in result.final_text
    assert [tc.name for tc in result.tool_calls] == case.expect_tool_names
    assert result.stopped_reason == "completed"

    if case.expect_calc_result is not None:
        assert result.steps, "expected at least one step with tool use"
        payload = json.loads(result.steps[0].tool_results[0].content)
        assert payload["result"] == case.expect_calc_result
