"""Scripted scenarios shared by the golden-trajectory tests and the examples."""

from __future__ import annotations

from dataclasses import dataclass

from agent_runtime import Agent, FakeLLM, FakeTurn, Usage, built_in_registry


@dataclass
class Scenario:
    name: str
    user: str
    script: list[FakeTurn]
    max_steps: int = 5

    def build_agent(self, **kwargs) -> Agent:
        return Agent(
            llm=FakeLLM(script=self.script, model="example-small"),
            tools=built_in_registry(),
            max_steps=self.max_steps,
            **kwargs,
        )


def _calc(call_id: str, expr: str) -> dict:
    return {"id": call_id, "name": "calculator", "arguments": {"expression": expr}}


SCENARIOS: list[Scenario] = [
    Scenario(
        name="single_tool_call",
        user="Compute 12 * 11",
        script=[
            FakeTurn(tool_calls=[_calc("c1", "12 * 11")], usage=Usage(310, 22)),
            FakeTurn(content="12 * 11 = 132.", usage=Usage(352, 9)),
        ],
    ),
    Scenario(
        name="parallel_tool_calls",
        user="What are 2**10 and sqrt(144)?",
        script=[
            FakeTurn(
                tool_calls=[_calc("c1", "2**10"), _calc("c2", "sqrt(144)")],
                usage=Usage(330, 41, cache_read_input_tokens=256),
            ),
            FakeTurn(content="2**10 is 1024 and sqrt(144) is 12.", usage=Usage(410, 15)),
        ],
    ),
    Scenario(
        name="recover_from_tool_error",
        user="Divide 10 by 0, and if that fails divide 10 by 4",
        script=[
            FakeTurn(tool_calls=[_calc("c1", "10 / 0")], usage=Usage(320, 20)),
            FakeTurn(tool_calls=[_calc("c2", "10 / 4")], usage=Usage(370, 20)),
            FakeTurn(content="10 / 0 is undefined; 10 / 4 = 2.5.", usage=Usage(420, 16)),
        ],
    ),
    Scenario(
        name="no_tool_needed",
        user="Say hello",
        script=[FakeTurn(content="Hello!", usage=Usage(280, 3))],
    ),
    Scenario(
        name="hits_max_steps",
        user="Keep adding one",
        script=[FakeTurn(tool_calls=[_calc(f"c{i}", f"{i} + 1")], usage=Usage(300, 10)) for i in range(5)],
        max_steps=3,
    ),
]

BY_NAME: dict[str, Scenario] = {s.name: s for s in SCENARIOS}
