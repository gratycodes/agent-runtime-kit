"""Deterministic FakeLLM demo showing tool use.

Run with::

    python -m agent_runtime.demo
"""

from __future__ import annotations

from agent_runtime.agent import Agent
from agent_runtime.llm import FakeLLM, FakeTurn
from agent_runtime.tools import built_in_registry


def build_demo_agent() -> Agent:
    script = [
        FakeTurn(
            content="",
            tool_calls=[
                {
                    "id": "call_calc_1",
                    "name": "calculator",
                    "arguments": {"expression": "(17 + 25) * 3"},
                }
            ],
        ),
        FakeTurn(
            content="(17 + 25) * 3 = 126. I used the calculator tool for accuracy.",
        ),
    ]
    llm = FakeLLM(script=script)
    return Agent(
        llm=llm,
        tools=built_in_registry(),
        system_prompt=(
            "You are a careful assistant. Use the calculator tool for arithmetic."
        ),
        max_steps=5,
    )


def main() -> None:
    print("=== agent-runtime-kit demo (FakeLLM) ===\n")
    agent = build_demo_agent()
    question = "What is (17 + 25) * 3?"
    print(f"User: {question}\n")
    result = agent.run(question)

    for step in result.steps:
        msg = step.assistant_message
        if msg.tool_calls:
            for tc in msg.tool_calls:
                print(f"[step {step.index}] tool → {tc.name}({tc.arguments})")
            for tr in step.tool_results:
                print(f"[step {step.index}] result ← {tr.content}")
        else:
            print(f"[step {step.index}] assistant → {msg.content}")

    print(f"\nFinal: {result.final_text}")
    print(
        f"\nStats: steps={result.step_count} "
        f"tool_calls={len(result.tool_calls)} "
        f"latency_ms={result.total_latency_ms:.1f} "
        f"reason={result.stopped_reason}"
    )


if __name__ == "__main__":
    main()
