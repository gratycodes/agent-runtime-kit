"""Token usage is carried from the LLM onto messages and summed per run."""

from __future__ import annotations

from types import SimpleNamespace

from agent_runtime import Agent, FakeLLM, FakeTurn, Usage, built_in_registry
from agent_runtime.llm import _usage_from_openai


def test_usage_addition_and_round_trip() -> None:
    a = Usage(input_tokens=10, output_tokens=2, cache_read_input_tokens=4)
    b = Usage(input_tokens=5, output_tokens=3, reasoning_output_tokens=1)
    total = a + b
    assert total == Usage(15, 5, 4, 1)
    assert Usage.from_dict(total.to_dict()) == total
    assert Usage.from_dict(None) is None


def test_run_sums_usage_and_keeps_response_model() -> None:
    llm = FakeLLM(
        model="example-small",
        script=[
            FakeTurn(
                tool_calls=[{"id": "c1", "name": "calculator", "arguments": {"expression": "2+2"}}],
                usage=Usage(input_tokens=120, output_tokens=18),
            ),
            FakeTurn(content="4", usage=Usage(input_tokens=160, output_tokens=6), model="example-small-2026"),
        ],
    )
    result = Agent(llm=llm, tools=built_in_registry()).run("2+2?")
    assert result.usage == Usage(input_tokens=280, output_tokens=24)
    assert result.steps[0].assistant_message.model == "example-small"
    assert result.steps[1].assistant_message.model == "example-small-2026"


def test_steps_without_usage_count_as_zero() -> None:
    result = Agent(llm=FakeLLM(script=[FakeTurn(content="hi")])).run("hello")
    assert result.usage == Usage()


def test_openai_usage_mapping_handles_details_and_missing_fields() -> None:
    u = SimpleNamespace(
        prompt_tokens=1000,
        completion_tokens=300,
        prompt_tokens_details=SimpleNamespace(cached_tokens=600),
        completion_tokens_details=SimpleNamespace(reasoning_tokens=120),
    )
    assert _usage_from_openai(u) == Usage(1000, 300, 600, 120)
    bare = SimpleNamespace(prompt_tokens=7, completion_tokens=None)
    assert _usage_from_openai(bare) == Usage(7, 0, 0, 0)
    assert _usage_from_openai(None) is None
