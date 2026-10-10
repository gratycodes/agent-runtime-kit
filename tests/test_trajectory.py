"""Trajectory record/replay and golden-trajectory regression tests.

Golden files live in tests/golden/. After an intentional behaviour change,
regenerate them with::

    UPDATE_GOLDEN=1 pytest tests/test_trajectory.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from scenarios import BY_NAME, SCENARIOS

from agent_runtime import (
    Agent,
    FakeLLM,
    FakeTurn,
    ReplayLLM,
    ReplayToolRegistry,
    ToolRegistry,
    Trajectory,
    TrajectoryMismatch,
    Usage,
    built_in_registry,
    diff_trajectories,
)

GOLDEN_DIR = Path(__file__).parent / "golden"
UPDATE = os.environ.get("UPDATE_GOLDEN") == "1"


def _run(name: str) -> Trajectory:
    scenario = BY_NAME[name]
    result = scenario.build_agent().run(scenario.user)
    return Trajectory.from_run(scenario.user, result)


@pytest.mark.parametrize("name", [s.name for s in SCENARIOS])
def test_matches_golden_trajectory(name: str) -> None:
    path = GOLDEN_DIR / f"{name}.jsonl"
    actual = _run(name)
    if UPDATE:
        actual.write(path)
    assert path.exists(), f"missing golden file {path}; run with UPDATE_GOLDEN=1"
    diffs = diff_trajectories(Trajectory.read(path), actual, compare_usage=True)
    assert diffs == [], "\n".join(diffs)


def test_golden_files_are_stable_text() -> None:
    # Re-serialising must be byte-identical, so golden diffs in PRs stay readable.
    for path in GOLDEN_DIR.glob("*.jsonl"):
        assert Trajectory.read(path).to_jsonl() == path.read_text(encoding="utf-8"), path.name


def test_jsonl_round_trip_keeps_everything() -> None:
    traj = _run("parallel_tool_calls")
    again = Trajectory.from_jsonl(traj.to_jsonl())
    assert again == traj
    assert again.usage == Usage(740, 56, 256, 0)
    assert [e.name for e in again.tool_events] == ["calculator", "calculator"]
    assert json.loads(again.tool_events[1].output)["result"] == 12.0


def test_diff_reports_behaviour_changes() -> None:
    golden = _run("single_tool_call")
    changed = BY_NAME["single_tool_call"]
    llm = FakeLLM(
        script=[
            FakeTurn(tool_calls=[{"id": "c1", "name": "calculator", "arguments": {"expression": "12 * 12"}}]),
            FakeTurn(content="144"),
        ]
    )
    actual = Trajectory.from_run(changed.user, Agent(llm=llm).run(changed.user))
    diffs = diff_trajectories(golden, actual)
    assert any("12 * 11" in d and "12 * 12" in d for d in diffs)
    assert any(d.startswith("final_text") for d in diffs)
    assert not any("usage" in d for d in diffs), "usage ignored unless compare_usage=True"
    assert any("usage" in d for d in diffs_with_usage(golden, actual))


def diffs_with_usage(a: Trajectory, b: Trajectory) -> list[str]:
    return diff_trajectories(a, b, compare_usage=True)


def test_diff_ignores_generated_tool_call_ids_by_default() -> None:
    a = _run("single_tool_call")
    b = Trajectory.from_jsonl(a.to_jsonl().replace('"c1"', '"call_x9"'))
    assert diff_trajectories(a, b) == []
    assert diff_trajectories(a, b, compare_tool_call_ids=True) != []


def test_diff_reports_missing_and_extra_events() -> None:
    a = _run("parallel_tool_calls")
    b = Trajectory.from_jsonl(a.to_jsonl())
    b.events.pop()
    assert any("missing" in d for d in diff_trajectories(a, b))
    assert any("unexpected extra" in d for d in diff_trajectories(b, a))


def test_replay_runs_agent_offline_without_executing_tools() -> None:
    calls: list[str] = []
    real = ToolRegistry()

    @real.register(name="lookup_order", description="Look up an order")
    def lookup_order(order_id: str) -> dict:
        calls.append(order_id)
        return {"order_id": order_id, "status": "shipped"}

    script = [
        FakeTurn(tool_calls=[{"id": "t1", "name": "lookup_order", "arguments": {"order_id": "A-1"}}],
                 usage=Usage(200, 15), model="example-large"),
        FakeTurn(content="Order A-1 has shipped.", usage=Usage(260, 8), model="example-large"),
    ]
    recorded = Trajectory.from_run("Where is A-1?", Agent(llm=FakeLLM(script=script), tools=real).run("Where is A-1?"))
    assert calls == ["A-1"]

    replay_agent = Agent(llm=ReplayLLM(recorded), tools=ReplayToolRegistry(recorded, specs_from=real))
    result = replay_agent.run("Where is A-1?")
    assert calls == ["A-1"], "replay must not execute the real tool"
    assert result.final_text == "Order A-1 has shipped."
    assert result.usage == Usage(460, 23)
    assert replay_agent.llm.model == "example-large"
    assert diff_trajectories(recorded, Trajectory.from_run("Where is A-1?", result), compare_usage=True) == []


def test_replay_tools_reject_divergent_calls() -> None:
    recorded = _run("single_tool_call")
    tools = ReplayToolRegistry(recorded)
    with pytest.raises(TrajectoryMismatch, match="recorded calculator"):
        tools.call("calculator", {"expression": "1 + 1"})
    assert tools.call("calculator", {"expression": "12 * 11"}) == recorded.tool_events[0].output
    with pytest.raises(TrajectoryMismatch, match="extra tool call"):
        tools.call("calculator", {"expression": "12 * 11"})


def test_replay_llm_rejects_extra_model_calls() -> None:
    llm = ReplayLLM(_run("no_tool_needed"))
    llm.complete([])
    assert llm.exhausted
    with pytest.raises(TrajectoryMismatch, match="only 1 were recorded"):
        llm.complete([])


def test_replay_registry_keeps_real_tool_specs() -> None:
    tools = ReplayToolRegistry(_run("single_tool_call"), specs_from=built_in_registry())
    assert {s["name"] for s in tools.list_specs()} == {"calculator", "http_get"}


def test_read_rejects_bad_files(tmp_path: Path) -> None:
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"kind": "llm", "step": 0}\n')
    with pytest.raises(ValueError, match="must start"):
        Trajectory.read(bad)
    bad.write_text('{"kind": "run", "version": 99, "user": "x"}\n')
    with pytest.raises(ValueError, match="version"):
        Trajectory.read(bad)
    bad.write_text('{"kind": "run", "version": 1, "user": "x"}\n{"kind": "weird"}\n')
    with pytest.raises(ValueError, match="unknown record kind"):
        Trajectory.read(bad)
