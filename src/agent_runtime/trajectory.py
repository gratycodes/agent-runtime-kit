"""Record agent runs as JSONL trajectories, replay them, and diff them.

A trajectory is what an agent did in one run: each model response (text,
tool calls, usage, model), each tool result, and the outcome. Uses:

* **Record** a run against a real model once, commit the JSONL file.
* **Replay** it offline: ``ReplayLLM`` returns the recorded model responses
  and ``ReplayToolRegistry`` returns the recorded tool outputs, so the agent
  loop runs deterministically with no API key and no side effects.
* **Regression-test** it: ``diff_trajectories`` compares a fresh run with a
  golden file and lists every divergence (different tool, different
  arguments, different answer), ignoring volatile fields like latency.

File format, one JSON object per line::

    {"kind": "run", "version": 1, "user": "...", "agent": "..."}
    {"kind": "llm", "step": 0, "content": "", "tool_calls": [...], "usage": {...}, "model": "..."}
    {"kind": "tool", "step": 0, "tool_call_id": "...", "name": "...", "arguments": {...}, "output": "..."}
    {"kind": "result", "final_text": "...", "stopped_reason": "completed", "usage": {...}}
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent_runtime.tools import ToolRegistry
from agent_runtime.types import Message, RunResult, ToolCall, Usage

FORMAT_VERSION = 1


class TrajectoryMismatch(AssertionError):
    """Raised during replay when the agent asks for something not recorded."""


@dataclass
class LLMEvent:
    step: int
    content: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    usage: Usage | None = None
    model: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "kind": "llm",
            "step": self.step,
            "content": self.content,
            "tool_calls": self.tool_calls,
            "usage": self.usage.to_dict() if self.usage else None,
            "model": self.model,
        }


@dataclass
class ToolEvent:
    step: int
    tool_call_id: str
    name: str
    arguments: dict[str, Any]
    output: str

    def to_json(self) -> dict[str, Any]:
        return {
            "kind": "tool",
            "step": self.step,
            "tool_call_id": self.tool_call_id,
            "name": self.name,
            "arguments": self.arguments,
            "output": self.output,
        }


@dataclass
class Trajectory:
    user: str
    events: list[LLMEvent | ToolEvent] = field(default_factory=list)
    final_text: str = ""
    stopped_reason: str = "completed"
    usage: Usage = field(default_factory=Usage)
    agent: str = "agent"

    # -- building -----------------------------------------------------------

    @classmethod
    def from_run(cls, user: str, result: RunResult, *, agent: str = "agent") -> Trajectory:
        """Build a trajectory from a finished ``Agent.run`` result."""
        events: list[LLMEvent | ToolEvent] = []
        for step in result.steps:
            msg = step.assistant_message
            calls = msg.tool_calls or []
            events.append(
                LLMEvent(
                    step=step.index,
                    content=msg.content or "",
                    tool_calls=[
                        {"id": tc.id, "name": tc.name, "arguments": tc.arguments} for tc in calls
                    ],
                    usage=msg.usage,
                    model=msg.model,
                )
            )
            by_id = {tc.id: tc for tc in calls}
            for tr in step.tool_results:
                tc = by_id.get(tr.tool_call_id or "")
                events.append(
                    ToolEvent(
                        step=step.index,
                        tool_call_id=tr.tool_call_id or "",
                        name=tr.name or (tc.name if tc else ""),
                        arguments=tc.arguments if tc else {},
                        output=tr.content,
                    )
                )
        return cls(
            user=user,
            events=events,
            final_text=result.final_text,
            stopped_reason=result.stopped_reason,
            usage=result.usage,
            agent=agent,
        )

    @property
    def llm_events(self) -> list[LLMEvent]:
        return [e for e in self.events if isinstance(e, LLMEvent)]

    @property
    def tool_events(self) -> list[ToolEvent]:
        return [e for e in self.events if isinstance(e, ToolEvent)]

    # -- JSONL --------------------------------------------------------------

    def to_jsonl(self) -> str:
        lines = [{"kind": "run", "version": FORMAT_VERSION, "user": self.user, "agent": self.agent}]
        lines += [e.to_json() for e in self.events]
        lines.append(
            {
                "kind": "result",
                "final_text": self.final_text,
                "stopped_reason": self.stopped_reason,
                "usage": self.usage.to_dict(),
            }
        )
        return "".join(json.dumps(line, sort_keys=True) + "\n" for line in lines)

    def write(self, path: str | Path) -> None:
        Path(path).write_text(self.to_jsonl(), encoding="utf-8")

    @classmethod
    def from_jsonl(cls, text: str) -> Trajectory:
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
        if not records or records[0].get("kind") != "run":
            raise ValueError("trajectory must start with a 'run' record")
        head = records[0]
        if head.get("version") != FORMAT_VERSION:
            raise ValueError(f"unsupported trajectory version {head.get('version')!r}")
        traj = cls(user=head["user"], agent=head.get("agent", "agent"))
        for i, rec in enumerate(records[1:], start=2):
            kind = rec.get("kind")
            if kind == "llm":
                traj.events.append(
                    LLMEvent(
                        step=rec["step"],
                        content=rec.get("content", ""),
                        tool_calls=rec.get("tool_calls", []),
                        usage=Usage.from_dict(rec.get("usage")),
                        model=rec.get("model"),
                    )
                )
            elif kind == "tool":
                traj.events.append(
                    ToolEvent(
                        step=rec["step"],
                        tool_call_id=rec.get("tool_call_id", ""),
                        name=rec["name"],
                        arguments=rec.get("arguments", {}),
                        output=rec["output"],
                    )
                )
            elif kind == "result":
                traj.final_text = rec.get("final_text", "")
                traj.stopped_reason = rec.get("stopped_reason", "completed")
                traj.usage = Usage.from_dict(rec.get("usage")) or Usage()
            else:
                raise ValueError(f"line {i}: unknown record kind {kind!r}")
        return traj

    @classmethod
    def read(cls, path: str | Path) -> Trajectory:
        return cls.from_jsonl(Path(path).read_text(encoding="utf-8"))


# -- replay ------------------------------------------------------------------


class ReplayLLM:
    """LLM that returns the model responses recorded in a trajectory, in order.

    It keeps the recorded model name and usage, so traces and cost numbers
    from a replay match the original run.
    """

    provider_name = "replay"

    def __init__(self, trajectory: Trajectory) -> None:
        self._events = trajectory.llm_events
        self._index = 0
        first = self._events[0].model if self._events else None
        self.model = first or "replay"

    def complete(self, messages: list[Message], tools: list[dict[str, Any]] | None = None) -> Message:
        if self._index >= len(self._events):
            raise TrajectoryMismatch(
                f"agent made model call #{self._index + 1}, but only "
                f"{len(self._events)} were recorded"
            )
        ev = self._events[self._index]
        self._index += 1
        calls = [
            ToolCall(id=tc.get("id", ""), name=tc["name"], arguments=tc.get("arguments", {}))
            for tc in ev.tool_calls
        ]
        return Message(
            role="assistant",
            content=ev.content,
            tool_calls=calls or None,
            usage=ev.usage,
            model=ev.model,
        )

    @property
    def exhausted(self) -> bool:
        return self._index >= len(self._events)


class ReplayToolRegistry(ToolRegistry):
    """Tool registry that returns recorded tool outputs instead of running tools.

    Calls must arrive in the recorded order with the recorded name and
    arguments; anything else raises ``TrajectoryMismatch``. Tool specs are
    copied from ``specs_from`` (usually the real registry) so the model sees
    the same tool list it saw when recording.
    """

    def __init__(self, trajectory: Trajectory, specs_from: ToolRegistry | None = None) -> None:
        super().__init__()
        self._events = trajectory.tool_events
        self._index = 0
        names = {e.name for e in self._events}
        if specs_from is not None:
            self._tools = dict(specs_from._tools)  # noqa: SLF001 - same class family
        for name in names - set(self._tools):
            self.register(lambda **_: "", name=name, description="(recorded)",
                          parameters={"type": "object", "properties": {}})

    def call(self, name: str, arguments: dict[str, Any]) -> str:
        if self._index >= len(self._events):
            raise TrajectoryMismatch(f"unexpected extra tool call {name}({arguments})")
        ev = self._events[self._index]
        if ev.name != name or ev.arguments != arguments:
            raise TrajectoryMismatch(
                f"tool call #{self._index + 1}: recorded {ev.name}({ev.arguments}), "
                f"got {name}({arguments})"
            )
        self._index += 1
        return ev.output


# -- regression diff ---------------------------------------------------------


def diff_trajectories(
    expected: Trajectory,
    actual: Trajectory,
    *,
    compare_usage: bool = False,
    compare_tool_call_ids: bool = False,
) -> list[str]:
    """Return human-readable differences between two trajectories.

    An empty list means they match. Latency is never compared; tool call ids
    and token usage are ignored unless asked for, since they usually vary
    between runs without meaning anything changed.
    """
    diffs: list[str] = []
    if expected.user != actual.user:
        diffs.append(f"user message: expected {expected.user!r}, got {actual.user!r}")

    exp_events, act_events = expected.events, actual.events
    for i in range(max(len(exp_events), len(act_events))):
        if i >= len(exp_events):
            diffs.append(f"event {i}: unexpected extra {_describe(act_events[i])}")
            continue
        if i >= len(act_events):
            diffs.append(f"event {i}: missing {_describe(exp_events[i])}")
            continue
        diffs.extend(
            f"event {i}: {d}"
            for d in _diff_event(exp_events[i], act_events[i], compare_usage, compare_tool_call_ids)
        )

    if expected.final_text != actual.final_text:
        diffs.append(f"final_text: expected {expected.final_text!r}, got {actual.final_text!r}")
    if expected.stopped_reason != actual.stopped_reason:
        diffs.append(
            f"stopped_reason: expected {expected.stopped_reason!r}, got {actual.stopped_reason!r}"
        )
    if compare_usage and expected.usage != actual.usage:
        diffs.append(f"usage: expected {expected.usage}, got {actual.usage}")
    return diffs


def _describe(ev: LLMEvent | ToolEvent) -> str:
    if isinstance(ev, ToolEvent):
        return f"tool {ev.name}({ev.arguments})"
    names = [tc["name"] for tc in ev.tool_calls]
    return f"llm step {ev.step} tool_calls={names}" if names else f"llm step {ev.step} answer"


def _strip_ids(calls: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"name": c["name"], "arguments": c.get("arguments", {})} for c in calls]


def _diff_event(
    exp: LLMEvent | ToolEvent,
    act: LLMEvent | ToolEvent,
    compare_usage: bool,
    compare_ids: bool,
) -> list[str]:
    if type(exp) is not type(act):
        return [f"expected {_describe(exp)}, got {_describe(act)}"]
    out: list[str] = []
    if isinstance(exp, LLMEvent) and isinstance(act, LLMEvent):
        e_calls = exp.tool_calls if compare_ids else _strip_ids(exp.tool_calls)
        a_calls = act.tool_calls if compare_ids else _strip_ids(act.tool_calls)
        if e_calls != a_calls:
            out.append(f"tool_calls: expected {e_calls}, got {a_calls}")
        if exp.content != act.content:
            out.append(f"content: expected {exp.content!r}, got {act.content!r}")
        if compare_usage and exp.usage != act.usage:
            out.append(f"usage: expected {exp.usage}, got {act.usage}")
    elif isinstance(exp, ToolEvent) and isinstance(act, ToolEvent):
        if (exp.name, exp.arguments) != (act.name, act.arguments):
            out.append(f"tool: expected {exp.name}({exp.arguments}), got {act.name}({act.arguments})")
        if exp.output != act.output:
            out.append(f"tool output for {exp.name}: expected {exp.output!r}, got {act.output!r}")
        if compare_ids and exp.tool_call_id != act.tool_call_id:
            out.append(f"tool_call_id: expected {exp.tool_call_id!r}, got {act.tool_call_id!r}")
    return out
