"""LLM protocol, FakeLLM for tests, and optional OpenAI adapter."""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from agent_runtime.types import Message, ToolCall, Usage


@runtime_checkable
class LLM(Protocol):
    """Minimal chat interface the agent loop depends on."""

    def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
    ) -> Message:
        """Return the next assistant message (optionally with tool_calls)."""
        ...


@dataclass
class FakeTurn:
    """One scripted FakeLLM response."""

    content: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    usage: Usage | None = None  # optional scripted token usage
    model: str | None = None  # optional response model, defaults to FakeLLM.model


class FakeLLM:
    """Deterministic LLM that replays a scripted sequence of turns.

    Each ``complete`` call consumes the next FakeTurn. When the script is
    exhausted it returns a fixed final message.
    """

    provider_name = "fake"

    def __init__(
        self,
        script: list[FakeTurn] | None = None,
        *,
        fallback: str = "Done.",
        model: str = "fake-llm",
    ) -> None:
        self.model = model
        self._script = list(script or [])
        self._index = 0
        self._fallback = fallback
        self.calls: list[list[Message]] = []

    def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
    ) -> Message:
        self.calls.append(list(messages))
        _ = tools
        if self._index < len(self._script):
            turn = self._script[self._index]
            self._index += 1
            tool_calls = [
                ToolCall(
                    id=tc.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                    name=tc["name"],
                    arguments=tc.get("arguments", {}),
                )
                for tc in turn.tool_calls
            ]
            return Message(
                role="assistant",
                content=turn.content,
                tool_calls=tool_calls or None,
                usage=turn.usage,
                model=turn.model or self.model,
            )
        return Message(role="assistant", content=self._fallback, model=self.model)


class OpenAIAdapter:
    """Thin OpenAI Chat Completions adapter (optional; needs OPENAI_API_KEY)."""

    provider_name = "openai"

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        *,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        try:
            from openai import OpenAI  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "openai package is required for OpenAIAdapter. "
                "Install with: pip install agent-runtime-kit[openai]"
            ) from exc

        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY is not set")
        kwargs: dict[str, Any] = {"api_key": key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = OpenAI(**kwargs)
        self.model = model

    def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
    ) -> Message:
        payload = [_to_openai_message(m) for m in messages]
        kwargs: dict[str, Any] = {"model": self.model, "messages": payload}
        if tools:
            kwargs["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t.get("description", ""),
                        "parameters": t.get("parameters", {"type": "object", "properties": {}}),
                    },
                }
                for t in tools
            ]
        resp = self._client.chat.completions.create(**kwargs)
        choice = resp.choices[0].message
        tool_calls: list[ToolCall] | None = None
        if choice.tool_calls:
            tool_calls = []
            for tc in choice.tool_calls:
                args = tc.function.arguments
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {"raw": args}
                tool_calls.append(
                    ToolCall(id=tc.id, name=tc.function.name, arguments=args)
                )
        return Message(
            role="assistant",
            content=choice.content or "",
            tool_calls=tool_calls,
            usage=_usage_from_openai(getattr(resp, "usage", None)),
            model=getattr(resp, "model", None),
        )


def _usage_from_openai(u: Any) -> Usage | None:
    """Map Chat Completions usage (prompt/completion tokens) to Usage."""
    if u is None:
        return None
    prompt_details = getattr(u, "prompt_tokens_details", None)
    completion_details = getattr(u, "completion_tokens_details", None)
    return Usage(
        input_tokens=getattr(u, "prompt_tokens", 0) or 0,
        output_tokens=getattr(u, "completion_tokens", 0) or 0,
        cache_read_input_tokens=getattr(prompt_details, "cached_tokens", 0) or 0,
        reasoning_output_tokens=getattr(completion_details, "reasoning_tokens", 0) or 0,
    )


def _to_openai_message(m: Message) -> dict[str, Any]:
    msg: dict[str, Any] = {"role": m.role, "content": m.content or ""}
    if m.name:
        msg["name"] = m.name
    if m.tool_call_id:
        msg["tool_call_id"] = m.tool_call_id
    if m.tool_calls:
        msg["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": json.dumps(tc.arguments),
                },
            }
            for tc in m.tool_calls
        ]
    return msg
