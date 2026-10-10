"""Optional OpenTelemetry tracing that follows the GenAI semantic conventions.

The agent loop talks to a small tracing interface. ``NoopTracing`` (the
default) does nothing and needs no dependencies. ``GenAITracing`` emits real
OpenTelemetry spans and is only importable-at-runtime when the ``otel`` extra
is installed::

    pip install "agent-runtime-kit[otel]"

Spans produced per ``Agent.run``:

* ``invoke_agent {agent name}`` (INTERNAL) wrapping the whole run, with the
  summed token usage of the run.
* ``chat {model}`` (CLIENT) for every model call, with token usage, the
  response model and finish reason.
* ``execute_tool {tool name}`` (INTERNAL) for every tool call.

Attribute names follow https://github.com/open-telemetry/semantic-conventions-genai.
Prompt and completion text are never recorded. Tool arguments and results are
recorded only when ``capture_tool_content=True``.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any, Protocol

from agent_runtime.types import Usage

AttributeValue = str | bool | int | float | list[str]

# GenAI semantic convention attribute keys.
OPERATION_NAME = "gen_ai.operation.name"
PROVIDER_NAME = "gen_ai.provider.name"
AGENT_NAME = "gen_ai.agent.name"
REQUEST_MODEL = "gen_ai.request.model"
RESPONSE_MODEL = "gen_ai.response.model"
FINISH_REASONS = "gen_ai.response.finish_reasons"
INPUT_TOKENS = "gen_ai.usage.input_tokens"
OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
CACHE_READ_TOKENS = "gen_ai.usage.cache_read.input_tokens"
REASONING_TOKENS = "gen_ai.usage.reasoning.output_tokens"
TOOL_NAME = "gen_ai.tool.name"
TOOL_CALL_ID = "gen_ai.tool.call.id"
TOOL_TYPE = "gen_ai.tool.type"
TOOL_CALL_ARGUMENTS = "gen_ai.tool.call.arguments"
TOOL_CALL_RESULT = "gen_ai.tool.call.result"
ERROR_TYPE = "error.type"


class SpanHandle(Protocol):
    """What the agent loop may record on an open span."""

    def set_usage(self, usage: Usage | None) -> None: ...

    def set_attribute(self, key: str, value: AttributeValue) -> None: ...

    def set_error(self, error_type: str, description: str = "") -> None: ...


class Tracing(Protocol):
    """Interface the agent loop uses; see NoopTracing and GenAITracing."""

    def invoke_agent(
        self, *, agent_name: str, provider: str | None, request_model: str | None,
        attributes: Mapping[str, AttributeValue],
    ) -> Any: ...

    def chat(
        self, *, provider: str | None, request_model: str | None,
        attributes: Mapping[str, AttributeValue],
    ) -> Any: ...

    def execute_tool(
        self, *, tool_name: str, call_id: str, agent_name: str,
        arguments: Mapping[str, Any], attributes: Mapping[str, AttributeValue],
    ) -> Any: ...

    def record_tool_result(self, span: SpanHandle, result: str) -> None: ...


class _NoopSpan:
    def set_usage(self, usage: Usage | None) -> None:
        pass

    def set_attribute(self, key: str, value: AttributeValue) -> None:
        pass

    def set_error(self, error_type: str, description: str = "") -> None:
        pass


class NoopTracing:
    """Default tracing: records nothing and imports nothing."""

    _span = _NoopSpan()

    @contextmanager
    def invoke_agent(self, **_: Any) -> Iterator[SpanHandle]:
        yield self._span

    @contextmanager
    def chat(self, **_: Any) -> Iterator[SpanHandle]:
        yield self._span

    @contextmanager
    def execute_tool(self, **_: Any) -> Iterator[SpanHandle]:
        yield self._span

    def record_tool_result(self, span: SpanHandle, result: str) -> None:
        pass


class _OtelSpan:
    def __init__(self, span: Any, status_cls: Any, status_code: Any) -> None:
        self._span = span
        self._status_cls = status_cls
        self._status_code = status_code

    def set_usage(self, usage: Usage | None) -> None:
        if usage is None:
            return
        self._span.set_attribute(INPUT_TOKENS, usage.input_tokens)
        self._span.set_attribute(OUTPUT_TOKENS, usage.output_tokens)
        if usage.cache_read_input_tokens:
            self._span.set_attribute(CACHE_READ_TOKENS, usage.cache_read_input_tokens)
        if usage.reasoning_output_tokens:
            self._span.set_attribute(REASONING_TOKENS, usage.reasoning_output_tokens)

    def set_attribute(self, key: str, value: AttributeValue) -> None:
        self._span.set_attribute(key, value)

    def set_error(self, error_type: str, description: str = "") -> None:
        self._span.set_attribute(ERROR_TYPE, error_type)
        self._span.set_status(self._status_cls(self._status_code.ERROR, description or error_type))


class GenAITracing:
    """Emit GenAI semantic-convention spans through OpenTelemetry.

    Args:
        tracer_provider: provider to use; defaults to the global one.
        attributes: extra attributes put on every span, e.g. ``{"feature":
            "refund-assistant", "tenant.id": "tenant-a"}``. Useful for cost
            attribution downstream.
        capture_tool_content: record tool arguments and results on
            ``execute_tool`` spans. Off by default because they can contain
            user data.
    """

    def __init__(
        self,
        tracer_provider: Any = None,
        *,
        attributes: Mapping[str, AttributeValue] | None = None,
        capture_tool_content: bool = False,
    ) -> None:
        try:
            from opentelemetry import trace
            from opentelemetry.trace import SpanKind, Status, StatusCode
        except ImportError as exc:  # pragma: no cover - exercised without the extra
            raise ImportError(
                "GenAITracing needs OpenTelemetry. Install with: "
                'pip install "agent-runtime-kit[otel]"'
            ) from exc
        from agent_runtime import __version__

        self._tracer = trace.get_tracer(
            "agent_runtime", __version__, tracer_provider=tracer_provider
        )
        self._kind = SpanKind
        self._status = Status
        self._status_code = StatusCode
        self._defaults = dict(attributes or {})
        self._capture = capture_tool_content

    @contextmanager
    def _span(
        self, name: str, kind: Any, attributes: Mapping[str, AttributeValue]
    ) -> Iterator[SpanHandle]:
        attrs = {**self._defaults, **{k: v for k, v in attributes.items() if v is not None}}
        with self._tracer.start_as_current_span(
            name, kind=kind, attributes=attrs, record_exception=True, set_status_on_exception=True
        ) as span:
            handle = _OtelSpan(span, self._status, self._status_code)
            try:
                yield handle
            except BaseException as exc:
                span.set_attribute(ERROR_TYPE, type(exc).__qualname__)
                raise

    def invoke_agent(
        self, *, agent_name: str, provider: str | None, request_model: str | None,
        attributes: Mapping[str, AttributeValue],
    ) -> Any:
        return self._span(
            f"invoke_agent {agent_name}",
            self._kind.INTERNAL,
            {
                **attributes,
                OPERATION_NAME: "invoke_agent",
                AGENT_NAME: agent_name,
                **_optional(PROVIDER_NAME, provider),
                **_optional(REQUEST_MODEL, request_model),
            },
        )

    def chat(
        self, *, provider: str | None, request_model: str | None,
        attributes: Mapping[str, AttributeValue],
    ) -> Any:
        name = f"chat {request_model}" if request_model else "chat"
        return self._span(
            name,
            self._kind.CLIENT,
            {
                **attributes,
                OPERATION_NAME: "chat",
                **_optional(PROVIDER_NAME, provider),
                **_optional(REQUEST_MODEL, request_model),
            },
        )

    def execute_tool(
        self, *, tool_name: str, call_id: str, agent_name: str,
        arguments: Mapping[str, Any], attributes: Mapping[str, AttributeValue],
    ) -> Any:
        attrs: dict[str, AttributeValue] = {
            **attributes,
            OPERATION_NAME: "execute_tool",
            TOOL_NAME: tool_name,
            TOOL_CALL_ID: call_id,
            TOOL_TYPE: "function",
            AGENT_NAME: agent_name,
        }
        if self._capture:
            attrs[TOOL_CALL_ARGUMENTS] = json.dumps(dict(arguments), sort_keys=True)
        return self._span(f"execute_tool {tool_name}", self._kind.INTERNAL, attrs)

    def record_tool_result(self, span: SpanHandle, result: str) -> None:
        if self._capture:
            span.set_attribute(TOOL_CALL_RESULT, result)


def _optional(key: str, value: str | None) -> dict[str, AttributeValue]:
    return {key: value} if value else {}


def tool_error_type(result: str) -> str | None:
    """Return an error.type for a tool result that reports an error.

    ToolRegistry turns tool exceptions into ``{"error": "..."}`` JSON so the
    model can see them; tracing should still mark those calls as failed.
    """
    try:
        parsed = json.loads(result)
    except (ValueError, TypeError):
        return None
    if isinstance(parsed, dict) and "error" in parsed:
        return "tool_error"
    return None
