"""Tool registry, decorator, and built-in tools."""

from __future__ import annotations

import ast
import json
import math
import operator
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass
class ToolSpec:
    """Metadata + callable for a registered tool."""

    name: str
    description: str
    parameters: dict[str, Any]
    fn: Callable[..., Any]


class ToolRegistry:
    """Named collection of tools the agent can call."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(
        self,
        fn: Callable[..., Any] | None = None,
        *,
        name: str | None = None,
        description: str = "",
        parameters: dict[str, Any] | None = None,
    ) -> Callable[..., Any] | Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Register a tool. Usable as ``@registry.register`` or ``@registry.register(name=...)``."""

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            tool_name = name or func.__name__
            tool_desc = description or (func.__doc__ or "").strip().split("\n")[0]
            schema = parameters or _infer_parameters(func)
            self._tools[tool_name] = ToolSpec(
                name=tool_name,
                description=tool_desc,
                parameters=schema,
                fn=func,
            )
            return func

        if fn is not None:
            return decorator(fn)
        return decorator

    def get(self, name: str) -> ToolSpec:
        if name not in self._tools:
            raise KeyError(f"Unknown tool: {name}")
        return self._tools[name]

    def list_specs(self) -> list[dict[str, Any]]:
        return [
            {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            }
            for t in self._tools.values()
        ]

    def call(self, name: str, arguments: dict[str, Any]) -> str:
        spec = self.get(name)
        try:
            result = spec.fn(**arguments)
        except TypeError as exc:
            return json.dumps({"error": f"Invalid arguments: {exc}"})
        except Exception as exc:  # noqa: BLE001 — surface tool errors to the LLM
            return json.dumps({"error": str(exc)})
        if isinstance(result, str):
            return result
        return json.dumps(result)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)


# Module-level decorator that targets a default registry instance.
_default_registry = ToolRegistry()


def tool(
    fn: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str = "",
    parameters: dict[str, Any] | None = None,
) -> Callable[..., Any] | Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Register a function on the default tool registry."""
    return _default_registry.register(
        fn, name=name, description=description, parameters=parameters
    )


def default_registry() -> ToolRegistry:
    return _default_registry


def _infer_parameters(fn: Callable[..., Any]) -> dict[str, Any]:
    """Minimal JSON-schema-ish parameter object from annotations."""
    import inspect

    sig = inspect.signature(fn)
    properties: dict[str, Any] = {}
    required: list[str] = []
    for pname, param in sig.parameters.items():
        if pname in ("self", "cls"):
            continue
        ann = param.annotation
        json_type = "string"
        if ann is int:
            json_type = "integer"
        elif ann is float:
            json_type = "number"
        elif ann is bool:
            json_type = "boolean"
        elif ann is list or getattr(ann, "__origin__", None) is list:
            json_type = "array"
        properties[pname] = {"type": json_type}
        if param.default is inspect.Parameter.empty:
            required.append(pname)
    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


# ---------------------------------------------------------------------------
# Built-in tools
# ---------------------------------------------------------------------------

_SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

_SAFE_FUNCS = {
    "sqrt": math.sqrt,
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _SAFE_OPS:
            raise ValueError(f"Unsupported operator: {op_type.__name__}")
        return _SAFE_OPS[op_type](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _SAFE_OPS:
            raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
        return _SAFE_OPS[op_type](_eval_node(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        fname = node.func.id
        if fname not in _SAFE_FUNCS:
            raise ValueError(f"Unsupported function: {fname}")
        args = [_eval_node(a) for a in node.args]
        return float(_SAFE_FUNCS[fname](*args))
    raise ValueError(f"Unsupported expression node: {type(node).__name__}")


def calculator(expression: str) -> str:
    """Evaluate a safe arithmetic expression (e. of ``2 + 3 * sqrt(4)``)."""
    cleaned = expression.strip()
    if not cleaned or len(cleaned) > 200:
        return json.dumps({"error": "Expression empty or too long"})
    if re.search(r"[^0-9+\-*/%().\s,a-zA-Z_]", cleaned):
        return json.dumps({"error": "Expression contains disallowed characters"})
    try:
        tree = ast.parse(cleaned, mode="eval")
        value = _eval_node(tree)
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"error": str(exc)})
    return json.dumps({"expression": cleaned, "result": value})


_HTTP_TIMEOUT_S = 5.0
_HTTP_MAX_BYTES = 50_000


def http_get(url: str) -> str:
    """Fetch a URL with timeout and response size limit (safe for demos)."""
    if not url.startswith(("http://", "https://")):
        return json.dumps({"error": "Only http/https URLs are allowed"})
    req = Request(url, headers={"User-Agent": "agent-runtime-kit/0.1"})
    try:
        with urlopen(req, timeout=_HTTP_TIMEOUT_S) as resp:  # noqa: S310 — URL scheme checked
            raw = resp.read(_HTTP_MAX_BYTES + 1)
            truncated = len(raw) > _HTTP_MAX_BYTES
            body = raw[:_HTTP_MAX_BYTES].decode("utf-8", errors="replace")
            return json.dumps(
                {
                    "url": url,
                    "status": getattr(resp, "status", 200),
                    "truncated": truncated,
                    "body": body,
                }
            )
    except HTTPError as exc:
        return json.dumps({"error": f"HTTP {exc.code}: {exc.reason}"})
    except URLError as exc:
        return json.dumps({"error": f"URL error: {exc.reason}"})
    except TimeoutError:
        return json.dumps({"error": "Request timed out"})
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"error": str(exc)})


def built_in_registry() -> ToolRegistry:
    """Fresh registry with calculator + http_get."""
    reg = ToolRegistry()
    reg.register(
        calculator,
        name="calculator",
        description="Evaluate a safe arithmetic expression and return the numeric result.",
        parameters={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Arithmetic expression, e.g. '2 + 3 * 4'",
                }
            },
            "required": ["expression"],
        },
    )
    reg.register(
        http_get,
        name="http_get",
        description="HTTP GET a URL (timeout + size limit). Returns status and body text.",
        parameters={
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "http or https URL to fetch"}
            },
            "required": ["url"],
        },
    )
    return reg
