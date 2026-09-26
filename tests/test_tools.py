"""Tests for built-in tools and the registry."""

from __future__ import annotations

import json

from agent_runtime.tools import ToolRegistry, built_in_registry, calculator, tool


def test_calculator_basic() -> None:
    raw = calculator("2 + 3 * 4")
    data = json.loads(raw)
    assert data["result"] == 14.0


def test_calculator_sqrt() -> None:
    raw = calculator("sqrt(16) + 1")
    data = json.loads(raw)
    assert data["result"] == 5.0


def test_calculator_rejects_bad_chars() -> None:
    raw = calculator("__import__('os').system('id')")
    data = json.loads(raw)
    assert "error" in data


def test_calculator_rejects_names() -> None:
    raw = calculator("os.getcwd()")
    data = json.loads(raw)
    assert "error" in data


def test_registry_register_and_call() -> None:
    reg = ToolRegistry()

    @reg.register(description="Add two numbers")
    def add(a: int, b: int) -> dict:
        return {"sum": a + b}

    assert "add" in reg
    assert len(reg) == 1
    out = json.loads(reg.call("add", {"a": 2, "b": 5}))
    assert out["sum"] == 7


def test_tool_decorator_default_registry() -> None:
    # Use a unique name to avoid colliding across test runs if module reloaded
    @tool(name="echo_test_unique", description="Echo text")
    def echo_test_unique(text: str) -> str:
        return text

    from agent_runtime.tools import default_registry

    assert "echo_test_unique" in default_registry()
    assert default_registry().call("echo_test_unique", {"text": "hi"}) == "hi"


def test_built_in_registry_has_calculator() -> None:
    reg = built_in_registry()
    specs = {s["name"] for s in reg.list_specs()}
    assert specs == {"calculator", "http_get"}
    data = json.loads(reg.call("calculator", {"expression": "10 / 2"}))
    assert data["result"] == 5.0
