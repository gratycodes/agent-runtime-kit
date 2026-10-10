"""The OTLP/JSON export example produces valid, collector-ready documents."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip("opentelemetry.exporter.otlp.proto.common", reason="needs the otel extra")

EXAMPLE = Path(__file__).parent.parent / "examples" / "export_otlp_traces.py"


def _load_example():
    spec = importlib.util.spec_from_file_location("export_otlp_traces", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _attrs(span: dict) -> dict:
    out = {}
    for kv in span.get("attributes", []):
        (value,) = kv["value"].values()
        out[kv["key"]] = value
    return out


def test_example_writes_otlp_json(tmp_path: Path) -> None:
    out = tmp_path / "traces.json"
    n = _load_example().export(out)
    assert n == 9  # 2 runs: (agent + 2 chat + 2 tools) + (agent + 2 chat + 1 tool)

    doc = json.loads(out.read_text())
    (rs,) = doc["resourceSpans"]
    assert {"key": "service.name", "value": {"stringValue": "support-bot-demo"}} in rs["resource"]["attributes"]
    spans = [s for ss in rs["scopeSpans"] for s in ss["spans"]]
    assert len(spans) == 9

    for s in spans:
        assert len(s["traceId"]) == 32 and int(s["traceId"], 16) >= 0
        assert len(s["spanId"]) == 16
        assert isinstance(s["kind"], int)

    chats = [s for s in spans if _attrs(s)["gen_ai.operation.name"] == "chat"]
    assert len(chats) == 4
    first = _attrs(next(s for s in chats if _attrs(s).get("gen_ai.response.model") == "example-large-2026-09-01"))
    assert first["gen_ai.usage.input_tokens"] == "2650"  # OTLP/JSON encodes int64 as strings
    assert first["gen_ai.usage.cache_read.input_tokens"] == "2300"
    assert first["feature"] == "refund-assistant"
    assert first["tenant.id"] == "tenant-a"

    agents = [s for s in spans if _attrs(s)["gen_ai.operation.name"] == "invoke_agent"]
    assert {a["traceId"] for a in agents} == {s["traceId"] for s in spans}, "one trace per run"
    assert len(agents) == 2


def test_committed_fixture_is_current_shape() -> None:
    fixture = EXAMPLE.parent / "fixtures" / "agent_runs.otlp.json"
    doc = json.loads(fixture.read_text())
    spans = [s for rs in doc["resourceSpans"] for ss in rs["scopeSpans"] for s in ss["spans"]]
    ops = sorted(_attrs(s)["gen_ai.operation.name"] for s in spans)
    assert ops == ["chat"] * 4 + ["execute_tool"] * 3 + ["invoke_agent"] * 2
