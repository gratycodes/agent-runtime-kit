"""Serialise finished OpenTelemetry spans as OTLP/JSON.

The output is the same JSON an OTLP/HTTP exporter would POST to
``/v1/traces`` (and what the collector's file exporter writes), so it can be
used as a test fixture or sent to a collector with curl. Requires the
``otel`` extra.
"""

from __future__ import annotations

import base64
import json
from collections.abc import Sequence
from typing import Any

_ID_FIELDS = ("traceId", "spanId", "parentSpanId")


def spans_to_otlp_json(spans: Sequence[Any], *, indent: int | None = 1) -> str:
    """Encode SDK ``ReadableSpan`` objects as an OTLP/JSON traces document."""
    try:
        from google.protobuf.json_format import MessageToDict
        from opentelemetry.exporter.otlp.proto.common.trace_encoder import encode_spans
    except ImportError as exc:  # pragma: no cover - exercised without the extra
        raise ImportError(
            'spans_to_otlp_json needs the otel extra: pip install "agent-runtime-kit[otel]"'
        ) from exc

    request = encode_spans(spans)
    doc = MessageToDict(request, use_integers_for_enums=True)
    _hex_ids(doc)
    return json.dumps(doc, indent=indent, sort_keys=False)


def _hex_ids(node: Any) -> None:
    # Protobuf's JSON mapping writes bytes as base64, but OTLP/JSON requires
    # trace and span ids as lowercase hex.
    if isinstance(node, dict):
        for key, value in node.items():
            if key in _ID_FIELDS and isinstance(value, str):
                node[key] = base64.b64decode(value).hex()
            else:
                _hex_ids(value)
    elif isinstance(node, list):
        for item in node:
            _hex_ids(item)
