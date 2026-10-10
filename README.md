# agent-runtime-kit

A small Python agent loop with a tool registry, a pluggable LLM interface, transcript memory and golden evals. No framework, no graph DSL: the whole runtime is about 750 lines you can read in one sitting.

## Why

Most agent frameworks hide the core loop behind abstractions. I wanted a minimal, tested version of that loop (call the model, run the requested tools, feed results back, stop on a final answer or a step limit) with a deterministic `FakeLLM`, so tool wiring can be tested in CI without an API key.

## Features

- **Agent loop** — messages, `max_steps`, tool registry (`@tool` / `register`), `RunResult` with steps, tool calls, final text, latency counters
- **Built-in tools** — `calculator` (AST-safe arithmetic) and `http_get` (timeout + size limit)
- **Pluggable LLM** — `LLM` protocol, `FakeLLM` for deterministic tests/demos, optional `OpenAIAdapter` behind `OPENAI_API_KEY`
- **Short-term memory** — ordered transcript buffer across steps
- **Golden evals** — fixed FakeLLM scenarios with expected tool use and answers
- **CI** — GitHub Actions runs pytest on Python 3.11+

## Architecture

```mermaid
flowchart LR
  User --> Agent
  Agent --> Memory["TranscriptMemory"]
  Agent --> LLM["LLM protocol"]
  Agent --> Tools["ToolRegistry"]
  LLM --> FakeLLM
  LLM --> OpenAIAdapter
  Tools --> Calculator
  Tools --> HTTPGet["http_get"]
  Agent --> Result["RunResult"]
```

**Loop (simplified):**

1. Append user message to memory
2. Call `llm.complete(messages, tools)`
3. If the assistant returns `tool_calls`, invoke each tool, append tool results, repeat
4. Otherwise treat content as the final answer and return `RunResult`
5. Stop early on `max_steps`

## Install

```bash
# clone
git clone https://github.com/gratycodes/agent-runtime-kit.git
cd agent-runtime-kit

python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# optional OpenAI support
pip install -e ".[openai]"
```

Requires **Python 3.11+**.

## Demo

Deterministic FakeLLM scenario (no API key required):

```bash
python -m agent_runtime.demo
```

Expected shape of output: a calculator tool call for `(17 + 25) * 3`, then a final answer of `126`.

## Quick start (library)

```python
from agent_runtime import Agent, FakeLLM, FakeTurn, built_in_registry

llm = FakeLLM(
    script=[
        FakeTurn(
            tool_calls=[{
                "name": "calculator",
                "arguments": {"expression": "6 * 7"},
            }]
        ),
        FakeTurn(content="42"),
    ]
)
agent = Agent(llm=llm, tools=built_in_registry())
result = agent.run("What is 6*7?")
print(result.final_text)       # 42
print(result.tool_calls)       # [ToolCall(name='calculator', ...)]
print(result.total_latency_ms)
```

### Optional OpenAI

```python
import os
from agent_runtime import Agent, OpenAIAdapter, built_in_registry

assert os.environ.get("OPENAI_API_KEY"), "set OPENAI_API_KEY"
agent = Agent(llm=OpenAIAdapter(model="gpt-4o-mini"), tools=built_in_registry())
print(agent.run("What is 19*21?").final_text)
```

## Tests

```bash
pytest -q
```

Coverage includes:

- Agent loop (final answer, tool use, unknown tool, max_steps)
- Tool selection / calculator safety
- **3 golden eval cases** (`simple_arithmetic`, `nested_expression`, `no_tool_needed`)

## Design notes

| Choice | Why |
| --- | --- |
| Plain loop, not a graph | Interviewers can read the control flow without a framework mental model |
| `LLM` protocol + `FakeLLM` | Tests and demos stay deterministic; OpenAI is optional |
| AST calculator | Safer than `eval`; still shows tool-use plumbing |
| Bounded `http_get` | Timeout + byte cap so demos cannot hang or buffer large bodies |
| `RunResult` counters | Latency and step traces make behavior inspectable |
| Golden evals | Small, fixed scenarios that fail loudly if tool wiring regresses |

## Limitations

- Single agent only: no multi-agent orchestration, planning, or streaming.
- Memory is an in-process transcript; nothing is persisted between runs.
- `http_get` checks the URL scheme, timeout (5 s) and size (50 KB) but does not block private or internal addresses, so do not expose it to untrusted input.
- The OpenAI adapter is optional and not exercised in CI; tests use `FakeLLM`.
- Golden evals check tool selection and final answers for three fixed scenarios; they are regression tests, not a model-quality benchmark.

## Stack

Python 3.11+ (standard library only for the core), optional `openai` SDK, pytest, Hatchling, GitHub Actions (Python 3.11, 3.12, 3.13).

## Project layout

```
src/agent_runtime/
  agent.py      # multi-step loop
  tools.py      # registry + calculator + http_get
  llm.py        # protocol, FakeLLM, OpenAIAdapter
  memory.py     # short-term transcript
  types.py      # Message, ToolCall, Step, RunResult
  demo.py       # python -m agent_runtime.demo
tests/
  test_agent.py
  test_tools.py
  test_golden_evals.py
```

## License

MIT
