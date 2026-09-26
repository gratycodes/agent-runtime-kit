"""Short-term transcript memory for the agent loop."""

from __future__ import annotations

from agent_runtime.types import Message


class TranscriptMemory:
    """In-memory ordered message buffer (short-term / session scope)."""

    def __init__(self, system_prompt: str | None = None) -> None:
        self._messages: list[Message] = []
        if system_prompt:
            self._messages.append(Message(role="system", content=system_prompt))

    def add(self, message: Message) -> None:
        self._messages.append(message)

    def extend(self, messages: list[Message]) -> None:
        self._messages.extend(messages)

    def get(self) -> list[Message]:
        return list(self._messages)

    def clear(self, *, keep_system: bool = True) -> None:
        if keep_system and self._messages and self._messages[0].role == "system":
            self._messages = [self._messages[0]]
        else:
            self._messages = []

    def __len__(self) -> int:
        return len(self._messages)
