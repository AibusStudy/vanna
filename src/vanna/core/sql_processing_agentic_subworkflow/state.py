from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SqlProcessingInput:
    system_prompt: str | None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SqlProcessingFinalResult:
    status: str
    attempts: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "attempts": list(self.attempts),
            "errors": list(self.errors),
        }
