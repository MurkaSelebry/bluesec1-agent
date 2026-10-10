"""Local collaboration contracts; these do not change the BlueSec wire schema."""

from dataclasses import dataclass
from typing import Any, Literal, Protocol

ObjectKind = Literal["entity", "relation"]


@dataclass(frozen=True)
class EvidenceRef:
    object_id: str
    kind: ObjectKind
    state: Literal["referenced", "observed"]
    source_step: int
    source_tool: str
    fields: dict[str, Any]


@dataclass(frozen=True)
class EvidenceSnapshot:
    items: tuple[EvidenceRef, ...]
    omitted_count: int


@dataclass(frozen=True)
class Lead:
    object_id: str
    kind: ObjectKind
    priority: int
    reason: str


@dataclass(frozen=True)
class EvidenceViolation:
    code: str
    object_id: str | None
    field: str | None
    message: str


class EventSink(Protocol):
    def record(self, event: str, **payload: Any) -> None: ...
