"""Lossless, task-local evidence with explicit navigation namespaces.

Only successful, shape-checked reads are observed. Initial alert objects and
self-describing search/AD items are references, never proof of their properties.
Unknown envelopes and failures remain in records; they are not negative findings.
"""

import json
from copy import deepcopy
from typing import Any, get_args

from bluesec1_agent.contracts import EvidenceRef, EvidenceSnapshot, ObjectKind
from bluesec1_agent.models import EntitySearchType, RelationSearchType

_ENTITY_TYPES = frozenset(get_args(EntitySearchType))
_RELATION_TYPES = frozenset(get_args(RelationSearchType))


def _is_id(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _object_shape(value: Any, kind: ObjectKind) -> bool:
    if not isinstance(value, dict) or not _is_id(value.get("id")) or not _is_id(value.get("type")):
        return False
    if kind == "relation":
        return (
            value["type"] not in _ENTITY_TYPES
            and all(_is_id(value.get(k)) for k in ("source_id", "target_id"))
            and all(
                value.get(k) is None or isinstance(value[k], dict)
                for k in ("properties", "logon_properties", "access_properties")
            )
        )
    if value["type"] in _RELATION_TYPES or not isinstance(value.get("properties"), dict):
        return False
    if value.get("host_id") is not None and not _is_id(value["host_id"]):
        return False
    relations = value.get("relation_ids", {})
    return isinstance(relations, dict) and all(
        isinstance(relations.get(k, []), list) and all(_is_id(v) for v in relations.get(k, []))
        for k in ("incoming", "outgoing")
    )


class EvidenceStore:
    """Keep every raw response and sourced version without mutable aliases.

    ``get``/``snapshot`` choose the latest observed version, falling back to a
    reference. ``history`` retains all versions; graph adjacency is their union.
    None of the public accessors expose mutable internal state.
    """

    def __init__(self) -> None:
        self._history: dict[tuple[ObjectKind, str], list[EvidenceRef]] = {}
        self._records: list[dict[str, Any]] = []
        self._triggers: dict[str, None] = {}
        self._adjacency: dict[str, set[str]] = {}
        self._ad_ids: set[str] = set()
        self._evidence_version = 0

    def __len__(self) -> int:
        return len(self._history)

    @property
    def trigger_ids(self) -> tuple[str, ...]:
        return tuple(self._triggers)

    @property
    def evidence_version(self) -> int:
        """Number of distinct successfully observed object values, across both namespaces.

        References, failures and re-reading any previously observed value do not
        advance this counter. Source history still retains repeated reads.
        """
        return self._evidence_version

    def first_observed_step(self, object_id: str, kind: ObjectKind) -> int | None:
        return next(
            (
                ref.source_step
                for ref in self._history.get((kind, object_id), ())
                if ref.state == "observed"
            ),
            None,
        )

    @property
    def records(self) -> tuple[dict[str, Any], ...]:
        """Raw inputs, status and normalization outcome, including unsupported data."""
        return deepcopy(tuple(self._records))

    def is_known(self, object_id: str, kind: ObjectKind) -> bool:
        return (kind, object_id) in self._history

    def is_known_ad_id(self, object_id: str) -> bool:
        """AD lookup IDs are distinct from graph entity and relation IDs."""
        return object_id in self._ad_ids

    def history(self, object_id: str, kind: ObjectKind) -> tuple[EvidenceRef, ...]:
        return deepcopy(tuple(self._history.get((kind, object_id), ())))

    def get(self, object_id: str, kind: ObjectKind) -> EvidenceRef | None:
        versions = self._history.get((kind, object_id), ())
        if not versions:
            return None
        return deepcopy(
            next((v for v in reversed(versions) if v.state == "observed"), versions[-1])
        )

    def relation_ids_for(self, entity_id: str) -> tuple[str, ...]:
        """All known incident links, including earlier/conflicting observations."""
        return tuple(sorted(self._adjacency.get(entity_id, ())))

    def snapshot(self, *, max_items: int) -> EvidenceSnapshot:
        if max_items < 0:
            raise ValueError("max_items must be nonnegative")
        keys = list(self._history)
        items = tuple(self.get(object_id, kind) for kind, object_id in keys[:max_items])
        return EvidenceSnapshot(items=items, omitted_count=max(0, len(keys) - max_items))

    def _add(
        self,
        object_id: str,
        kind: ObjectKind,
        *,
        step: int,
        tool: str,
        fields: dict[str, Any] | None = None,
        observed: bool = False,
    ) -> None:
        ref = EvidenceRef(
            object_id,
            kind,
            "observed" if observed else "referenced",
            step,
            tool,
            deepcopy(fields or {}),
        )
        versions = self._history.setdefault((kind, object_id), [])
        if observed:
            serialized = json.dumps(ref.fields, sort_keys=True, default=str)
            if not any(
                v.state == "observed"
                and json.dumps(v.fields, sort_keys=True, default=str) == serialized
                for v in versions
            ):
                self._evidence_version += 1
        if ref not in versions:
            versions.append(ref)

    def _links(self, obj: dict[str, Any], kind: ObjectKind, *, step: int, tool: str) -> None:
        if kind == "relation":
            for key in ("source_id", "target_id"):
                endpoint = obj.get(key)
                if _is_id(endpoint):
                    self._add(endpoint, "entity", step=step, tool=tool)
                    self._adjacency.setdefault(endpoint, set()).add(obj["id"])
            return
        relations = obj.get("relation_ids", {})
        if isinstance(relations, dict):
            for direction in ("incoming", "outgoing"):
                values = relations.get(direction, [])
                if isinstance(values, list):
                    for relation_id in values:
                        if _is_id(relation_id):
                            self._add(relation_id, "relation", step=step, tool=tool)
                            self._adjacency.setdefault(obj["id"], set()).add(relation_id)
        if _is_id(obj.get("host_id")):
            self._add(obj["host_id"], "entity", step=step, tool=tool)
        properties = obj.get("properties")
        if isinstance(properties, dict) and _is_id(properties.get("ad_object_id")):
            self._ad_ids.add(properties["ad_object_id"])

    def seed_observation(self, observation: dict[str, Any]) -> None:
        """Accept dict or JSON alert_text; never extract IDs from free text."""
        alert = observation.get("alert_text")
        if isinstance(alert, str):
            try:
                alert = json.loads(alert)
            except (ValueError, RecursionError):
                alert = None
        supported = isinstance(alert, dict) and any(
            key in alert for key in ("trigger_entities", "trigger_relations")
        )
        if supported:
            for key, kind in (("trigger_entities", "entity"), ("trigger_relations", "relation")):
                objects = alert.get(key, [])
                if not isinstance(objects, list):
                    supported = False
                    continue
                for obj in objects:
                    if not isinstance(obj, dict) or not _is_id(obj.get("id")):
                        supported = False
                        continue
                    self._triggers[obj["id"]] = None
                    self._add(obj["id"], kind, step=0, tool="initial_observation", fields=obj)
                    self._links(obj, kind, step=0, tool="initial_observation")
        self._records.append(
            {
                "step": 0,
                "tool_name": "initial_observation",
                "arguments": {},
                "result": deepcopy(observation),
                "status": "ok",
                "supported": supported,
                "issue": None if supported else "unsupported_initial_observation",
            }
        )

    def ingest(
        self, *, step: int, tool_name: str, arguments: dict[str, Any], result: Any, status: str
    ) -> None:
        """Consume normalized _tool_feedback result/status exactly once per call.

        Search/AD output schemas are unavailable in the current SDK/traces. For
        those tools, only root items or immediate list items with a recognized
        id/type pair become references. Their enclosing result stays unsupported;
        arbitrary nested properties and strings are never scanned for references.
        """
        supported = False
        issue = "tool_not_ok" if status != "ok" else "unsupported_result"
        if status == "ok" and tool_name in {"get_entity", "get_relation"}:
            kind: ObjectKind = "entity" if tool_name == "get_entity" else "relation"
            expected_id = arguments.get(f"{kind}_id")
            if _object_shape(result, kind) and result["id"] == expected_id:
                self._add(
                    result["id"], kind, step=step, tool=tool_name, fields=result, observed=True
                )
                self._links(result, kind, step=step, tool=tool_name)
                supported, issue = True, None
        elif status == "ok" and tool_name in {"search", "get_ad_object_info"}:
            if isinstance(result, list):
                candidates = result
            elif isinstance(result, dict):
                candidates = (
                    [result]
                    if "id" in result
                    else [
                        item
                        for value in result.values()
                        if isinstance(value, list)
                        for item in value
                    ]
                )
            else:
                candidates = []
            for obj in candidates:
                if not isinstance(obj, dict) or not _is_id(obj.get("id")):
                    continue
                object_type = obj.get("type")
                if not isinstance(object_type, str):
                    continue
                if object_type in _ENTITY_TYPES:
                    self._add(obj["id"], "entity", step=step, tool=tool_name, fields=obj)
                elif object_type in _RELATION_TYPES:
                    self._add(obj["id"], "relation", step=step, tool=tool_name, fields=obj)
        self._records.append(
            {
                "step": step,
                "tool_name": tool_name,
                "arguments": deepcopy(arguments),
                "result": deepcopy(result),
                "status": status,
                "supported": supported,
                "issue": issue,
            }
        )
