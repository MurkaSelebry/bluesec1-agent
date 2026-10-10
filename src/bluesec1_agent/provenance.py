"""Local provenance checks, independent of verdict correctness and graph coverage."""

from typing import Any

from bluesec1_agent.contracts import EvidenceViolation, ObjectKind
from bluesec1_agent.evidence import EvidenceStore
from bluesec1_agent.models import FinishInvestigationSubmission

# Only unambiguous ontology/action correspondences. Broad actions such as
# cleanup_to_verify and persistence_to_remove need semantic judgment by the SOC.
_ARTIFACT_TYPES = {
    "host_to_isolate": {"host"},
    "file_to_delete": {"windows_file", "linux_file"},
    "process_observed": {"windows_process", "linux_process"},
    "registry_observed": {"registry_key"},
    "file_observed": {"windows_file", "linux_file"},
    "mount_observed": {"mounted_volume"},
    "lnk_observed": {"lnk_shortcut"},
    "ticket_observed": {"kerberos_ticket"},
}


def _violation(code: str, object_id: str, field: str | None = None) -> EvidenceViolation:
    messages = {
        "unknown_object": "ID has no structured evidence reference.",
        "wrong_kind": "ID is known only in a different navigation namespace.",
        "not_observed": "ID was referenced but its object has not been successfully read.",
        "unknown_field": "Exact field is absent from all successful observations of this object.",
        "artifact_type_mismatch": "Artifact kind is incompatible with the observed entity type.",
        "conflicting_type": "Conflicting observed entity types prevent this artifact type check.",
    }
    return EvidenceViolation(code, object_id, field, messages[code])


def _check_object(
    store: EvidenceStore, object_id: str, kind: ObjectKind, *, observed: bool
) -> EvidenceViolation | None:
    ref = store.get(object_id, kind)
    if ref is None:
        other_kind = "relation" if kind == "entity" else "entity"
        return _violation(
            "wrong_kind" if store.is_known(object_id, other_kind) else "unknown_object", object_id
        )
    if observed and ref.state != "observed":
        return _violation("not_observed", object_id)
    return None


def validate_navigation(
    tool_name: str, arguments: dict[str, Any], store: EvidenceStore
) -> list[EvidenceViolation]:
    """Validate get_* namespace before RPC; searches need no existing object ID."""
    if tool_name == "get_ad_object_info":
        object_id = arguments.get("ad_object_id", "")
        if store.is_known_ad_id(object_id):
            return []
        known_elsewhere = any(store.is_known(object_id, k) for k in ("entity", "relation"))
        return [_violation("wrong_kind" if known_elsewhere else "unknown_object", object_id)]
    if tool_name not in {"get_entity", "get_relation"}:
        return []
    kind: ObjectKind = "entity" if tool_name == "get_entity" else "relation"
    error = _check_object(store, arguments.get(f"{kind}_id", ""), kind, observed=False)
    return [error] if error else []


def _has_path(data: Any, path: str) -> bool:
    if not isinstance(data, dict):
        return False
    if path in data:
        return True
    head, separator, tail = path.partition(".")
    return bool(separator) and head in data and _has_path(data[head], tail)


def _has_field(fields: dict[str, Any], field: str) -> bool:
    # The runtime accepts bare entity property names (e.g. signed). Explicit
    # paths are resolved exactly; suffix matching or neighbor lookup is forbidden.
    return _has_path(fields, field) or (
        "." not in field
        and isinstance(fields.get("properties"), dict)
        and field in fields["properties"]
    )


def validate_submission(
    submission: FinishInvestigationSubmission, store: EvidenceStore
) -> list[EvidenceViolation]:
    """Check observed IDs/fields/types, without asserting the verdict is justified.

    Empty relation property_fields are valid. No completeness requirement forces
    enumeration of all high-degree neighbors before a benign submission.
    """
    violations: list[EvidenceViolation] = []
    if submission.verdict == "benign":
        for citation in submission.legitimacy_evidence:
            kind = citation.anchor
            object_id = citation.entity_id if kind == "entity" else citation.relation_id
            error = _check_object(store, object_id, kind, observed=True)
            if error:
                violations.append(error)
                continue
            versions = [v for v in store.history(object_id, kind) if v.state == "observed"]
            for field in citation.property_fields:
                if not any(_has_field(v.fields, field) for v in versions):
                    violations.append(_violation("unknown_field", object_id, field))
    else:
        for artifact in submission.ir_artifacts:
            error = _check_object(store, artifact.entity_id, "entity", observed=True)
            if error:
                violations.append(error)
                continue
            allowed = _ARTIFACT_TYPES.get(artifact.kind)
            if allowed:
                observed_types = {
                    v.fields["type"]
                    for v in store.history(artifact.entity_id, "entity")
                    if v.state == "observed"
                }
                if len(observed_types) > 1:
                    violations.append(_violation("conflicting_type", artifact.entity_id, "type"))
                elif not observed_types <= allowed:
                    violations.append(
                        _violation("artifact_type_mismatch", artifact.entity_id, "type")
                    )
    return violations
