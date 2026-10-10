"""Deterministic navigation priorities from known graph topology, never ID text."""

from bluesec1_agent.contracts import Lead
from bluesec1_agent.evidence import EvidenceStore

_BROAD_TYPES = {"host", "windows_user", "linux_user"}
_PROCESS_TYPES = {"windows_process", "linux_process"}


def navigation_groups(store: EvidenceStore, *, trigger_ids: tuple[str, ...]) -> dict[str, set[str]]:
    """Topology scopes for prioritization and typed coverage, never verdict labels.

    Relation groups can overlap: a shared host/process alert link belongs to
    each applicable group. Only links from read non-broad neighbors extend the
    active frontier; reading a host/user does not promote its entire adjacency.
    """
    groups: dict[str, set[str]] = {
        key: set()
        for key in ("direct", "focused", "process", "broad", "alert", "neighbors", "active")
    }
    for tid in trigger_ids:
        if store.is_known(tid, "relation"):
            groups["alert"].add(tid)
        ref = store.get(tid, "entity")
        if ref is None:
            continue
        relations = set(store.relation_ids_for(tid))
        groups["direct"].update(relations)
        object_type = ref.fields.get("type")
        groups["broad" if object_type in _BROAD_TYPES else "focused"].update(relations)
        if object_type in _PROCESS_TYPES:
            groups["process"].update(relations)
    for rid in groups["focused"] | groups["alert"]:
        for version in store.history(rid, "relation"):
            for key in ("source_id", "target_id"):
                endpoint = version.fields.get(key)
                if isinstance(endpoint, str):
                    groups["neighbors"].add(endpoint)
    for endpoint in groups["neighbors"] - set(trigger_ids):
        ref = store.get(endpoint, "entity")
        if ref and ref.state == "observed" and ref.fields.get("type") not in _BROAD_TYPES:
            groups["active"].update(store.relation_ids_for(endpoint))
    groups["active"].difference_update(groups["direct"] | groups["alert"])
    return groups


def rank_frontier(store: EvidenceStore, *, trigger_ids: tuple[str, ...]) -> list[Lead]:
    """Return each unread object once, with lower numeric priorities first.

    Read triggers first, then their direct unknown links, then neighbors and
    links of read non-broad neighbors. Broad host/user enumeration ranks later.
    This is prioritization, not a requirement to exhaust the graph before finish.
    """
    triggers = set(trigger_ids)
    groups = navigation_groups(store, trigger_ids=trigger_ids)
    leads = []
    for ref in store.snapshot(max_items=len(store)).items:
        if ref.state == "observed":
            continue
        if ref.object_id in triggers:
            priority, reason = (0 if ref.kind == "entity" else 5), "Unread alert trigger"
        elif ref.kind == "relation" and ref.object_id in groups["focused"]:
            priority, reason = 10, "Unread direct trigger relation; type requires reading"
        elif ref.kind == "entity" and ref.object_id in groups["neighbors"]:
            priority, reason = 20, "Endpoint of a known trigger relation"
        elif ref.kind == "relation" and ref.object_id in groups["active"]:
            priority, reason = 25, "Unread relation of a read non-broad trigger neighbor"
        elif ref.kind == "relation" and ref.object_id in groups["broad"]:
            priority, reason = 30, "Unread trigger host/user relation; broad enumeration"
        else:
            priority, reason = 50, "Other referenced object; avoid exhaustive broad enumeration"
        leads.append(Lead(ref.object_id, ref.kind, priority, reason))
    return sorted(leads, key=lambda lead: (lead.priority, lead.kind, lead.object_id))
