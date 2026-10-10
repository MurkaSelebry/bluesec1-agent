"""Deterministic navigation priorities from known graph topology, never ID text."""

from bluesec1_agent.contracts import Lead
from bluesec1_agent.evidence import EvidenceStore


def rank_frontier(store: EvidenceStore, *, trigger_ids: tuple[str, ...]) -> list[Lead]:
    """Return each unread object once, with lower numeric priorities first.

    Read triggers first, then their direct unknown links, then neighbors of read
    trigger links. Broad host/user enumeration ranks after direct trigger links.
    This is prioritization, not a requirement to exhaust the graph before finish.
    """
    triggers = set(trigger_ids)
    direct_relations = {rid for tid in triggers for rid in store.relation_ids_for(tid)}
    broad_types = {"host", "windows_user", "linux_user"}
    focused_relations = {
        rid
        for tid in triggers
        if (ref := store.get(tid, "entity")) is not None
        and ref.fields.get("type") not in broad_types
        for rid in store.relation_ids_for(tid)
    }
    trigger_neighbors: set[str] = set()
    for rid in direct_relations | triggers:
        for version in store.history(rid, "relation"):
            for key in ("source_id", "target_id"):
                endpoint = version.fields.get(key)
                if isinstance(endpoint, str):
                    trigger_neighbors.add(endpoint)
    leads = []
    for ref in store.snapshot(max_items=len(store)).items:
        if ref.state == "observed":
            continue
        if ref.object_id in triggers:
            priority, reason = (0 if ref.kind == "entity" else 5), "Unread alert trigger"
        elif ref.kind == "relation" and ref.object_id in focused_relations:
            priority, reason = 10, "Unread direct trigger relation; type requires reading"
        elif ref.kind == "relation" and ref.object_id in direct_relations:
            priority, reason = 30, "Unread trigger host/user relation; broad enumeration"
        elif ref.kind == "entity" and ref.object_id in trigger_neighbors:
            priority, reason = 20, "Endpoint of a known trigger relation"
        else:
            priority, reason = 50, "Other referenced object; avoid exhaustive broad enumeration"
        leads.append(Lead(ref.object_id, ref.kind, priority, reason))
    return sorted(leads, key=lambda lead: (lead.priority, lead.kind, lead.object_id))
