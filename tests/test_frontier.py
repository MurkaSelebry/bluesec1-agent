import json
from pathlib import Path

from bluesec1_agent.evidence import EvidenceStore
from bluesec1_agent.frontier import rank_frontier


def graph():
    fixture = json.loads(
        (Path(__file__).parent / "fixtures/investigation/trigger_graph.json").read_text()
    )
    store = EvidenceStore()
    store.seed_observation(fixture["observation"])
    for read in fixture["reads"]:
        store.ingest(**read)
    return store


def test_direct_trigger_links_precede_bulk_user_enumeration_deterministically():
    store = graph()
    frontier = rank_frontier(store, trigger_ids=store.trigger_ids)
    assert [lead.object_id for lead in frontier] == ["a-unread", "z-unread", "bulk-1", "bulk-2"]
    assert frontier[0].priority < frontier[2].priority
    assert all(lead.kind == "relation" for lead in frontier)
    assert all("injected_into" not in lead.reason for lead in frontier)
    assert rank_frontier(store, trigger_ids=store.trigger_ids) == frontier


def test_unread_trigger_entity_precedes_indirect_links():
    store = graph()
    store.seed_observation({"alert_text": {"trigger_entities": [{"id": "new-trigger"}]}})
    assert rank_frontier(store, trigger_ids=store.trigger_ids)[0].object_id == "new-trigger"


def test_cycles_and_duplicate_ids_produce_one_pending_lead():
    store = graph()
    store.ingest(
        step=4,
        tool_name="get_relation",
        arguments={"relation_id": "a-unread"},
        status="ok",
        result={
            "id": "a-unread",
            "type": "parent_of",
            "source_id": "trigger",
            "target_id": "trigger",
        },
    )
    ids = [lead.object_id for lead in rank_frontier(store, trigger_ids=store.trigger_ids)]
    assert ids == ["z-unread", "bulk-1", "bulk-2"]


def test_read_type_never_guessed_from_opaque_relation_id():
    store = EvidenceStore()
    store.ingest(
        step=1,
        tool_name="get_entity",
        arguments={"entity_id": "p"},
        status="ok",
        result={
            "id": "p",
            "type": "windows_process",
            "properties": {},
            "relation_ids": {"outgoing": ["injected_into-looking-id", "aaa"]},
        },
    )
    frontier = rank_frontier(store, trigger_ids=("p",))
    assert frontier[0].object_id == "aaa"
    assert frontier[0].priority == frontier[1].priority
    assert store.get("injected_into-looking-id", "relation").fields == {}


def test_trigger_process_relations_precede_bulk_trigger_host_relations():
    store = EvidenceStore()
    store.seed_observation(
        {
            "alert_text": {
                "trigger_entities": [
                    {"id": "host", "type": "host"},
                    {"id": "process", "type": "windows_process"},
                ]
            }
        }
    )
    for object_id, object_type, relations in [
        ("host", "host", ["aaa-host-link"]),
        ("process", "windows_process", ["zzz-process-link"]),
    ]:
        store.ingest(
            step=1,
            tool_name="get_entity",
            arguments={"entity_id": object_id},
            status="ok",
            result={
                "id": object_id,
                "type": object_type,
                "properties": {},
                "relation_ids": {"outgoing": relations},
            },
        )
    leads = rank_frontier(store, trigger_ids=store.trigger_ids)
    assert [lead.object_id for lead in leads] == ["zzz-process-link", "aaa-host-link"]
    assert leads[0].priority < leads[1].priority


def test_child_process_leads_precede_host_expansion_without_promoting_user_enumeration():
    store = graph()
    store.seed_observation({"alert_text": {"trigger_entities": [{"id": "host", "type": "host"}]}})
    for object_id, object_type, outgoing in [
        ("host", "host", ["aaa-host-link"]),
        ("child", "windows_process", ["zzz-child-link"]),
    ]:
        store.ingest(
            step=4,
            tool_name="get_entity",
            arguments={"entity_id": object_id},
            status="ok",
            result={
                "id": object_id,
                "type": object_type,
                "properties": {},
                "relation_ids": {"outgoing": outgoing},
            },
        )
    store.ingest(
        step=5,
        tool_name="get_relation",
        arguments={"relation_id": "a-unread"},
        status="ok",
        result={
            "id": "a-unread",
            "type": "parent_of",
            "source_id": "trigger",
            "target_id": "child",
        },
    )
    ids = [lead.object_id for lead in rank_frontier(store, trigger_ids=store.trigger_ids)]
    assert ids == ["z-unread", "zzz-child-link", "aaa-host-link", "bulk-1", "bulk-2"]
