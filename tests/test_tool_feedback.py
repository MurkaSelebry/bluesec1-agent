import pytest

from bluesec1_agent.agent import SimpleSGRAgent
from bluesec1_client import ToolCallResult


def test_tool_feedback_prefers_legacy_info_result_over_repeated_observation() -> None:
    response = ToolCallResult(
        task_id="task-17",
        tool_name="get_entity",
        observation={
            "alert_text": '{"trigger_entities":["alert-1"]}',
            "available_tools": '[{"name":"get_entity"}]',
            "last_tool_result": '{"entity_id":"new-shape"}',
        },
        reward=0.25,
        terminated=False,
        truncated=False,
        info={
            "tool_name": "get_entity",
            "latency_ms": 18,
            "episode_id": "episode-4",
            "step_index": 2,
            "tool_args": {"entity_id": "process-1"},
            "tool_status": "ok",
            "tool_result": {"entity_id": "legacy-shape", "pid": 42},
        },
        task_result=None,
    )

    assert SimpleSGRAgent._tool_feedback(response) == {
        "tool_name": "get_entity",
        "status": "ok",
        "result": {"entity_id": "legacy-shape", "pid": 42},
        "error": None,
        "reward": 0.25,
        "terminated": False,
        "truncated": False,
    }


@pytest.mark.parametrize(
    ("serialized_result", "expected_result"),
    [
        ('{"entity_id":"process-1","pid":42}', {"entity_id": "process-1", "pid": 42}),
        ('[{"entity_id":"process-1"}]', [{"entity_id": "process-1"}]),
        ('"{\\"nested\\":true}"', '{"nested":true}'),
        ("null", None),
        ("0", 0),
        ("[]", []),
        ("opaque runtime text", "opaque runtime text"),
    ],
)
def test_tool_feedback_decodes_last_tool_result_once(
    serialized_result: str,
    expected_result: object,
) -> None:
    response = ToolCallResult(
        task_id="task-17",
        tool_name="get_entity",
        observation={
            "alert_text": '{"trigger_entities":["alert-1"]}',
            "available_tools": '[{"name":"get_entity"}]',
            "last_tool_result": serialized_result,
        },
        reward=0.0,
        terminated=False,
        truncated=False,
        info={
            "tool_name": "get_entity",
            "latency_ms": 18,
            "episode_id": "episode-4",
            "step_index": 2,
            "tool_args": {"entity_id": "process-1"},
            "tool_status": "ok",
        },
        task_result=None,
    )

    feedback = SimpleSGRAgent._tool_feedback(response)

    assert feedback["result"] == expected_result
    assert feedback["error"] is None


@pytest.mark.parametrize("legacy_result", [0, [], ""])
def test_tool_feedback_preserves_falsey_legacy_results(legacy_result: object) -> None:
    response = ToolCallResult(
        task_id="task-17",
        tool_name="search_entities",
        observation={
            "alert_text": '{"trigger_entities":["alert-1"]}',
            "available_tools": '[{"name":"search_entities"}]',
            "last_tool_result": '{"unexpected":"fallback"}',
        },
        reward=0.0,
        terminated=False,
        truncated=False,
        info={
            "tool_name": "search_entities",
            "latency_ms": 7,
            "episode_id": "episode-4",
            "step_index": 3,
            "tool_args": {"query": "missing"},
            "tool_status": "ok",
            "tool_result": legacy_result,
        },
        task_result=None,
    )

    assert SimpleSGRAgent._tool_feedback(response)["result"] == legacy_result


def test_tool_feedback_keeps_error_evidence_and_exposes_its_message() -> None:
    response = ToolCallResult(
        task_id="task-17",
        tool_name="get_entity",
        observation={
            "alert_text": '{"trigger_entities":["alert-1"]}',
            "available_tools": '[{"name":"get_entity"}]',
            "last_tool_result": (
                '{"message":"Entity process-404 was not found","code":"not_found"}'
            ),
        },
        reward=-0.1,
        terminated=False,
        truncated=False,
        info={
            "tool_name": "get_entity",
            "latency_ms": 12,
            "episode_id": "episode-4",
            "step_index": 4,
            "tool_args": {"entity_id": "process-404"},
            "tool_status": "error",
        },
        task_result=None,
    )

    assert SimpleSGRAgent._tool_feedback(response) == {
        "tool_name": "get_entity",
        "status": "error",
        "result": {
            "message": "Entity process-404 was not found",
            "code": "not_found",
        },
        "error": "Entity process-404 was not found",
        "reward": -0.1,
        "terminated": False,
        "truncated": False,
    }


def test_tool_feedback_fallback_removes_only_repeated_observation_fields() -> None:
    response = ToolCallResult(
        task_id="task-17",
        tool_name="inspect_relation",
        observation={
            "alert_text": '{"trigger_entities":["alert-1"]}',
            "available_tools": '[{"name":"inspect_relation"}]',
            "relation": {"source": "process-1", "target": "file-9"},
            "cursor": 0,
            "matches": [],
            "diagnostic": None,
        },
        reward=0.5,
        terminated=False,
        truncated=False,
        info={
            "tool_name": "inspect_relation",
            "latency_ms": 25,
            "episode_id": "episode-4",
            "step_index": 5,
            "tool_args": {"relation_id": "relation-2"},
            "tool_status": "ok",
        },
        task_result=None,
    )

    assert SimpleSGRAgent._tool_feedback(response)["result"] == {
        "relation": {"source": "process-1", "target": "file-9"},
        "cursor": 0,
        "matches": [],
        "diagnostic": None,
    }
