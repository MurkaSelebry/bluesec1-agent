import asyncio
from types import SimpleNamespace
from typing import Any, cast

import pytest
from openai import AsyncOpenAI

from bluesec1_agent.llm import (
    OpenAIStructuredStepClient,
    parse_next_step_payload,
)
from bluesec1_agent.models import (
    NEXT_STEP_RESPONSE_FORMAT,
    BenignFinishInvestigationSubmission,
    FinishInvestigationEntityEvidence,
    FinishInvestigationRequest,
    GetADObjectInfoRequest,
    GetEntityRequest,
    NextStep,
    SearchRequest,
)


class FakeCompletions:
    """Queue-backed fake for the OpenAI chat completion endpoint."""

    def __init__(self, contents: list[str]) -> None:
        """Store response content and request audit state."""
        self.contents = list(contents)
        self.requests: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        """Return the next queued assistant message."""
        self.requests.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=self.contents.pop(0),
                        refusal=None,
                    )
                )
            ],
            usage=SimpleNamespace(total_tokens=17),
        )


class FakeOpenAI:
    """Minimal object graph required by `OpenAIStructuredStepClient`."""

    def __init__(self, contents: list[str]) -> None:
        """Expose fake chat completions."""
        self.completions = FakeCompletions(contents)
        self.chat = SimpleNamespace(completions=self.completions)


def valid_process_step() -> NextStep:
    """Build one valid non-terminal graph navigation step."""
    return NextStep(
        current_state="The trigger entity needs its relations.",
        plan_remaining_steps_brief=["Read the trigger entity."],
        task_completed=False,
        function=GetEntityRequest(
            tool="get_entity",
            entity_id="entity-1",
            reasoning="Inspect the alert entity.",
        ),
    )


def test_structured_client_repairs_malformed_response() -> None:
    """Malformed provider output should be retried outside the main history."""
    fake_openai = FakeOpenAI(["not-json", valid_process_step().model_dump_json()])
    client = OpenAIStructuredStepClient(
        base_url="https://llm.example/v1",
        api_key="secret",
        model="test-model",
        timeout_seconds=10.0,
        max_completion_tokens=1000,
        max_retries=3,
        client=cast(AsyncOpenAI, fake_openai),
    )
    original_messages = [{"role": "system", "content": "Investigate."}]

    step = asyncio.run(client.next_step(original_messages))

    assert step.function.tool == "get_entity"
    assert len(fake_openai.completions.requests) == 2
    first_messages = fake_openai.completions.requests[0]["messages"]
    assert "Never use top-level `action`" in first_messages[0]["content"]
    assert "advertised_tool" in first_messages[0]["content"]
    repaired_messages = fake_openai.completions.requests[1]["messages"]
    assert repaired_messages[-1]["role"] == "user"
    assert original_messages == [{"role": "system", "content": "Investigate."}]


def test_parser_rejects_inconsistent_completion_flag() -> None:
    """A non-terminal function cannot claim that the investigation is complete."""
    payload = valid_process_step().model_copy(update={"task_completed": True}).model_dump_json()

    with pytest.raises(ValueError, match="task_completed"):
        parse_next_step_payload(payload)


def test_response_schema_requires_nullable_tool_fields() -> None:
    """Strict provider schema should require optional fields as nullable values."""
    schema = NEXT_STEP_RESPONSE_FORMAT["json_schema"]["schema"]
    process_schema = schema["$defs"]["GetEntityRequest"]

    assert set(process_schema["required"]) == set(process_schema["properties"])
    assert "default" not in str(schema)
    assert "discriminator" not in str(schema)
    assert "oneOf" not in str(schema)
    assert set(schema["$defs"]["SearchRequest"]["required"]) == set(
        schema["$defs"]["SearchRequest"]["properties"]
    )
    assert set(schema["$defs"]["FinishInvestigationRequest"]["properties"]) == {
        "tool",
        "submission",
    }


def test_competition_search_serializes_extended_arguments() -> None:
    """Search should retain every argument supported by the competition tool."""
    request = SearchRequest.model_validate(
        {
            "tool": "search",
            "query": "powershell",
            "reasoning": "Find the related process.",
            "page_id": 2,
            "scope": {"kind": "entity", "type": "windows_process"},
            "time_window": {"from": "2026-09-01", "to": "2026-09-02"},
            "sort_order": "descending",
        }
    )

    arguments = request.model_dump(mode="json", by_alias=True, exclude={"tool"}, exclude_none=True)

    assert arguments["page_id"] == 2
    assert arguments["scope"] == {"kind": "entity", "type": "windows_process"}
    assert arguments["time_window"] == {
        "from": "2026-09-01T00:00:00Z",
        "to": "2026-09-02T23:59:59.999999Z",
    }
    assert arguments["sort_order"] == "descending"
    with pytest.raises(ValueError):
        SearchRequest.model_validate(
            {"tool": "search", "query": "process", "reasoning": "Inspect.", "page_id": 0}
        )
    with pytest.raises(ValueError):
        SearchRequest.model_validate(
            {
                "tool": "search",
                "query": "process",
                "reasoning": "Inspect.",
                "scope": {"kind": "entity", "type": "nonexistent"},
            }
        )
    with pytest.raises(ValueError):
        SearchRequest.model_validate(
            {
                "tool": "search",
                "query": "process",
                "reasoning": "Inspect.",
                "time_window": {"from": "2026-09-02", "to": "2026-09-01"},
            }
        )


def test_navigation_requires_reasoning_and_supports_ad_lookup() -> None:
    """Competition navigation tools should require a bounded rationale."""
    with pytest.raises(ValueError):
        GetEntityRequest.model_validate({"tool": "get_entity", "entity_id": "entity-1"})
    with pytest.raises(ValueError):
        GetADObjectInfoRequest.model_validate(
            {"tool": "get_ad_object_info", "ad_object_id": "ad-1", "reasoning": "x" * 4097}
        )

    request = GetADObjectInfoRequest(
        tool="get_ad_object_info", ad_object_id="ad-1", reasoning="Inspect the account."
    )
    assert request.model_dump(exclude={"tool"}) == {
        "ad_object_id": "ad-1",
        "reasoning": "Inspect the account.",
    }


def test_benign_finish_uses_submission_envelope() -> None:
    """Benign conclusions should contain only legitimacy evidence."""
    request = FinishInvestigationRequest(
        tool="finish_investigation",
        submission=BenignFinishInvestigationSubmission(
            verdict="benign",
            legitimacy_evidence=[
                FinishInvestigationEntityEvidence(
                    anchor="entity", entity_id="entity-1", property_fields=["command_line"]
                )
            ],
            reasoning="The process matches authorized activity.",
        ),
    )

    assert request.model_dump(exclude={"tool"}) == {
        "submission": {
            "verdict": "benign",
            "legitimacy_evidence": [
                {"anchor": "entity", "entity_id": "entity-1", "property_fields": ["command_line"]}
            ],
            "reasoning": "The process matches authorized activity.",
        }
    }
    step = NextStep(
        current_state="The alert is explained by authorized activity.",
        plan_remaining_steps_brief=["Submit benign evidence."],
        task_completed=True,
        function=request,
    )
    assert isinstance(
        parse_next_step_payload(step.model_dump_json()).function, FinishInvestigationRequest
    )
    with pytest.raises(ValueError):
        FinishInvestigationRequest.model_validate(
            {
                "tool": "finish_investigation",
                "submission": {"verdict": "benign", "ir_artifacts": [], "reasoning": "Safe."},
            }
        )


class CollectEvents:
    def __init__(self):
        self.events = []

    def record(self, event, **payload):
        self.events.append((event, payload))


def completion(content=None, *, finish_reason="stop", refusal=None, choices=True, usage=None):
    return SimpleNamespace(
        model="returned-model",
        usage=usage,
        choices=[
            SimpleNamespace(
                finish_reason=finish_reason,
                message=SimpleNamespace(content=content, refusal=refusal),
            )
        ]
        if choices
        else [],
    )


class FaultCompletions:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.requests = []

    async def create(self, **kwargs):
        self.requests.append(kwargs)
        reply = next(self.replies)
        if isinstance(reply, BaseException):
            raise reply
        if callable(reply):
            return await reply()
        return reply


def fault_client(replies, **options):
    endpoint = FaultCompletions(replies)
    sdk = SimpleNamespace(chat=SimpleNamespace(completions=endpoint))
    events = CollectEvents()
    client = OpenAIStructuredStepClient(
        base_url="https://llm.invalid/v1",
        api_key="synthetic-secret",
        model="test-model",
        timeout_seconds=1,
        max_completion_tokens=1000,
        max_retries=3,
        client=sdk,
        event_sink=events,
        **options,
    )
    return client, endpoint, events


@pytest.mark.parametrize(
    "bad, category",
    [
        (completion("broken synthetic-secret"), "schema"),
        (completion(""), "empty_response"),
        (completion(choices=False), "empty_response"),
        (completion(valid_process_step().model_dump_json(), finish_reason="length"), "truncated"),
        (completion('{"current_state":', finish_reason="length"), "truncated"),
    ],
)
def test_format_faults_are_bounded_and_observed_without_usage(bad, category):
    from bluesec1_agent.llm import NextStepRequestError

    client, endpoint, sink = fault_client([bad] * 5)
    with pytest.raises(NextStepRequestError) as error:
        asyncio.run(client.next_step([]))
    assert error.value.category == category
    assert len(endpoint.requests) == 3
    starts = [p for event, p in sink.events if event == "llm_attempt_started"]
    ends = [p for event, p in sink.events if event == "llm_attempt_finished"]
    assert len(starts) == len(ends) == 3
    assert [p["attempt_id"] for p in starts] == [p["attempt_id"] for p in ends]
    assert all(p["error_type"] == category for p in ends)
    assert all(p["total_tokens"] is None and p["prompt_tokens"] is None for p in ends)


def test_refusal_stops_without_retry_or_leaking_body():
    from bluesec1_agent.llm import NextStepRequestError

    client, endpoint, sink = fault_client([completion(refusal="synthetic-secret")])
    with pytest.raises(NextStepRequestError) as error:
        asyncio.run(client.next_step([]))
    assert len(endpoint.requests) == 1
    assert "synthetic-secret" not in str(error.value)
    assert error.value.category == "refusal"
    assert sink.events[-1][1]["error_type"] == "refusal"


@pytest.mark.parametrize("status, attempts", [(429, 3), (500, 3), (400, 1), (401, 1), (403, 1)])
def test_real_sdk_http_retries_share_three_attempt_cap_with_repair(status, attempts):
    import httpx

    from bluesec1_agent.llm import NextStepRequestError

    requests = []

    def handle(request):
        requests.append(request)
        if len(requests) == 2 and attempts == 3:
            return httpx.Response(
                200,
                json={
                    "id": "test",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "test",
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {"role": "assistant", "content": "bad"},
                        }
                    ],
                },
            )
        return httpx.Response(
            status, headers={"retry-after": "0"}, json={"error": {"message": "synthetic-secret"}}
        )

    async def exercise():
        async with AsyncOpenAI(
            api_key="synthetic-secret",
            base_url="https://test.invalid/v1",
            max_retries=9,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)),
        ) as sdk:
            sink = CollectEvents()
            client = OpenAIStructuredStepClient(
                base_url="https://test.invalid/v1",
                api_key="synthetic-secret",
                model="test",
                timeout_seconds=10,
                max_completion_tokens=100,
                max_retries=10,
                client=sdk,
                event_sink=sink,
            )
            with pytest.raises(NextStepRequestError) as error:
                await client.next_step([])
            assert error.value.category == f"http_{status}"
            assert len([e for e, _ in sink.events if e == "llm_attempt_finished"]) == attempts

    asyncio.run(exercise())
    assert len(requests) == attempts


def test_cancellation_finishes_attempt_and_propagates():
    async def exercise():
        entered = asyncio.Event()

        async def stalled():
            entered.set()
            await asyncio.Event().wait()

        client, endpoint, sink = fault_client([stalled])
        task = asyncio.create_task(client.next_step([]))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert len(endpoint.requests) == 1
        assert sink.events[-1][1]["outcome"] == "cancelled"

    asyncio.run(exercise())


def test_timeout_retry_deadline_and_missing_usage():
    from bluesec1_agent.budget import Budget, BudgetExceededError
    from bluesec1_agent.llm import response_total_tokens

    async def stalled():
        await asyncio.Event().wait()

    client, endpoint, sink = fault_client([stalled], budget=Budget(0.015))
    with pytest.raises(BudgetExceededError):
        asyncio.run(client.next_step([]))
    assert len(endpoint.requests) == 1
    assert sink.events[-1][1]["error_type"] == "timeout"
    assert response_total_tokens(completion()) is None


def test_truncated_valid_json_is_not_accepted_on_legacy_constructor():
    from bluesec1_agent.llm import NextStepRequestError

    endpoint = FaultCompletions(
        [completion(valid_process_step().model_dump_json(), finish_reason="length")] * 3
    )
    client = OpenAIStructuredStepClient(
        base_url="https://test.invalid/v1",
        api_key="test",
        model="test",
        timeout_seconds=1,
        max_completion_tokens=100,
        max_retries=3,
        client=SimpleNamespace(chat=SimpleNamespace(completions=endpoint)),
    )
    with pytest.raises(NextStepRequestError):
        asyncio.run(client.next_step([]))


def test_missing_usage_is_unknown_on_legacy_helper():
    from bluesec1_agent.llm import response_total_tokens

    assert response_total_tokens(completion()) is None


def test_malformed_content_is_never_logged():
    from loguru import logger

    from bluesec1_agent.llm import NextStepRequestError

    logged = []
    handle = logger.add(lambda message: logged.append(str(message)))
    try:
        client = OpenAIStructuredStepClient(
            base_url="https://test.invalid/v1",
            api_key="test",
            model="test",
            timeout_seconds=1,
            max_completion_tokens=100,
            max_retries=1,
            client=FakeOpenAI(["malformed synthetic-secret"]),
        )
        with pytest.raises(NextStepRequestError):
            asyncio.run(client.next_step([]))
    finally:
        logger.remove(handle)
    assert "synthetic-secret" not in "".join(logged)


def test_usage_details_are_reported_without_double_counting_and_with_correlation():
    from bluesec1_agent.recording import SafeEventSink

    usage = SimpleNamespace(
        prompt_tokens=100,
        completion_tokens=20,
        total_tokens=120,
        prompt_tokens_details=SimpleNamespace(cached_tokens=50),
        completion_tokens_details=SimpleNamespace(reasoning_tokens=10),
    )
    collector = CollectEvents()
    sink = SafeEventSink(collector)
    sink.update_context(run_id="run-1", task_id="task-1", step=4)
    endpoint = FaultCompletions([completion(valid_process_step().model_dump_json(), usage=usage)])
    client = OpenAIStructuredStepClient(
        base_url="https://test.invalid",
        api_key="test",
        model="test",
        timeout_seconds=1,
        max_completion_tokens=100,
        max_retries=3,
        client=SimpleNamespace(chat=SimpleNamespace(completions=endpoint)),
        event_sink=sink,
    )
    asyncio.run(client.next_step([]))
    event, payload = collector.events[-1]
    assert event == "llm_attempt_finished"
    assert {key: payload[key] for key in ("run_id", "task_id", "step")} == {
        "run_id": "run-1",
        "task_id": "task-1",
        "step": 4,
    }
    assert payload["total_tokens"] == 120
    assert payload["cached_tokens"] == 50 and payload["reasoning_tokens"] == 10


def test_retry_after_cannot_extend_task_budget():
    import httpx
    from openai import RateLimitError

    from bluesec1_agent.budget import Budget, BudgetExceededError

    error = RateLimitError(
        "synthetic-secret",
        body=None,
        response=httpx.Response(
            429,
            headers={"retry-after": "600"},
            request=httpx.Request("POST", "https://test.invalid"),
        ),
    )
    client, endpoint, sink = fault_client([error], budget=Budget(0.5))
    with pytest.raises(BudgetExceededError):
        asyncio.run(client.next_step([]))
    assert len(endpoint.requests) == 1
    assert sink.events[-1][1]["error_type"] == "http_429"


def test_recording_failure_does_not_replace_cancellation(tmp_path):
    from bluesec1_agent.recording import RunRecorder

    async def exercise():
        entered = asyncio.Event()

        async def stalled():
            entered.set()
            await asyncio.Event().wait()

        recorder = RunRecorder(tmp_path)
        (recorder.directory / "trace.jsonl").mkdir()
        endpoint = FaultCompletions([stalled])
        client = OpenAIStructuredStepClient(
            base_url="https://test.invalid",
            api_key="test",
            model="test",
            timeout_seconds=1,
            max_completion_tokens=100,
            max_retries=3,
            client=SimpleNamespace(chat=SimpleNamespace(completions=endpoint)),
            event_sink=recorder,
        )
        task = asyncio.create_task(client.next_step([]))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(exercise())


@pytest.mark.parametrize("kind", ["timeout", "connection"])
def test_transient_transport_faults_retry_and_count_every_attempt(kind):
    import httpx
    from openai import APIConnectionError, APITimeoutError

    request = httpx.Request("POST", "https://test.invalid")
    error = (
        APITimeoutError(request=request)
        if kind == "timeout"
        else APIConnectionError(request=request)
    )
    client, endpoint, sink = fault_client(
        [error, error, completion(valid_process_step().model_dump_json())]
    )
    result = asyncio.run(client.next_step([]))
    assert result.function.tool == "get_entity"
    assert len(endpoint.requests) == 3
    finished = [p for event, p in sink.events if event == "llm_attempt_finished"]
    assert [p["outcome"] for p in finished] == ["failed", "failed", "success"]
    assert [p["error_type"] for p in finished] == [
        "timeout" if kind == "timeout" else "transport",
        "timeout" if kind == "timeout" else "transport",
        None,
    ]


def test_http_timeout_is_capped_and_expired_budget_starts_no_attempt(monkeypatch):
    from bluesec1_agent import budget as module

    now = [100.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: now[0])
    budget = module.Budget(150)
    endpoint = FaultCompletions([completion(valid_process_step().model_dump_json())])
    client = OpenAIStructuredStepClient(
        base_url="https://test.invalid",
        api_key="test",
        model="test",
        timeout_seconds=999,
        max_completion_tokens=100,
        max_retries=3,
        client=SimpleNamespace(chat=SimpleNamespace(completions=endpoint)),
        budget=budget,
    )
    asyncio.run(client.next_step([]))
    assert endpoint.requests[0]["timeout"] == 120
    now[0] = 251
    with pytest.raises(module.BudgetExceededError):
        asyncio.run(client.next_step([]))
    assert len(endpoint.requests) == 1


def test_request_error_category_preserves_legacy_message_constructor():
    from bluesec1_agent.llm import NextStepRequestError

    error = NextStepRequestError("legacy diagnostic")
    assert str(error) == "legacy diagnostic"
    assert error.args == ("legacy diagnostic",)
    assert error.category == "provider_error"
