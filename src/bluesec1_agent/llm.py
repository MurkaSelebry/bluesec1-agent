import asyncio
import json
import math
import random
import re
import time
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from types import TracebackType
from typing import Any, Protocol, Self, cast
from uuid import uuid4

from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from openai.types.shared_params import ResponseFormatJSONSchema
from pydantic import ValidationError

from bluesec1_agent.budget import Budget, BudgetExceededError
from bluesec1_agent.contracts import EventSink
from bluesec1_agent.models import (
    NEXT_STEP_RESPONSE_FORMAT,
    FinishInvestigationRequest,
    NextStep,
)
from bluesec1_agent.recording import SafeEventSink

STRUCTURED_OUTPUT_INSTRUCTION = """
Every assistant reply must be exactly one JSON object with these top-level fields:
{
  "current_state": "short evidence-focused string",
  "plan_remaining_steps_brief": ["one to five short step strings"],
  "task_completed": false,
  "function": {
    "tool": "one advertised tool name",
    "...": "that tool's arguments placed directly in this function object"
  }
}

Never use top-level `action`, `parameters`, `tool_name`, or `arguments` fields.
Never wrap tool arguments in a nested `parameters` or `arguments` object.
For example, a call with one advertised argument is:
{"tool":"advertised_tool","argument_name":"value"}
Set task_completed=true only for a function whose tool is finish_investigation.
Return no prose, markdown, XML, comments, or keys outside the required object.
""".strip()

REPAIR_INSTRUCTION = (
    "Your previous reply violated the required NextStep contract. " + STRUCTURED_OUTPUT_INSTRUCTION
)


class NextStepRequestError(RuntimeError):
    """Raised when an LLM cannot produce a valid structured step."""


class StructuredStepClient(Protocol):
    """Narrow async contract consumed by the investigation loop."""

    async def next_step(self, messages: list[dict[str, Any]]) -> NextStep:
        """Generate and validate one investigation step."""
        ...


class OpenAIStructuredStepClient:
    """Portable structured-output client for OpenAI-compatible providers."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout_seconds: float,
        max_completion_tokens: int,
        max_retries: int,
        client: AsyncOpenAI | None = None,
        event_sink: EventSink | None = None,
        budget: Budget | None = None,
    ) -> None:
        """Initialize an OpenAI-compatible structured step generator.

        Args:
            base_url: OpenAI-compatible API root.
            api_key: Provider credential.
            model: Provider model identifier.
            timeout_seconds: Timeout applied to each provider request.
            max_completion_tokens: Maximum generated tokens per attempt.
            max_retries: Total HTTP attempts per decision, including repairs, capped at three.
            client: Optional injected SDK client used by tests.
        """
        self.model = model
        self.max_completion_tokens = max_completion_tokens
        if max_retries < 1:
            raise ValueError("max_retries must allow at least one attempt")
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be finite and positive")
        self.max_retries = min(max_retries, 3)
        self.timeout_seconds = min(timeout_seconds, 120.0)
        self.budget = budget
        self._sink = (
            event_sink
            if isinstance(event_sink, SafeEventSink)
            else SafeEventSink(event_sink, secrets=(api_key,))
        )
        self._owns_client = client is None
        self._client = client or AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout_seconds,
            max_retries=0,
        )
        if isinstance(self._client, AsyncOpenAI):
            self._client = self._client.with_options(max_retries=0)

    async def __aenter__(self) -> Self:
        """Enter the async LLM client context."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close an internally owned provider client."""
        await self.aclose()

    async def aclose(self) -> None:
        """Close the provider transport created by this wrapper."""
        if self._owns_client:
            await self._client.close()

    async def next_step(self, messages: list[dict[str, Any]]) -> NextStep:
        """Return one valid `NextStep`, repairing malformed replies when possible.

        Args:
            messages: Current investigation conversation.

        Returns:
            A locally validated structured investigation step.

        Raises:
            NextStepRequestError: If every provider reply violates the schema.
        """
        request_messages = list(messages)
        if request_messages and request_messages[0].get("role") == "system":
            first_message = dict(request_messages[0])
            first_message["content"] = (
                f"{first_message.get('content', '')}\n\n{STRUCTURED_OUTPUT_INSTRUCTION}"
            )
            request_messages[0] = first_message
        else:
            request_messages.insert(
                0,
                {"role": "system", "content": STRUCTURED_OUTPUT_INSTRUCTION},
            )
        budget = self.budget or Budget(self.timeout_seconds * self.max_retries)
        last_category = "schema"
        for attempt in range(1, self.max_retries + 1):
            budget.check()
            attempt_id = uuid4().hex
            started_at = time.monotonic()
            response = None
            category = None
            outcome = "failed"
            retryable = False
            delay = 0.0
            raw_content = ""
            self._sink.record(
                "llm_attempt_started",
                attempt_id=attempt_id,
                attempt=attempt,
                requested_model=self.model,
            )
            try:
                timeout = min(self.timeout_seconds, budget.remaining_seconds())
                async with asyncio.timeout(timeout):
                    response = await self._client.chat.completions.create(
                        model=self.model,
                        messages=cast(list[ChatCompletionMessageParam], list(request_messages)),
                        response_format=cast(ResponseFormatJSONSchema, NEXT_STEP_RESPONSE_FORMAT),
                        max_completion_tokens=self.max_completion_tokens,
                        timeout=timeout,
                    )
                choices = getattr(response, "choices", None)
                if not choices or getattr(choices[0], "message", None) is None:
                    raise _ResponseFault("empty_response")
                choice = choices[0]
                if getattr(choice.message, "refusal", None):
                    raise _ResponseFault("refusal", retryable=False)
                if getattr(choice, "finish_reason", None) == "content_filter":
                    raise _ResponseFault("refusal", retryable=False)
                if getattr(choice, "finish_reason", None) == "length":
                    raise _ResponseFault("truncated")
                raw_content = extract_message_text(choice.message)
                if not raw_content:
                    raise _ResponseFault("empty_response")
                try:
                    step = parse_next_step_payload(raw_content)
                except ValueError:
                    raise _ResponseFault("schema") from None
                outcome = "success"
                return step
            except asyncio.CancelledError:
                outcome, category = "cancelled", "cancelled"
                raise
            except _ResponseFault as exc:
                category, retryable = exc.category, exc.retryable
                if retryable and attempt < self.max_retries:
                    self._sink.record("schema_repair", attempt_id=attempt_id, error_type=category)
                    request_messages.extend(
                        [
                            {"role": "assistant", "content": raw_content or "<unusable reply>"},
                            {"role": "user", "content": REPAIR_INSTRUCTION},
                        ]
                    )
            except (TimeoutError, APITimeoutError):
                category, retryable = "timeout", True
                delay = _backoff(attempt)
            except APIConnectionError:
                category, retryable = "transport", True
                delay = _backoff(attempt)
            except APIStatusError as exc:
                category = f"http_{exc.status_code}"
                retryable = exc.status_code == 429 or exc.status_code >= 500
                delay = _retry_delay(exc, attempt)
            except Exception:
                category = "provider_error"
                # Never forward provider text or request objects to console error paths.
            finally:
                self._sink.record(
                    "llm_attempt_finished",
                    attempt_id=attempt_id,
                    attempt=attempt,
                    requested_model=self.model,
                    returned_model=getattr(response, "model", None),
                    latency_ms=int((time.monotonic() - started_at) * 1000),
                    outcome=outcome,
                    error_type=category,
                    **response_usage(response),
                    finish_reason=_finish_reason(response),
                )
            last_category = category or "provider_error"
            if not retryable or attempt == self.max_retries:
                break
            budget.check()
            if delay >= budget.remaining_seconds():
                raise BudgetExceededError("Insufficient task time for the next provider attempt.")
            if delay:
                await asyncio.sleep(delay)
        raise NextStepRequestError(f"Structured decision failed ({last_category}).")


class _ResponseFault(ValueError):
    def __init__(self, category: str, *, retryable: bool = True) -> None:
        super().__init__(category)
        self.category = category
        self.retryable = retryable


def _backoff(attempt: int) -> float:
    return random.uniform(0.0, min(2.0, 0.25 * 2 ** (attempt - 1)))


def _retry_delay(error: APIStatusError, attempt: int) -> float:
    value = error.response.headers.get("retry-after")
    if value:
        try:
            delay = float(value)
        except ValueError:
            try:
                date = parsedate_to_datetime(value)
                delay = (date - datetime.now(UTC)).total_seconds()
            except (ValueError, TypeError, OverflowError):
                delay = -1
        if math.isfinite(delay) and delay >= 0:
            return delay
    return _backoff(attempt)


def _finish_reason(response: Any) -> str | None:
    choices = getattr(response, "choices", None)
    return getattr(choices[0], "finish_reason", None) if choices else None


def _token_count(value: Any) -> int | None:
    return value if type(value) is int and value >= 0 else None


def response_usage(response: Any) -> dict[str, int | None]:
    usage = getattr(response, "usage", None)
    return {
        "prompt_tokens": _token_count(getattr(usage, "prompt_tokens", None)),
        "completion_tokens": _token_count(getattr(usage, "completion_tokens", None)),
        "total_tokens": _token_count(getattr(usage, "total_tokens", None)),
        "cached_tokens": _token_count(
            getattr(getattr(usage, "prompt_tokens_details", None), "cached_tokens", None)
        ),
        "reasoning_tokens": _token_count(
            getattr(getattr(usage, "completion_tokens_details", None), "reasoning_tokens", None)
        ),
    }


def extract_message_text(message: Any) -> str:
    """Normalize string and multipart assistant content into plain text.

    Args:
        message: Provider-specific chat completion message.

    Returns:
        Concatenated textual message content.
    """
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""

    text_parts: list[str] = []
    for part in content:
        if isinstance(part, dict) and part.get("type") == "text":
            text_parts.append(str(part.get("text", "")))
        elif getattr(part, "type", None) == "text":
            text_parts.append(str(getattr(part, "text", "")))
    return "".join(text_parts)


def extract_raw_content(response: Any) -> str:
    """Extract the first assistant message from a completion response.

    Args:
        response: OpenAI-compatible chat completion response.

    Returns:
        Raw assistant text.

    Raises:
        ValueError: If the provider response is missing a usable message.
    """
    choices = getattr(response, "choices", None)
    if not choices:
        raise ValueError("Model response is missing choices.")

    message = getattr(choices[0], "message", None)
    if message is None:
        raise ValueError("Model response is missing a message.")

    refusal = getattr(message, "refusal", None)
    if refusal:
        raise ValueError("Model refused structured output.")
    return extract_message_text(message)


def extract_json_object(raw_content: str) -> str:
    """Remove common provider wrappers around a JSON object.

    Args:
        raw_content: Untrusted assistant response text.

    Returns:
        Candidate JSON object text for local validation.
    """
    stripped = raw_content.strip()
    if not stripped:
        return ""
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped).strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped

    match = re.search(r"\{.*\}", stripped, re.DOTALL)
    return match.group(0) if match is not None else stripped


def parse_next_step_payload(raw_content: str) -> NextStep:
    """Validate one raw provider reply as a consistent `NextStep`.

    Args:
        raw_content: Assistant response containing a structured step.

    Returns:
        Parsed and validated step.

    Raises:
        ValueError: If JSON, schema, or completion semantics are invalid.
    """
    if not raw_content:
        raise ValueError("Model returned empty content for NextStep response.")

    normalized_content = extract_json_object(raw_content)
    try:
        step = NextStep.model_validate_json(normalized_content)
    except ValidationError as exc:
        raise ValueError("Model returned invalid NextStep JSON.") from exc

    is_finish = isinstance(step.function, FinishInvestigationRequest)
    if step.task_completed != is_finish:
        raise ValueError(
            "task_completed must be true exactly when finish_investigation is selected."
        )
    return step


def response_total_tokens(response: Any) -> int | None:
    """Return reported usage; absence is unknown, never an invented zero."""
    return response_usage(response)["total_tokens"]


def render_json(value: Any) -> str:
    """Serialize participant-visible context for an LLM message.

    Args:
        value: Public JSON-compatible value.

    Returns:
        Stable readable JSON text.
    """
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str)
