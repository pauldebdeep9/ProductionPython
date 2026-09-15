"""Topic 5: Structured logging and observability.

Structured, correlated events show what happened; metrics show how often; traces
show where time went. This standard-library tutorial uses no telemetry backend.
"""

import asyncio
import json
import logging
import sys
import time
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import UTC, datetime


REQUEST_ID: ContextVar[str] = ContextVar("request_id", default="unscoped")
LOGGER = logging.getLogger("production_core.observability")
HANDLER_NAME = "production-core-json"

# Arbitrary LogRecord attributes and raw payloads are deliberately excluded.
STRUCTURED_FIELDS = (
    "operation",
    "status",
    "latency_ms",
    "documents_retrieved",
    "model",
    "provider",
    "input_tokens",
    "output_tokens",
    "prompt_chars",
    "api_key_configured",
    "error_type",
)


class RequestContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        # ContextVar is task-local, unlike one mutable global request ID.
        setattr(record, "request_id", REQUEST_ID.get())
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.fromtimestamp(record.created, tz=UTC).isoformat(
            timespec="milliseconds"
        )
        payload: dict[str, object] = {
            "timestamp": timestamp,
            "level": record.levelname,
            "event": record.getMessage(),
            "request_id": getattr(record, "request_id", "unscoped"),
        }
        for field_name in STRUCTURED_FIELDS:
            if hasattr(record, field_name):
                payload[field_name] = getattr(record, field_name)
        return json.dumps(payload, separators=(",", ":"))


def configure_logging() -> logging.Logger:
    LOGGER.setLevel(logging.DEBUG)
    LOGGER.propagate = False
    if not any(handler.get_name() == HANDLER_NAME for handler in LOGGER.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler.set_name(HANDLER_NAME)
        handler.setFormatter(JsonFormatter())
        handler.addFilter(RequestContextFilter())
        LOGGER.addHandler(handler)
    return LOGGER


@dataclass
class Metrics:
    requests_total: int = 0
    requests_failed: int = 0
    llm_calls_total: int = 0
    llm_errors_total: int = 0


@dataclass(frozen=True)
class Span:
    name: str
    duration_ms: float


@dataclass
class RequestObservation:
    request_id: str
    status: str
    latency_ms: float
    documents_retrieved: int
    spans: list[Span]


class FakeModelError(Exception):
    pass


def milliseconds_since(started: float) -> float:
    return round((time.perf_counter() - started) * 1_000, 2)


async def retrieve(
    documents_to_return: int, metrics: Metrics
) -> tuple[list[str], float]:
    LOGGER.debug("retrieval_started", extra={"operation": "retrieve"})
    started = time.perf_counter()
    await asyncio.sleep(0.02)
    documents = [f"document-{number}" for number in range(1, documents_to_return + 1)]
    latency_ms = milliseconds_since(started)
    LOGGER.info(
        "retrieval_completed",
        extra={
            "operation": "retrieve",
            "status": "success",
            "documents_retrieved": len(documents),
            "latency_ms": latency_ms,
        },
    )
    if len(documents) < 2:
        LOGGER.warning(
            "retrieval_low_coverage",
            extra={
                "operation": "retrieve",
                "status": "degraded",
                "documents_retrieved": len(documents),
            },
        )
    return documents, latency_ms


async def generate(
    prompt: str,
    metrics: Metrics,
    *,
    should_fail: bool,
) -> tuple[str, float]:
    model = "demo-model"
    provider = "fake-provider"
    input_tokens = 120
    metrics.llm_calls_total += 1
    LOGGER.debug(
        "llm_call_started",
        extra={
            "operation": "llm_generate",
            "model": model,
            "provider": provider,
            "input_tokens": input_tokens,
            "prompt_chars": len(prompt),
        },
    )
    started = time.perf_counter()
    await asyncio.sleep(0.04)
    latency_ms = milliseconds_since(started)
    if should_fail:
        metrics.llm_errors_total += 1
        LOGGER.error(
            "llm_call_failed",
            extra={
                "operation": "llm_generate",
                "status": "failed",
                "model": model,
                "provider": provider,
                "latency_ms": latency_ms,
                "error_type": "FakeModelError",
            },
        )
        raise FakeModelError

    output_tokens = 38
    LOGGER.info(
        "llm_call_completed",
        extra={
            "operation": "llm_generate",
            "status": "success",
            "model": model,
            "provider": provider,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "latency_ms": latency_ms,
        },
    )
    return "fake answer", latency_ms


async def handle_request(
    request_id: str,
    prompt: str,
    metrics: Metrics,
    *,
    documents_to_return: int,
    model_should_fail: bool,
) -> RequestObservation:
    context_token = REQUEST_ID.set(request_id)
    request_started = time.perf_counter()
    spans: list[Span] = []
    documents_retrieved = 0
    metrics.requests_total += 1
    LOGGER.info(
        "request_started",
        extra={
            "operation": "answer_question",
            "status": "started",
            "prompt_chars": len(prompt),
            "api_key_configured": True,
        },
    )

    try:
        documents, retrieval_ms = await retrieve(documents_to_return, metrics)
        documents_retrieved = len(documents)
        spans.append(Span("retrieval", retrieval_ms))

        try:
            _, llm_ms = await generate(prompt, metrics, should_fail=model_should_fail)
        except FakeModelError:
            metrics.requests_failed += 1
            request_ms = milliseconds_since(request_started)
            LOGGER.error(
                "request_failed",
                extra={
                    "operation": "answer_question",
                    "status": "failed",
                    "latency_ms": request_ms,
                    "error_type": "FakeModelError",
                },
            )
            return RequestObservation(
                request_id, "failed", request_ms, documents_retrieved, spans
            )

        spans.append(Span("llm", llm_ms))
        request_ms = milliseconds_since(request_started)
        LOGGER.info(
            "request_completed",
            extra={
                "operation": "answer_question",
                "status": "success",
                "latency_ms": request_ms,
            },
        )
        return RequestObservation(
            request_id, "success", request_ms, documents_retrieved, spans
        )
    finally:
        REQUEST_ID.reset(context_token)


def show_trace(observation: RequestObservation) -> None:
    print(f"\nTrace summary: {observation.request_id}")
    print(f"request ({observation.latency_ms:.2f} ms, {observation.status})")
    for span in observation.spans:
        print(f"├── {span.name:<9} {span.duration_ms:.2f} ms")


def show_metrics(metrics: Metrics) -> None:
    print(
        "\nMetrics summary\n"
        f"requests_total: {metrics.requests_total}\n"
        f"requests_failed: {metrics.requests_failed}\n"
        f"llm_calls_total: {metrics.llm_calls_total}\n"
        f"llm_errors_total: {metrics.llm_errors_total}"
    )


def explain_observability_models() -> None:
    print(
        "\nLogs / metrics / traces\n"
        "LOG: llm_call_failed request_id=req-002 -> what happened?\n"
        "METRIC: llm_errors_total=1 -> how often?\n"
        "TRACE: request -> retrieval -> llm -> where was time spent?"
    )
    print("Keep request_id in logs/traces, not as a high-cardinality metric label.")
    print("Bounded metric labels such as model=demo-model aggregate efficiently.")


async def main() -> None:
    configure_logging()
    configure_logging()  # Idempotent: this does not add a second handler.
    handlers = [h for h in LOGGER.handlers if h.get_name() == HANDLER_NAME]
    if len(handlers) != 1:
        raise AssertionError("logging setup duplicated its JSON handler")

    print("Unstructured sentences are hard to aggregate; prefer event + safe fields.")
    print("DEBUG=diagnostic, INFO=lifecycle, WARNING=degraded, ERROR=failed.")
    metrics = Metrics()
    sensitive_prompt = "Sensitive customer question that must not appear in logs"

    successful = await handle_request(
        "req-001",
        sensitive_prompt,
        metrics,
        documents_to_return=3,
        model_should_fail=False,
    )
    failed = await handle_request(
        "req-002",
        sensitive_prompt,
        metrics,
        documents_to_return=1,
        model_should_fail=True,
    )

    print(
        f"\nOperational summary: request_id={successful.request_id} "
        f"status={successful.status} documents={successful.documents_retrieved} "
        f"latency_ms={successful.latency_ms:.2f}"
    )
    print(
        f"Failed request summary: request_id={failed.request_id} status={failed.status}"
    )
    show_metrics(metrics)
    show_trace(successful)
    explain_observability_models()
    print("Safe logging: record api_key_configured=true; never record the key itself.")


if __name__ == "__main__":
    asyncio.run(main())
