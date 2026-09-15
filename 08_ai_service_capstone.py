"""Topic 8: Offline AI assistant behind normal production-service boundaries."""

import asyncio
import json
import logging
import sys
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Annotated, Literal, Protocol
from uuid import uuid4

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: Literal["dev", "prod"] = "dev"
    llm_model: str = "fake-production-model"
    max_concurrency: int = Field(default=3, ge=1, le=20)
    request_timeout_seconds: float = Field(default=0.2, gt=0)
    max_attempts: int = Field(default=3, ge=1, le=5)
    app_host: str = "127.0.0.1"
    app_port: int = Field(default=8000, ge=1, le=65535)


class AnswerRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)


class Citation(BaseModel):
    document_id: str = Field(min_length=1)
    page: int | None = Field(default=None, ge=1)


class GeneratedAnswer(BaseModel):
    answer: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    citations: list[Citation]


class AnswerResponse(GeneratedAnswer):
    request_id: str


class StatusResponse(BaseModel):
    status: Literal["ok", "ready"]


@dataclass(frozen=True)
class Document:
    document_id: str
    text: str


KNOWLEDGE_BASE = (
    Document(
        "production-log-17", "Line 2 is delayed because material MX-42 has not arrived."
    ),
    Document(
        "maintenance-plan-03", "Line 3 has planned maintenance from 10:00 to 14:00."
    ),
    Document("production-plan-08", "Orders WO-101 and WO-102 can run on Line 1."),
)
TUTORIAL_MODEL_TRIGGERS = {"simulate transient failure", "simulate bad output"}


class FakeRetriever:
    async def retrieve(self, question: str, *, limit: int = 2) -> list[Document]:
        await asyncio.sleep(0.005)
        if question.casefold() in TUTORIAL_MODEL_TRIGGERS:
            return [KNOWLEDGE_BASE[0]]

        stop_words = {"a", "can", "is", "the", "what", "while", "why"}
        tokens = {
            token.strip(".,?!").casefold()
            for token in question.split()
            if token.strip(".,?!").casefold() not in stop_words
        }
        ranked = []
        for index, document in enumerate(KNOWLEDGE_BASE):
            searchable = f"{document.document_id} {document.text}".casefold()
            score = sum(token in searchable for token in tokens)
            if score:
                ranked.append((score, -index, document))
        ranked.sort(reverse=True)
        return [document for _, _, document in ranked[:limit]]


class ModelError(Exception): ...


class RetryableModelError(ModelError): ...


class RateLimitError(RetryableModelError): ...


class RetryExhausted(ModelError): ...


class InvalidModelOutput(ModelError): ...


class LLMClient(Protocol):
    async def generate(self, question: str, documents: list[Document]) -> str: ...


class FakeLLMClient:
    def __init__(self) -> None:
        self.initialized = False
        self.closed = False
        self.active_calls = 0
        self.maximum_active_calls = 0
        self.attempts_by_question: dict[str, int] = {}

    async def initialize(self) -> None:
        await asyncio.sleep(0)
        self.initialized = True

    async def generate(self, question: str, documents: list[Document]) -> str:
        if not self.initialized or self.closed:
            raise RetryableModelError
        attempt = self.attempts_by_question.get(question, 0) + 1
        self.attempts_by_question[question] = attempt
        self.active_calls += 1
        self.maximum_active_calls = max(self.maximum_active_calls, self.active_calls)
        try:
            await asyncio.sleep(0.015)
            if question.casefold() == "simulate transient failure" and attempt == 1:
                raise RateLimitError
            if question.casefold() == "simulate bad output":
                return '{"answer":"","confidence":1.5,"citations":[]}'

            payload = {
                "answer": documents[0].text,
                "confidence": 0.94,
                "citations": [
                    {"document_id": document.document_id, "page": None}
                    for document in documents
                ],
            }
            return json.dumps(payload)
        finally:
            self.active_calls -= 1

    async def close(self) -> None:
        await asyncio.sleep(0)
        self.closed = True


REQUEST_ID: ContextVar[str] = ContextVar("request_id", default="unscoped")
LOGGER = logging.getLogger("production_core.capstone")


class EventFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "level": record.levelname,
            "event": record.getMessage(),
            "request_id": REQUEST_ID.get(),
        }
        fields = getattr(record, "event_fields", {})
        if isinstance(fields, dict):
            payload.update(fields)
        return json.dumps(payload, separators=(",", ":"))


def configure_logging() -> None:
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False
    if not any(handler.get_name() == "capstone-json" for handler in LOGGER.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler.set_name("capstone-json")
        handler.setFormatter(EventFormatter())
        LOGGER.addHandler(handler)


def log_event(event: str, level: int = logging.INFO, **fields: object) -> None:
    LOGGER.log(level, event, extra={"event_fields": fields})


@dataclass
class Metrics:
    requests_total: int = 0
    requests_failed: int = 0
    llm_attempts_total: int = 0
    llm_retries_total: int = 0


class ModelGateway:
    def __init__(self, client: LLMClient, settings: Settings, metrics: Metrics) -> None:
        self.client = client
        self.model_name = settings.llm_model
        self.semaphore = asyncio.Semaphore(settings.max_concurrency)
        self.timeout_seconds = settings.request_timeout_seconds
        self.max_attempts = settings.max_attempts
        self.metrics = metrics

    async def generate(self, question: str, documents: list[Document]) -> str:
        for attempt in range(1, self.max_attempts + 1):
            self.metrics.llm_attempts_total += 1
            log_event("llm_attempt", model=self.model_name, attempt=attempt)
            started = time.perf_counter()
            try:
                # Capacity is held only during a downstream attempt, not backoff.
                async with self.semaphore:
                    async with asyncio.timeout(self.timeout_seconds):
                        raw_output = await self.client.generate(question, documents)
            except (RetryableModelError, TimeoutError) as error:
                if attempt == self.max_attempts:
                    raise RetryExhausted from error
                delay = 0.01 * (2 ** (attempt - 1))
                self.metrics.llm_retries_total += 1
                log_event(
                    "llm_retry",
                    level=logging.WARNING,
                    model=self.model_name,
                    attempt=attempt,
                    error_type=type(error).__name__,
                    backoff_seconds=delay,
                )
                await asyncio.sleep(delay)
                continue

            latency_ms = round((time.perf_counter() - started) * 1_000, 2)
            log_event(
                "llm_completed",
                model=self.model_name,
                attempt=attempt,
                latency_ms=latency_ms,
                status="success",
            )
            return raw_output
        raise AssertionError("retry loop should return or raise")


class AnswerService:
    def __init__(
        self,
        retriever: FakeRetriever,
        gateway: ModelGateway,
        metrics: Metrics,
    ) -> None:
        self.retriever = retriever
        self.gateway = gateway
        self.metrics = metrics

    async def answer(self, question: str) -> GeneratedAnswer:
        self.metrics.requests_total += 1
        request_started = time.perf_counter()
        log_event("request_started", status="started")
        try:
            retrieval_started = time.perf_counter()
            documents = await self.retriever.retrieve(question)
            retrieval_ms = round((time.perf_counter() - retrieval_started) * 1_000, 2)
            log_event(
                "retrieval_completed",
                documents_retrieved=len(documents),
                latency_ms=retrieval_ms,
                status="success",
            )

            if not documents:
                result = GeneratedAnswer(
                    answer="I do not have enough information to answer that question.",
                    confidence=0.0,
                    citations=[],
                )
                status_name = "no_evidence"
            else:
                raw_output = await self.gateway.generate(question, documents)
                try:
                    result = GeneratedAnswer.model_validate_json(raw_output)
                except ValidationError as error:
                    raise InvalidModelOutput from error
                status_name = "success"

            request_ms = round((time.perf_counter() - request_started) * 1_000, 2)
            log_event(
                "request_completed",
                latency_ms=request_ms,
                status=status_name,
            )
            return result
        except (RetryExhausted, InvalidModelOutput) as error:
            self.metrics.requests_failed += 1
            log_event(
                "request_failed",
                level=logging.ERROR,
                status="failed",
                error_type=type(error).__name__,
            )
            raise


@dataclass
class Resources:
    client: FakeLLMClient
    metrics: Metrics
    service: AnswerService


async def create_resources(settings: Settings) -> Resources:
    metrics = Metrics()
    client = FakeLLMClient()
    await client.initialize()
    gateway = ModelGateway(client, settings, metrics)
    service = AnswerService(FakeRetriever(), gateway, metrics)
    return Resources(client, metrics, service)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    configure_logging()
    app.state.ready = False
    settings = getattr(app.state, "settings_override", None) or Settings()
    resources = await create_resources(settings)
    app.state.resources = resources
    app.state.ready = True
    try:
        yield
    finally:
        app.state.ready = False
        await resources.client.close()


app = FastAPI(title="Manufacturing Knowledge Assistant", lifespan=lifespan)


def get_resources(request: Request) -> Resources:
    if not getattr(request.app.state, "ready", False):
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Not ready")
    resources: Resources = request.app.state.resources
    return resources


def safe_error(request: Request, detail: str, status_code: int) -> JSONResponse:
    request_id = getattr(request.state, "request_id", str(uuid4()))
    return JSONResponse(
        status_code=status_code,
        content={"detail": detail, "request_id": request_id},
        headers={"X-Request-ID": request_id},
    )


@app.exception_handler(ModelError)
async def handle_model_error(request: Request, error: ModelError) -> JSONResponse:
    if isinstance(error, RetryExhausted):
        return safe_error(
            request,
            "Model service temporarily unavailable",
            status.HTTP_503_SERVICE_UNAVAILABLE,
        )
    return safe_error(
        request,
        "Model returned an invalid response",
        status.HTTP_502_BAD_GATEWAY,
    )


@app.get("/health", response_model=StatusResponse)
async def health() -> StatusResponse:
    return StatusResponse(status="ok")


@app.get("/ready", response_model=StatusResponse)
async def ready(request: Request) -> StatusResponse:
    if not getattr(request.app.state, "ready", False):
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Not ready")
    return StatusResponse(status="ready")


@app.post("/answer", response_model=AnswerResponse)
async def answer(
    body: AnswerRequest,
    request: Request,
    response: Response,
    resources: Annotated[Resources, Depends(get_resources)],
) -> AnswerResponse:
    request_id = request.headers.get("X-Request-ID") or str(uuid4())
    request.state.request_id = request_id
    context_token = REQUEST_ID.set(request_id)
    try:
        result = await resources.service.answer(body.question)
        response.headers["X-Request-ID"] = request_id
        return AnswerResponse(request_id=request_id, **result.model_dump())
    finally:
        REQUEST_ID.reset(context_token)


async def run_direct_tutorial() -> None:
    configure_logging()
    settings = Settings(
        llm_model="fake-production-model",
        max_concurrency=2,
        request_timeout_seconds=0.1,
        max_attempts=3,
    )
    resources = await create_resources(settings)
    scenarios = (
        ("direct-001", "Why is Line 2 delayed?"),
        ("direct-002", "simulate transient failure"),
        ("direct-003", "simulate bad output"),
    )
    print("Production AI Service Capstone")
    try:
        for request_id, question in scenarios:
            token = REQUEST_ID.set(request_id)
            try:
                result = await resources.service.answer(question)
            except InvalidModelOutput:
                print(f"{request_id}: invalid model output rejected safely")
            else:
                citations = ", ".join(c.document_id for c in result.citations)
                print(f"{request_id}: {result.answer}")
                print(f"  confidence={result.confidence:.2f} citations={citations}")
            finally:
                REQUEST_ID.reset(token)
    finally:
        await resources.client.close()

    metrics = resources.metrics
    print(
        "Metrics: "
        f"requests={metrics.requests_total} failed={metrics.requests_failed} "
        f"llm_attempts={metrics.llm_attempts_total} "
        f"retries={metrics.llm_retries_total}"
    )
    print("Run the API: python 08_ai_service_capstone.py --serve")


def main() -> None:
    arguments = sys.argv[1:]
    if not arguments:
        asyncio.run(run_direct_tutorial())
        return
    if arguments == ["--serve"]:
        settings = Settings()
        uvicorn.run(app, host=settings.app_host, port=settings.app_port)
        return
    raise SystemExit("Usage: python 08_ai_service_capstone.py [--serve]")


if __name__ == "__main__":
    main()
