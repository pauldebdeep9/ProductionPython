"""Topic 7: HTTP service and application lifecycle.

The HTTP boundary validates untrusted requests before typed application logic
uses a lifespan-managed dependency:

    client -> HTTP/Pydantic -> answer service -> fake LLM -> HTTP response

Run ``python 07_service_api.py`` for usage or add ``--serve`` to start Uvicorn.
No API key, network dependency, or real model is required.
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Annotated, Literal
from uuid import uuid4

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field


class AnswerRequest(BaseModel):
    """The validated public request contract."""

    question: str = Field(min_length=1, max_length=500)
    priority: Literal["normal", "high"] = "normal"


class AnswerResponse(BaseModel):
    """The validated successful response contract."""

    answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    request_id: str


class StatusResponse(BaseModel):
    status: Literal["ok", "ready"]


class ErrorResponse(BaseModel):
    """Safe public error contract with a support-friendly correlation ID."""

    detail: str
    request_id: str


@dataclass(frozen=True)
class GeneratedAnswer:
    answer: str
    confidence: float


class FakeLLMError(Exception):
    """Known downstream availability failure translated at the HTTP boundary."""


class FakeLLMClient:
    """Deterministic async dependency initialized and closed by the application."""

    def __init__(self) -> None:
        self.initialized = False
        self.closed = False

    async def initialize(self) -> None:
        await asyncio.sleep(0)
        self.initialized = True

    async def generate(self, question: str) -> GeneratedAnswer:
        if not self.initialized or self.closed:
            raise FakeLLMError("dependency is not available")
        await asyncio.sleep(0.01)  # Simulate non-blocking network I/O.
        if question.casefold() == "simulate failure":
            # This internal detail is deliberately never returned to the client.
            raise FakeLLMError("tutorial-only downstream failure detail")
        return GeneratedAnswer(
            answer="Line 1 is available in this deterministic example.",
            confidence=0.92,
        )

    async def close(self) -> None:
        await asyncio.sleep(0)
        self.closed = True


async def answer_question(
    question: str,
    llm: FakeLLMClient,
) -> GeneratedAnswer:
    """Small application layer, independent of HTTP response construction."""
    return await llm.generate(question)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialize before readiness; mark unready and clean up on shutdown."""
    llm = FakeLLMClient()
    app.state.ready = False
    await llm.initialize()
    app.state.llm = llm
    app.state.ready = True
    try:
        yield
    finally:
        app.state.ready = False
        await llm.close()


app = FastAPI(
    title="Production Python Core - Topic 7",
    version="0.1.0",
    lifespan=lifespan,
)


def get_llm(request: Request) -> FakeLLMClient:
    """Obtain one application-level dependency initialized by lifespan."""
    if not getattr(request.app.state, "ready", False):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Application is not ready",
        )
    llm: FakeLLMClient = request.app.state.llm
    return llm


@app.exception_handler(FakeLLMError)
async def handle_fake_llm_error(
    request: Request,
    error: FakeLLMError,
) -> JSONResponse:
    """Translate a known internal failure into a stable, safe public response."""
    del error  # Internal exception text may contain details unsafe for clients.
    request_id = getattr(request.state, "request_id", str(uuid4()))
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content=ErrorResponse(
            detail="Model service temporarily unavailable",
            request_id=request_id,
        ).model_dump(),
        headers={"X-Request-ID": request_id},
    )


@app.get(
    "/health",
    response_model=StatusResponse,
    status_code=status.HTTP_200_OK,
)
async def health() -> StatusResponse:
    """Liveness: the process is alive; downstream readiness is irrelevant."""
    return StatusResponse(status="ok")


@app.get(
    "/ready",
    response_model=StatusResponse,
    status_code=status.HTTP_200_OK,
)
async def ready(request: Request) -> StatusResponse:
    """Readiness: required resources are initialized and can receive traffic."""
    if not getattr(request.app.state, "ready", False):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Application is not ready",
        )
    return StatusResponse(status="ready")


@app.post(
    "/answer",
    response_model=AnswerResponse,
    status_code=status.HTTP_200_OK,
)
async def answer(
    payload: AnswerRequest,
    request: Request,
    response: Response,
    llm: Annotated[FakeLLMClient, Depends(get_llm)],
) -> AnswerResponse:
    """Map validated HTTP input to application logic and a typed response."""
    request_id = request.headers.get("X-Request-ID") or str(uuid4())
    request.state.request_id = request_id
    generated = await answer_question(payload.question, llm)
    response.headers["X-Request-ID"] = request_id
    return AnswerResponse(
        answer=generated.answer,
        confidence=generated.confidence,
        request_id=request_id,
    )


def show_tutorial_summary() -> None:
    """Explain local usage without starting a blocking server."""
    print(
        """Production Python Core — Topic 7

Routes:
GET  /health  process alive? (liveness)
GET  /ready   safe to receive traffic? (readiness)
POST /answer  validated async AI-service boundary

Lifecycle: initialize dependency -> ready -> serve -> close dependency

Run the service:
python 07_service_api.py --serve

Try it:
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/ready
curl -X POST -H "Content-Type: application/json" \
  -d '{"question":"Is Line 1 available?"}' http://127.0.0.1:8000/answer
curl -X POST -H "Content-Type: application/json" \
  -d '{"question":"simulate failure"}' http://127.0.0.1:8000/answer

Interactive API contract: http://127.0.0.1:8000/docs
OpenAPI document: http://127.0.0.1:8000/openapi.json"""
    )


def main() -> None:
    """Choose concise tutorial output or an explicitly requested local server."""
    arguments = sys.argv[1:]
    if not arguments:
        show_tutorial_summary()
        return
    if arguments == ["--serve"]:
        uvicorn.run(app, host="127.0.0.1", port=8000)
        return
    raise SystemExit("Usage: python 07_service_api.py [--serve]")


if __name__ == "__main__":
    main()
