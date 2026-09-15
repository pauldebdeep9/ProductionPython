"""Consolidated high-value contract tests for ProductionPythonCore."""

from __future__ import annotations

import asyncio
import importlib.util
import io
import json
import logging
import os
import subprocess
import sys
from functools import cache
from pathlib import Path
from types import ModuleType
from typing import Protocol, cast
from unittest.mock import patch

import httpx
import pytest
from pydantic import ValidationError


ROOT = Path(__file__).resolve().parent
CONFIG_ENV_NAMES = (
    "APP_ENV APP_HOST APP_PORT LLM_MODEL LLM_API_KEY MAX_CONCURRENCY "
    "REQUEST_TIMEOUT_SECONDS MAX_ATTEMPTS"
).split()


class Topic1Settings(Protocol):
    app_env: str
    max_concurrency: int
    llm_api_key: object


@cache
def load_topic(filename: str) -> ModuleType:
    module_name = f"production_core_test_{Path(filename).stem}"
    spec = importlib.util.spec_from_file_location(module_name, ROOT / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load tutorial module: {filename}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


TUTORIAL_SCRIPTS = (
    ("01_config_secrets.py", "External configuration"),
    ("02_validation_typing.py", "Type annotations guide tools"),
    ("03_async_concurrency.py", "Waiting can overlap"),
    ("04_failures_retries.py", "Classify -> attempt"),
    ("05_observability.py", "Unstructured sentences"),
    ("06_testing_patterns.py", "Testing mental model"),
    ("07_service_api.py", "Production Python Core"),
    ("08_ai_service_capstone.py", "Production AI Service Capstone"),
)


@pytest.mark.parametrize(("filename", "identity"), TUTORIAL_SCRIPTS)
def test_tutorial_script_smoke(filename: str, identity: str) -> None:
    environment = os.environ.copy()
    for name in CONFIG_ENV_NAMES:
        environment.pop(name, None)
    result = subprocess.run(
        [sys.executable, filename],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert identity in result.stdout


def make_topic1_settings(topic: ModuleType, **overrides: object) -> Topic1Settings:
    values: dict[str, object] = {
        "app_env": "dev",
        "llm_model": "test-model",
        "llm_api_key": "SYNTHETIC_TEST_SECRET",
        "max_concurrency": 2,
        "request_timeout_seconds": 1.0,
    }
    values.update(overrides)
    return cast(Topic1Settings, topic.Settings(**values))


def test_topic1_configuration_and_secret_contracts() -> None:
    topic = load_topic("01_config_secrets.py")
    settings = make_topic1_settings(topic)
    assert settings.app_env == "dev"
    assert settings.max_concurrency == 2
    with pytest.raises(ValidationError):
        make_topic1_settings(topic, max_concurrency=0)
    with pytest.raises(ValidationError):
        make_topic1_settings(topic, app_env="prod", llm_api_key=None)
    marker = "SYNTHETIC_TEST_SECRET"
    assert marker not in str(settings.llm_api_key)
    assert marker not in repr(settings.llm_api_key)


def valid_answer_data() -> dict[str, object]:
    return {
        "answer": "Line 2 is delayed.",
        "confidence": 0.91,
        "citations": [{"document_id": "production-log-17", "page": 3}],
    }


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [("confidence", 1.1), ("page", -1)],
)
def test_topic2_invalid_nested_values(field: str, invalid_value: object) -> None:
    topic = load_topic("02_validation_typing.py")
    data = valid_answer_data()
    if field == "confidence":
        data["confidence"] = invalid_value
    else:
        data["citations"] = [
            {"document_id": "production-log-17", "page": invalid_value}
        ]
    with pytest.raises(ValidationError):
        topic.Answer.model_validate(data)


@pytest.mark.parametrize("valid", [True, False])
def test_topic2_llm_json_schema_validation(valid: bool) -> None:
    topic = load_topic("02_validation_typing.py")
    confidence = 0.87 if valid else 4.2
    payload = json.dumps(
        {
            "answer": "Production can continue.",
            "confidence": confidence,
            "citations": [{"document_id": "shift-report-8", "page": None}],
        }
    )
    if valid:
        assert topic.Answer.model_validate_json(payload).confidence == 0.87
    else:
        with pytest.raises(ValidationError):
            topic.Answer.model_validate_json(payload)


def test_topic3_semaphore_bounds_downstream_calls() -> None:
    topic = load_topic("03_async_concurrency.py")

    async def scenario() -> None:
        service = topic.FakeLLMService(default_delay_seconds=0.005)
        semaphore = asyncio.Semaphore(2)
        await asyncio.gather(
            *(
                topic.call_with_limit(service, semaphore, f"prompt-{number}")
                for number in range(6)
            )
        )
        assert service.maximum_active_calls == 2

    asyncio.run(scenario())


def test_topic4_transient_failures_recover() -> None:
    topic = load_topic("04_failures_retries.py")
    client = topic.FakeLLMClient(
        [topic.RateLimitError("429"), topic.ServerError("503"), "success"]
    )
    result = asyncio.run(
        topic.call_with_retry(client, "safe prompt", max_attempts=4, base_delay=0)
    )
    assert result == "success"
    assert client.calls == 3


def test_topic4_non_retryable_error_fails_immediately() -> None:
    topic = load_topic("04_failures_retries.py")
    client = topic.FakeLLMClient([topic.AuthenticationError("synthetic")])
    with pytest.raises(topic.AuthenticationError):
        asyncio.run(topic.call_with_retry(client, "safe prompt", max_attempts=4))
    assert client.calls == 1
    assert not topic.is_retryable(topic.InvalidLLMOutput("invalid schema"))


def test_topic4_retry_exhaustion_has_no_final_sleep() -> None:
    topic = load_topic("04_failures_retries.py")
    client = topic.FakeLLMClient(topic.ServerError("503") for _ in range(3))
    observed_delays: list[float] = []

    async def record_sleep(delay: float) -> None:
        observed_delays.append(delay)

    with patch.object(topic.asyncio, "sleep", new=record_sleep):
        with pytest.raises(topic.RetryExhausted):
            asyncio.run(
                topic.call_with_retry(
                    client,
                    "safe prompt",
                    max_attempts=3,
                    base_delay=0.001,
                )
            )
    assert client.calls == 3
    assert len(observed_delays) == 2


def test_topic5_pipeline_logs_correlate_without_sensitive_prompt() -> None:
    topic = load_topic("05_observability.py")
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(topic.JsonFormatter())
    handler.addFilter(topic.RequestContextFilter())
    logger = topic.LOGGER
    previous_handlers = logger.handlers[:]
    previous_level, previous_propagate = logger.level, logger.propagate
    logger.handlers = [handler]
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    marker = "TOP_SECRET_TEST_VALUE PRIVATE_PROMPT_TEST_VALUE"
    try:
        observation = asyncio.run(
            topic.handle_request(
                "req-safe-test",
                marker,
                topic.Metrics(),
                documents_to_return=3,
                model_should_fail=False,
            )
        )
    finally:
        logger.handlers = previous_handlers
        logger.setLevel(previous_level)
        logger.propagate = previous_propagate
    records = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert observation.status == "success"
    assert records and all(
        record["request_id"] == "req-safe-test" for record in records
    )
    assert marker not in stream.getvalue()
    assert all("event" in record and "level" in record for record in records)
    assert any(
        record["event"] == "llm_call_completed"
        and record["model"] == "demo-model"
        and record["status"] == "success"
        for record in records
    )


def test_topic7_http_contracts_and_lifecycle() -> None:
    topic = load_topic("07_service_api.py")

    async def scenario() -> None:
        async with topic.app.router.lifespan_context(topic.app):
            client_resource = topic.app.state.llm
            transport = httpx.ASGITransport(app=topic.app, raise_app_exceptions=False)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                assert (await client.get("/health")).status_code == 200
                assert (await client.get("/ready")).status_code == 200
                topic.app.state.ready = False
                assert (await client.get("/health")).status_code == 200
                assert (await client.get("/ready")).status_code == 503
                topic.app.state.ready = True
                response = await client.post(
                    "/answer",
                    json={"question": "Is Line 1 available?"},
                    headers={"X-Request-ID": "test-request-123"},
                )
                assert response.status_code == 200
                assert response.json()["request_id"] == "test-request-123"
                assert response.headers["X-Request-ID"] == "test-request-123"
                assert response.json()["answer"]
                assert 0 <= response.json()["confidence"] <= 1
                assert (await client.post("/answer", json={})).status_code == 422
                failure = await client.post(
                    "/answer", json={"question": "simulate failure"}
                )
                assert failure.status_code == 503
                assert (
                    failure.json()["detail"] == "Model service temporarily unavailable"
                )
                assert "tutorial-only" not in failure.text
        assert not topic.app.state.ready and client_resource.closed

    asyncio.run(scenario())


def capstone_settings(topic: ModuleType, *, max_concurrency: int = 2) -> object:
    return topic.Settings(
        app_env="dev",
        app_host="127.0.0.1",
        app_port=8000,
        llm_model="fake-production-model",
        max_concurrency=max_concurrency,
        request_timeout_seconds=0.1,
        max_attempts=3,
    )


@pytest.mark.parametrize(
    "scenario_name",
    ["normal", "no_evidence", "transient"],
)
def test_capstone_service_scenarios(scenario_name: str) -> None:
    topic = load_topic("08_ai_service_capstone.py")
    questions = {
        "normal": "Why is Line 2 delayed?",
        "no_evidence": "What is the weather tomorrow?",
        "transient": "simulate transient failure",
    }

    async def scenario() -> None:
        resources = await topic.create_resources(capstone_settings(topic))
        try:
            result = await resources.service.answer(questions[scenario_name])
            assert result.answer
            if scenario_name == "normal":
                assert result.confidence > 0
                assert "production-log-17" in {
                    citation.document_id for citation in result.citations
                }
            elif scenario_name == "no_evidence":
                assert result.confidence == 0.0 and result.citations == []
                assert resources.metrics.llm_attempts_total == 0
            else:
                assert (
                    resources.client.attempts_by_question["simulate transient failure"]
                    == 2
                )
                assert resources.metrics.llm_attempts_total == 2
                assert resources.metrics.llm_retries_total == 1
        finally:
            await resources.client.close()

    asyncio.run(scenario())


def test_capstone_http_errors_request_id_and_lifecycle() -> None:
    topic = load_topic("08_ai_service_capstone.py")
    app = topic.app
    previous_settings_override = getattr(app.state, "settings_override", None)
    app.state.settings_override = capstone_settings(topic)
    logger = topic.LOGGER
    previous_handlers = logger.handlers[:]
    null_handler = logging.NullHandler()
    null_handler.set_name("capstone-json")
    logger.handlers = [null_handler]

    async def scenario() -> None:
        async with app.router.lifespan_context(app):
            resource = app.state.resources.client
            assert app.state.ready and resource.initialized
            transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                bad = await client.post(
                    "/answer",
                    json={"question": "simulate bad output"},
                    headers={"X-Request-ID": "capstone-test-001"},
                )
                assert bad.status_code == 502
                assert bad.headers["X-Request-ID"] == "capstone-test-001"
                assert bad.json() == {
                    "detail": "Model returned an invalid response",
                    "request_id": "capstone-test-001",
                }
                assert 'confidence":1.5' not in bad.text
                assert (await client.post("/answer", json={})).status_code == 422
        assert not app.state.ready and resource.closed

    try:
        asyncio.run(scenario())
    finally:
        app.state.settings_override = previous_settings_override
        logger.handlers = previous_handlers


def test_capstone_model_gateway_enforces_concurrency_limit() -> None:
    topic = load_topic("08_ai_service_capstone.py")

    async def scenario() -> None:
        resources = await topic.create_resources(
            capstone_settings(topic, max_concurrency=2)
        )
        try:
            await asyncio.gather(
                *(
                    resources.service.answer(
                        f"Why is Line 2 delayed? concurrent {number}"
                    )
                    for number in range(6)
                )
            )
            assert resources.client.maximum_active_calls == 2
        finally:
            await resources.client.close()

    asyncio.run(scenario())


def test_capstone_structured_logs_do_not_leak_inputs() -> None:
    topic = load_topic("08_ai_service_capstone.py")
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(topic.EventFormatter())
    logger = topic.LOGGER
    previous_handlers = logger.handlers[:]
    previous_level, previous_propagate = logger.level, logger.propagate
    logger.handlers = [handler]
    logger.setLevel(logging.INFO)
    logger.propagate = False
    request_id = "capstone-log-test"
    prompt_marker = "PRIVATE_PROMPT_TEST_VALUE"
    secret_marker = "TOP_SECRET_TEST_VALUE"

    async def scenario() -> None:
        resources = await topic.create_resources(capstone_settings(topic))
        token = topic.REQUEST_ID.set(request_id)
        try:
            await resources.service.answer(
                f"Why is Line 2 delayed? {prompt_marker} {secret_marker}"
            )
        finally:
            topic.REQUEST_ID.reset(token)
            await resources.client.close()

    try:
        asyncio.run(scenario())
    finally:
        logger.handlers = previous_handlers
        logger.setLevel(previous_level)
        logger.propagate = previous_propagate

    output = stream.getvalue()
    records = [json.loads(line) for line in output.splitlines()]
    assert records and all(record["request_id"] == request_id for record in records)
    assert {"request_started", "retrieval_completed", "llm_completed"} <= {
        record["event"] for record in records
    }
    assert prompt_marker not in output
    assert secret_marker not in output
    assert KNOWLEDGE_TEXT_MARKER not in output
    assert '"answer"' not in output


KNOWLEDGE_TEXT_MARKER = "Line 2 is delayed because material MX-42 has not arrived."
