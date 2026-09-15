"""Topic 4: Failure handling, retries, backoff, and timeouts.

External dependencies fail in different ways. Production code should classify a
failure before deciding to retry, fail fast, repair output, or use a fallback:

    external operation fails
              |
              v
       classify failure
              |
       +------+------+----------+
       |             |          |
     retry       fail fast   repair/fallback

The key rule is that not every failure should be retried. This tutorial uses a
deterministic fake LLM client and never contacts an external service.
"""

from __future__ import annotations

import asyncio
import json
import random
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass


class LLMError(Exception):
    """Base error for expected failures at the fake LLM boundary."""


class RetryableLLMError(LLMError):
    """A transient service failure for which another attempt may help."""


class RateLimitError(RetryableLLMError):
    """The provider temporarily rejected excess traffic (for example, 429)."""


class ServerError(RetryableLLMError):
    """The provider temporarily failed (for example, 5xx)."""


class AuthenticationError(LLMError):
    """Credentials are absent or invalid; retrying unchanged credentials will not help."""


class BadRequestError(LLMError):
    """The request itself must change before another call can succeed."""


class InvalidLLMOutput(LLMError):
    """The call succeeded, but returned content that violates the application schema."""


class RetryExhausted(LLMError):
    """All allowed attempts failed with retryable errors."""

    def __init__(self, attempts: int, last_error: Exception) -> None:
        self.attempts = attempts
        self.last_error = last_error
        error_type = type(last_error).__name__
        super().__init__(
            f"retry exhausted after {attempts} attempts; last_error={error_type}"
        )


@dataclass(frozen=True)
class DelayedResponse:
    """A successful response that takes long enough to exercise a timeout."""

    value: str
    delay_seconds: float


Outcome = str | Exception | DelayedResponse


class FakeLLMClient:
    """Consume a predefined outcome on every call."""

    def __init__(self, outcomes: Iterable[Outcome]) -> None:
        self._outcomes: deque[Outcome] = deque(outcomes)
        self.calls = 0

    async def generate(self, prompt: str) -> str:
        """Return, delay, or raise exactly as the next outcome specifies."""
        del prompt  # Sensitive prompt text is intentionally not logged.
        self.calls += 1
        if not self._outcomes:
            raise RuntimeError("fake client has no configured outcome")

        outcome = self._outcomes.popleft()
        if isinstance(outcome, DelayedResponse):
            await asyncio.sleep(outcome.delay_seconds)
            return outcome.value
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def is_retryable(error: Exception) -> bool:
    """Make the infrastructure retry policy explicit and reviewable."""
    return isinstance(error, (RetryableLLMError, TimeoutError))


def calculate_backoff(
    base_delay: float,
    failed_attempt: int,
    random_source: random.Random,
) -> float:
    """Return exponential delay plus small deterministic jitter."""
    exponential_delay = base_delay * (2 ** (failed_attempt - 1))
    jitter = random_source.uniform(0.0, base_delay / 2)
    return exponential_delay + jitter


async def call_with_retry(
    client: FakeLLMClient,
    prompt: str,
    *,
    max_attempts: int = 3,
    base_delay: float = 0.01,
    timeout_seconds: float = 0.05,
    random_source: random.Random | None = None,
) -> str:
    """Call an LLM with visible max-attempt, timeout, and backoff semantics."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    if base_delay < 0 or timeout_seconds <= 0:
        raise ValueError("delays must be non-negative and timeout must be positive")

    jitter_source = random_source or random.Random(0)
    for attempt in range(1, max_attempts + 1):
        print(f"operation=llm_generate attempt={attempt}/{max_attempts}")
        try:
            # This timeout applies to one attempt, not the entire retry sequence.
            async with asyncio.timeout(timeout_seconds):
                return await client.generate(prompt)
        except (LLMError, TimeoutError) as error:
            error_type = type(error).__name__
            if not is_retryable(error):
                print(f"error_type={error_type} classification=fail_fast")
                raise

            print(f"error_type={error_type} classification=retryable")
            if attempt == max_attempts:
                # There is deliberately no sleep after the final failed attempt.
                raise RetryExhausted(attempt, error) from error

            delay = calculate_backoff(base_delay, attempt, jitter_source)
            print(f"backoff={delay:.3f}s (exponential + deterministic jitter)")
            await asyncio.sleep(delay)

    raise AssertionError("retry loop should always return or raise")


def demo_failure_classification() -> None:
    print("\nExample A - explicit failure classification")
    failures: tuple[Exception, ...] = (
        RateLimitError("rate limited"),
        ServerError("service unavailable"),
        TimeoutError("attempt timed out"),
        AuthenticationError("invalid credentials"),
        BadRequestError("invalid request"),
    )
    for failure in failures:
        decision = "retry" if is_retryable(failure) else "fail fast"
        print(f"{type(failure).__name__:20} -> {decision}")


async def demo_successful_retry() -> None:
    print("\nExample B - transient failures then success")
    client = FakeLLMClient(
        [
            RateLimitError("429"),
            ServerError("503"),
            "valid response",
        ]
    )
    response = await call_with_retry(
        client,
        "Summarize the work order",
        max_attempts=4,
        random_source=random.Random(7),
    )
    print(f"Result: {response}; calls={client.calls}")
    if client.calls != 3:
        raise AssertionError("successful sequence should use exactly three attempts")


async def demo_fail_fast() -> None:
    print("\nExample C - non-retryable authentication failure")
    client = FakeLLMClient([AuthenticationError("secret is never printed")])
    try:
        await call_with_retry(client, "Private prompt", max_attempts=4)
    except AuthenticationError:
        print("Authentication failure is non-retryable; failed immediately.")
    else:
        raise AssertionError("authentication failure should propagate")

    if client.calls != 1:
        raise AssertionError("non-retryable error must not receive a second attempt")


async def demo_retry_exhaustion() -> None:
    print("\nExample D - retry exhaustion")
    client = FakeLLMClient(ServerError("503") for _ in range(3))
    try:
        await call_with_retry(
            client,
            "Generate status",
            max_attempts=3,
            base_delay=0.005,
        )
    except RetryExhausted as error:
        print(f"Stopped clearly after attempts={error.attempts}; calls={client.calls}")
    else:
        raise AssertionError("repeated transient failures should exhaust retries")

    if client.calls != 3:
        raise AssertionError("max_attempts=3 means at most three total calls")


async def demo_timeout_then_success() -> None:
    print("\nExample E - per-attempt timeout then success")
    client = FakeLLMClient(
        [
            DelayedResponse("too late", delay_seconds=0.05),
            "response from the second attempt",
        ]
    )
    response = await call_with_retry(
        client,
        "Classify the request",
        max_attempts=2,
        base_delay=0.005,
        timeout_seconds=0.01,
    )
    print(f"Result: {response}; calls={client.calls}")
    print("Production callers often also enforce one overall request deadline.")


async def generate_with_fallback(
    primary: FakeLLMClient,
    fallback: FakeLLMClient,
    prompt: str,
) -> str:
    """Switch provider only after retryable failures exhaust the primary policy."""
    try:
        return await call_with_retry(
            primary,
            prompt,
            max_attempts=2,
            base_delay=0.005,
        )
    except RetryExhausted:
        print("Primary retry policy exhausted; switching to fallback provider.")
        return await call_with_retry(fallback, prompt, max_attempts=1)


async def demo_fallback() -> None:
    print("\nExample F - fallback after appropriate retry exhaustion")
    primary = FakeLLMClient([ServerError("503"), ServerError("503")])
    fallback = FakeLLMClient(["fallback response"])
    response = await generate_with_fallback(primary, fallback, "Answer safely")
    print(
        f"Result: {response}; primary_calls={primary.calls}; "
        f"fallback_calls={fallback.calls}"
    )


def validate_llm_output(raw_output: str) -> dict[str, object]:
    """Perform a tiny schema check after a successful infrastructure call."""
    try:
        value = json.loads(raw_output)
    except json.JSONDecodeError as error:
        raise InvalidLLMOutput("expected a JSON object") from error
    if not isinstance(value, dict) or not isinstance(value.get("answer"), str):
        raise InvalidLLMOutput("expected an object with a string answer")
    return value


def demo_invalid_llm_output() -> None:
    print("\nExample G - malformed output is not a transport failure")
    successful_call_output = "This is not the required JSON schema"
    try:
        validate_llm_output(successful_call_output)
    except InvalidLLMOutput as error:
        print(f"Invalid structured output detected: {error}.")
        print("Use a repair/regeneration policy, not blind infrastructure retries.")
    else:
        raise AssertionError("malformed structured output should be rejected")


def demo_idempotency() -> None:
    print("\nIdempotency check before retrying side effects")
    print("Usually safer to retry: read document, generate completion, fetch embedding.")
    print("Side effects need deduplication: use an idempotency key before retrying.")


async def main() -> None:
    """Run deterministic, offline failure-handling examples."""
    print("Classify -> attempt -> timeout -> backoff -> exhaust or fallback")
    demo_failure_classification()
    await demo_successful_retry()
    await demo_fail_fast()
    await demo_retry_exhaustion()
    await demo_timeout_then_success()
    await demo_fallback()
    demo_invalid_llm_output()
    demo_idempotency()


if __name__ == "__main__":
    asyncio.run(main())
