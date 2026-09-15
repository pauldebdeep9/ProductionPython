"""
fake_llm.py — a deterministic, offline stand-in for an async LLM / embedding client.

WHY THIS EXISTS
---------------
Every script in this tutorial needs something that behaves like Azure OpenAI:
network-bound, variable latency, occasionally rate-limited, occasionally
transiently broken. Hitting a real endpoint would make the lessons
non-reproducible and cost money. So we simulate.

The important property: `await asyncio.sleep(...)` is a *faithful* model of
network I/O for teaching purposes. Both yield control back to the event loop.
The event loop cannot tell the difference. What you learn about scheduling,
cancellation, and concurrency here transfers exactly.

The one thing sleep does NOT model is bandwidth/CPU cost of deserialising a
large response. That matters in real systems; it does not change the control
flow lessons.

DETERMINISM
-----------
Every client takes a seed. Latency and failure injection are drawn from a
seeded `random.Random` instance, NOT the global `random` module, so concurrent
clients don't interfere with each other's draw sequence.

NOTE: this file is imported by all the numbered scripts. Keep it in the same
directory as them.
"""

from __future__ import annotations

import asyncio
import hashlib
import random
import time
from dataclasses import dataclass, field
from typing import Self

# ---------------------------------------------------------------------------
# Exception taxonomy
# ---------------------------------------------------------------------------
# GOTCHA: The single most common production defect in LLM client code is
# retrying everything or retrying nothing. You need a taxonomy from day one so
# that retry logic can ask "is this class retryable?" rather than pattern-match
# on error strings. Real SDKs give you this (openai.RateLimitError,
# openai.APIStatusError, ...); we mirror the shape.

class LLMError(Exception):
    """Base for everything this fake client raises."""


class RateLimitError(LLMError):
    """HTTP 429. Retryable. Carries a server-suggested wait, like Retry-After."""

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class TransientServerError(LLMError):
    """HTTP 500/502/503/504. Retryable — the request may succeed unchanged."""


class ContentFilterError(LLMError):
    """HTTP 400 content filter. NOT retryable. The same input fails forever."""


class AuthError(LLMError):
    """HTTP 401/403. NOT retryable by the caller — a human must fix a secret."""


# ---------------------------------------------------------------------------
# Usage accounting
# ---------------------------------------------------------------------------

@dataclass
class Usage:
    """Token accounting. Real APIs return this; you should always capture it.

    REMEMBER: per-request token capture is what makes cost attribution by use
    case possible later. If you don't record it at the call site, you cannot
    reconstruct it.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            self.prompt_tokens + other.prompt_tokens,
            self.completion_tokens + other.completion_tokens,
        )


@dataclass
class Completion:
    """What a non-streaming chat call returns."""

    text: str
    usage: Usage
    model: str
    latency_s: float


# ---------------------------------------------------------------------------
# The client
# ---------------------------------------------------------------------------

@dataclass
class FakeLLMClient:
    """An async client whose surface mirrors a real chat/embedding SDK.

    Parameters
    ----------
    seed:
        Seeds this client's private RNG. Same seed + same call sequence =>
        same latencies and same injected failures.
    base_latency_s / jitter_s:
        Each call sleeps for base_latency_s + U(0, jitter_s).
    rate_limit_rate / server_error_rate:
        Probability of injecting the corresponding failure per call.
    model:
        Purely cosmetic here, but note the real-world distinction: in Azure
        OpenAI you pass a *deployment name*, which is an arbitrary string you
        chose, not the underlying model name. Code that hardcodes "gpt-4o-mini"
        as a deployment name breaks on the first tenant that named it
        differently.
    """

    seed: int = 0
    base_latency_s: float = 0.10
    jitter_s: float = 0.05
    rate_limit_rate: float = 0.0
    server_error_rate: float = 0.0
    model: str = "gpt-4o-mini-deployment"

    # Observability counters. In production these would be metrics/spans.
    calls_started: int = 0
    calls_succeeded: int = 0
    calls_failed: int = 0
    calls_cancelled: int = 0
    total_usage: Usage = field(default_factory=Usage)

    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    # -- internal helpers ---------------------------------------------------

    def _next_latency(self) -> float:
        return self.base_latency_s + self._rng.random() * self.jitter_s

    def _maybe_fail(self) -> None:
        """Draw once against each configured failure rate.

        NOTE: we draw *before* sleeping in some paths and *after* in others in
        real life. Here we always draw after the sleep, which models a server
        that accepted the request, worked on it, then failed. That ordering
        matters for timeout semantics: a request that fails after 5s of work
        has still consumed 5s of your timeout budget.
        """
        r = self._rng.random()
        if r < self.rate_limit_rate:
            # Servers often tell you how long to wait. Honouring this is
            # strictly better than your own backoff guess.
            raise RateLimitError("429 Too Many Requests", retry_after=0.25)
        if r < self.rate_limit_rate + self.server_error_rate:
            raise TransientServerError("503 Service Unavailable")

    @staticmethod
    def _count_tokens(text: str) -> int:
        """Crude token estimate. Real code uses tiktoken; the ratio is roughly
        4 characters per token for English. Never ship this heuristic to a
        billing path — use the usage object the API returns."""
        return max(1, len(text) // 4)

    # -- public API ---------------------------------------------------------

    async def complete(self, prompt: str, *, max_tokens: int = 64) -> Completion:
        """One non-streaming chat completion.

        The `await asyncio.sleep(...)` below is the suspension point. While
        this coroutine is parked there, the event loop is free to run every
        other ready task. That single line is the entire reason async is worth
        anything for I/O-bound work.
        """
        self.calls_started += 1
        started = time.perf_counter()
        try:
            await asyncio.sleep(self._next_latency())
            self._maybe_fail()
        except asyncio.CancelledError:
            # GOTCHA: cancellation is delivered *as an exception at the await
            # point*. If you swallow it, you break cancellation for everyone
            # upstream. We count it and re-raise. Never `except Exception` in a
            # way that catches this — in 3.8+ CancelledError inherits from
            # BaseException specifically to make bare `except Exception` safe,
            # but `except BaseException` is still a trap.
            self.calls_cancelled += 1
            raise
        except LLMError:
            self.calls_failed += 1
            raise

        # Deterministic pseudo-response derived from the prompt, so tests can
        # assert on it.
        digest = hashlib.sha256(prompt.encode()).hexdigest()[:8]
        text = f"[{self.model}] response({digest}) to: {prompt[:48]}"
        usage = Usage(
            prompt_tokens=self._count_tokens(prompt),
            completion_tokens=min(max_tokens, self._count_tokens(text)),
        )
        self.total_usage = self.total_usage + usage
        self.calls_succeeded += 1
        return Completion(
            text=text,
            usage=usage,
            model=self.model,
            latency_s=time.perf_counter() - started,
        )

    async def stream(self, prompt: str, *, n_tokens: int = 12):
        """Token-by-token streaming, exposed as an *async generator*.

        Consume with `async for tok in client.stream(...)`. See 07_streaming.py
        for why async generators need care around cancellation and cleanup.
        """
        self.calls_started += 1
        # Time-to-first-token is a separate latency from per-token latency, and
        # users perceive them very differently. Model them separately.
        await asyncio.sleep(self._next_latency())
        try:
            self._maybe_fail()
        except LLMError:
            self.calls_failed += 1
            raise

        words = f"streamed answer for {prompt[:32]}".split()
        for i in range(n_tokens):
            await asyncio.sleep(0.02)  # inter-token gap
            yield words[i % len(words)] + " "
        self.calls_succeeded += 1

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Batch embedding. Note it takes a LIST — batching at the API level is
        almost always cheaper than N concurrent single-item calls. Concurrency
        is not a substitute for batching; use both, in that order."""
        self.calls_started += 1
        await asyncio.sleep(self._next_latency() + 0.01 * len(texts))
        try:
            self._maybe_fail()
        except LLMError:
            self.calls_failed += 1
            raise
        self.calls_succeeded += 1
        out = []
        for t in texts:
            h = hashlib.sha256(t.encode()).digest()
            out.append([b / 255.0 for b in h[:8]])
        return out

    def stats(self) -> str:
        return (
            f"started={self.calls_started} ok={self.calls_succeeded} "
            f"failed={self.calls_failed} cancelled={self.calls_cancelled} "
            f"tokens={self.total_usage.total_tokens}"
        )


# ---------------------------------------------------------------------------
# A deliberately blocking (synchronous) client — used in 03_blocking_calls.py
# ---------------------------------------------------------------------------

class BlockingLLMClient:
    """Models a *synchronous* SDK, e.g. the non-async variant of a vendor
    library, or a CPU-bound tokeniser, or `requests`.

    Calling this from inside a coroutine without `to_thread` is the single most
    common async bug in LLM codebases. `time.sleep` here stands in for any
    call that does not yield to the event loop.
    """

    def __init__(self, latency_s: float = 0.10) -> None:
        self.latency_s = latency_s

    def complete(self, prompt: str) -> str:
        time.sleep(self.latency_s)  # <-- blocks the whole thread
        return f"blocking response to: {prompt[:32]}"


# ---------------------------------------------------------------------------
# Small helper used across scripts for readable output
# ---------------------------------------------------------------------------

class Timer:
    """Context manager printing wall-clock elapsed time.

    Wall clock is the only number that matters when comparing concurrency
    strategies. Sum-of-latencies tells you nothing about whether you actually
    overlapped anything.
    """

    def __init__(self, label: str) -> None:
        self.label = label
        self.elapsed = 0.0

    def __enter__(self) -> Self:
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        self.elapsed = time.perf_counter() - self._t0
        print(f"  ⏱  {self.label}: {self.elapsed:.3f}s")


def banner(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")
