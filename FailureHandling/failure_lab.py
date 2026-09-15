"""
failure_lab.py — shared harness for the failure-handling tutorial.

WHAT THIS PROVIDES
------------------
  * A realistic exception taxonomy (the thing your retry logic dispatches on).
  * A FaultInjector with deterministic, seeded, *scriptable* failure sequences —
    so "fails twice then succeeds" is a test you can write, not a coin flip.
  * Fake services: an LLM, a tool/API backend, and a key-value store.
  * A MetricsCollector, because half of failure handling is knowing it happened.

DESIGN NOTE ON DETERMINISM
--------------------------
Random failure injection produces flaky tutorials and flaky tests. Everything
here supports two modes:

    FaultInjector(script=[Fail(RateLimit), Fail(RateLimit), Ok()])   # exact
    FaultInjector(rate=0.3, seed=42)                                 # seeded

Use the scripted form whenever you are asserting on behaviour. Use the seeded
form only when you want to observe aggregate behaviour under load.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum

# ===========================================================================
# 1. THE TAXONOMY
# ===========================================================================
# This is the single most important design artefact in failure handling.
# Everything downstream — retry, fallback, circuit breaking, alerting — is a
# dispatch on these types. If your taxonomy is wrong, no amount of clever
# backoff saves you.
#
# The three axes that actually matter:
#   RETRYABLE?   will the identical request plausibly succeed later?
#   REPAIRABLE?  would a *different* request succeed? (distinct from retry!)
#   SAFE?        might the side effect have already happened?
#
# Most codebases collapse these into one boolean and get it wrong.


class Disposition(Enum):
    """What to do with a failure. Note REPAIR is separate from RETRY."""

    RETRY = "retry"          # same request, later
    REPAIR = "repair"        # different request (truncate, re-prompt, re-chunk)
    FALLBACK = "fallback"    # different provider/model/path
    FAIL_FAST = "fail_fast"  # nothing will help; surface it
    ESCALATE = "escalate"    # a human must act (auth, quota, config)


class AppError(Exception):
    """Base for every error this application raises.

    WHY WRAP VENDOR EXCEPTIONS AT ALL:
      * Vendor SDKs change their exception hierarchies between versions.
      * You want ONE place that decides retryability, not `except` clauses
        scattered across 40 call sites naming openai.* types.
      * Swapping Azure OpenAI for Bedrock should not require touching retry
        logic. The adapter translates; the core dispatches on your types.

    Always chain with `raise AppError(...) from original` — losing the original
    traceback is how you end up debugging a wrapper instead of a bug.
    """

    disposition: Disposition = Disposition.FAIL_FAST
    # Whether the side effect may have already occurred. Governs whether a
    # retry is SAFE, which is a different question from whether it is USEFUL.
    side_effect_uncertain: bool = False

    def __init__(self, message: str, *, retry_after: float | None = None,
                 context: dict | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after
        self.context = context or {}


# --- Transient: same request, later, plausibly works -----------------------

class TransientError(AppError):
    disposition = Disposition.RETRY


class RateLimitError(TransientError):
    """429. Retryable, and usually carries server guidance."""


class ServiceUnavailable(TransientError):
    """503/502/504. Retryable."""


class UpstreamTimeout(TransientError):
    """The request timed out. RETRYABLE but SIDE-EFFECT UNCERTAIN: the server
    may have processed it and we simply lost the response. This distinction is
    the entire reason idempotency keys exist (see 03)."""

    side_effect_uncertain = True


class ConnectionLost(TransientError):
    side_effect_uncertain = True


# --- Permanent: identical request will never work --------------------------

class PermanentError(AppError):
    disposition = Disposition.FAIL_FAST


class ValidationError(PermanentError):
    """Malformed request. Retrying is pure waste."""


class ContentFiltered(PermanentError):
    """400 content_filter. The same prompt is rejected forever."""


class NotFoundError(PermanentError):
    pass


# --- Repairable: a DIFFERENT request would work ----------------------------

class RepairableError(AppError):
    """The critical middle category almost every tutorial omits.

    A retry sends the same bytes and fails identically. A REPAIR changes the
    request — truncate the context, re-chunk, relax the schema, re-prompt with
    the parse error — and can succeed. Conflating these produces either
    pointless retry storms or premature give-ups.
    """

    disposition = Disposition.REPAIR


class ContextLengthExceeded(RepairableError):
    """Truncate or re-chunk, then it fits."""


class SchemaValidationError(RepairableError):
    """The model returned JSON that does not match the schema. Re-prompt WITH
    the validation error attached — that is a repair, not a retry."""


class TruncatedOutput(RepairableError):
    """finish_reason == 'length'. Raise max_tokens or continue the generation."""


# --- Escalate: a human must intervene --------------------------------------

class AuthError(AppError):
    disposition = Disposition.ESCALATE


class QuotaExhausted(AppError):
    """Not a 429 burst — the monthly/committed quota is gone. Backoff does not
    fix this and retrying it wastes the deadline. Page someone."""

    disposition = Disposition.ESCALATE


# --- Poison: this specific ITEM is broken ----------------------------------

class PoisonItemError(AppError):
    """This input will fail forever regardless of provider or timing.

    Distinct from PermanentError in intent: a poison item should be QUARANTINED
    and the batch should continue, not fail the run. See 07.
    """

    disposition = Disposition.FAIL_FAST


def classify(exc: BaseException) -> Disposition:
    """Single source of truth for 'what do we do with this?'.

    NOTE the CancelledError guard. Cancellation is never retried — doing so
    turns a prompt shutdown into a hang. It is not an application error at all.
    """
    if isinstance(exc, asyncio.CancelledError):
        raise AssertionError("cancellation must propagate, never be classified")
    if isinstance(exc, AppError):
        return exc.disposition
    if isinstance(exc, TimeoutError):
        return Disposition.RETRY
    return Disposition.FAIL_FAST


# ===========================================================================
# 2. FAULT INJECTION
# ===========================================================================

@dataclass
class Ok:
    """Scripted outcome: succeed, optionally after `latency`."""

    latency: float = 0.0


@dataclass
class Fail:
    """Scripted outcome: raise `error_cls` after `latency`."""

    error_cls: type[AppError]
    latency: float = 0.0
    message: str = ""
    retry_after: float | None = None


@dataclass
class Hang:
    """Scripted outcome: sleep far longer than any sane timeout.

    Models the nastiest real failure — not an error, but a socket that accepts
    your request and never answers. Only a timeout saves you.
    """

    latency: float = 30.0


class FaultInjector:
    """Deterministic or seeded failure injection.

    Scripted mode consumes outcomes in order; once exhausted it repeats the
    final outcome forever (so `[Fail, Fail, Ok]` means "flaky then healthy").
    """

    def __init__(
        self,
        *,
        script: list | None = None,
        rate: float = 0.0,
        error_cls: type[AppError] = ServiceUnavailable,
        seed: int = 0,
        base_latency: float = 0.01,
    ) -> None:
        self.script = list(script) if script else None
        self.rate = rate
        self.error_cls = error_cls
        self.base_latency = base_latency
        self._rng = random.Random(seed)
        self._i = 0
        self.calls = 0

    async def maybe_fail(self) -> None:
        self.calls += 1

        if self.script is not None:
            outcome = self.script[min(self._i, len(self.script) - 1)]
            self._i += 1
            if isinstance(outcome, Hang):
                await asyncio.sleep(outcome.latency)
                return
            await asyncio.sleep(outcome.latency)
            if isinstance(outcome, Fail):
                raise outcome.error_cls(
                    outcome.message or f"injected {outcome.error_cls.__name__}",
                    retry_after=outcome.retry_after,
                )
            return

        await asyncio.sleep(self.base_latency)
        if self._rng.random() < self.rate:
            raise self.error_cls(f"injected {self.error_cls.__name__}")


# ===========================================================================
# 3. METRICS
# ===========================================================================

class Metrics:
    """Minimal counter/histogram sink.

    The specific counters here are the ones that matter for failure handling,
    and most teams are missing at least two of them:
      attempts vs requests  -> retry AMPLIFICATION (see 02)
      outcomes by class     -> is it one dependency or everything?
      latency by outcome    -> failures are often FASTER than successes, which
                               makes a p50 latency dashboard look great during
                               a total outage
    """

    def __init__(self) -> None:
        self.counters: Counter[str] = Counter()
        self.latencies: dict[str, list[float]] = defaultdict(list)

    def incr(self, name: str, n: int = 1) -> None:
        self.counters[name] += n

    def observe(self, name: str, value: float) -> None:
        self.latencies[name].append(value)

    def pct(self, name: str, p: float) -> float:
        vals = sorted(self.latencies.get(name, []))
        if not vals:
            return 0.0
        k = min(len(vals) - 1, round((len(vals) - 1) * p))
        return vals[k]

    def report(self, *names: str) -> str:
        keys = names or sorted(self.counters)
        return "  ".join(f"{k}={self.counters[k]}" for k in keys)

    def amplification(self) -> float:
        """attempts / requests. The number nobody tracks and everybody needs.

        At 1.0 you are not retrying. At 3.0 you are sending 3x the load you
        think you are — and during an outage that is exactly when the
        dependency can least afford it.
        """
        req = self.counters.get("requests", 0)
        att = self.counters.get("attempts", 0)
        return att / req if req else 0.0


# ===========================================================================
# 4. FAKE SERVICES
# ===========================================================================

@dataclass
class Completion:
    text: str
    finish_reason: str = "stop"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = "gpt-4o-mini-deployment"


class FakeLLM:
    """An LLM endpoint with injectable faults and LLM-specific failure modes.

    `respond_with` lets a test force a particular *content* failure — malformed
    JSON, a refusal, truncation — which are the failures that matter for GenAI
    and which no generic HTTP retry library knows anything about.
    """

    def __init__(
        self,
        name: str = "gpt-4o-mini",
        injector: FaultInjector | None = None,
        *,
        cost_per_1k: float = 0.00015,
        respond_with: list[str] | None = None,
    ) -> None:
        self.name = name
        self.injector = injector or FaultInjector()
        self.cost_per_1k = cost_per_1k
        self.respond_with = list(respond_with) if respond_with else None
        self._i = 0
        self.calls = 0
        self.total_cost = 0.0

    async def complete(self, prompt: str, *, max_tokens: int = 256) -> Completion:
        self.calls += 1
        await self.injector.maybe_fail()

        prompt_tokens = max(1, len(prompt) // 4)

        if self.respond_with is not None:
            text = self.respond_with[min(self._i, len(self.respond_with) - 1)]
            self._i += 1
        else:
            digest = hashlib.sha256(prompt.encode()).hexdigest()[:6]
            text = json.dumps({"answer": f"resolved-{digest}", "confidence": 0.82})

        finish = "length" if len(text) // 4 >= max_tokens else "stop"
        completion_tokens = min(max_tokens, max(1, len(text) // 4))
        self.total_cost += (prompt_tokens + completion_tokens) / 1000 * self.cost_per_1k

        return Completion(
            text=text,
            finish_reason=finish,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            model=self.name,
        )


class FakeToolAPI:
    """A backend system (ERP, SAP, a REST service) an agent calls.

    Distinguishes READ tools (safe to retry freely) from WRITE tools (retry
    only with an idempotency key). That distinction is enforced here rather
    than left to convention, because convention loses.
    """

    def __init__(self, name: str, injector: FaultInjector | None = None) -> None:
        self.name = name
        self.injector = injector or FaultInjector()
        self.writes: dict[str, dict] = {}       # idempotency_key -> record
        self.write_attempts = 0
        self.duplicate_writes = 0

    async def read(self, key: str) -> dict:
        await self.injector.maybe_fail()
        return {"key": key, "value": f"data-for-{key}", "source": self.name}

    async def write(self, payload: dict, *, idempotency_key: str | None = None) -> dict:
        """Simulates a write that MAY succeed server-side and then lose the
        response — the case that makes naive retry dangerous."""
        self.write_attempts += 1

        if idempotency_key is not None and idempotency_key in self.writes:
            # Correct server behaviour: return the ORIGINAL result, do not
            # apply the write twice.
            return {**self.writes[idempotency_key], "deduplicated": True}

        await self.injector.maybe_fail()   # may raise AFTER we would have written

        record = {"id": f"rec-{self.write_attempts}", **payload}
        if idempotency_key is not None:
            self.writes[idempotency_key] = record
        else:
            self.duplicate_writes += 1     # no key => no dedup possible
        return record


class FlakyWriteAPI(FakeToolAPI):
    """A write API that applies the write and THEN fails.

    This is the specific, nasty case: the side effect happened, but you got an
    error. Without an idempotency key, your retry duplicates it.
    """

    def __init__(self, name: str, fail_first_n: int = 1) -> None:
        super().__init__(name)
        self.fail_first_n = fail_first_n
        self.applied: list[dict] = []

    async def write(self, payload: dict, *, idempotency_key: str | None = None) -> dict:
        self.write_attempts += 1

        if idempotency_key is not None and idempotency_key in self.writes:
            return {**self.writes[idempotency_key], "deduplicated": True}

        record = {"id": f"rec-{self.write_attempts}", **payload}
        self.applied.append(record)          # <-- the side effect HAPPENS
        if idempotency_key is not None:
            self.writes[idempotency_key] = record

        if self.write_attempts <= self.fail_first_n:
            # ...and then the response is lost.
            raise UpstreamTimeout("response lost after the write was applied")
        return record


# ===========================================================================
# 5. UTILITIES
# ===========================================================================

class Timer:
    def __init__(self, label: str) -> None:
        self.label = label
        self.elapsed = 0.0

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        self.elapsed = time.perf_counter() - self._t0
        print(f"  ⏱  {self.label}: {self.elapsed:.3f}s")


def banner(title: str) -> None:
    print(f"\n{'=' * 74}\n{title}\n{'=' * 74}")


def section(title: str) -> None:
    print(f"\n  --- {title} ---")


@dataclass
class Attempt:
    """One attempt record. Collecting these makes retry behaviour inspectable
    instead of a mystery — and makes it assertable in tests."""

    n: int
    outcome: str
    error: str | None = None
    delay_before: float = 0.0
    elapsed: float = 0.0


@dataclass
class CallResult:
    """The result of a resilient call, with its full attempt history.

    Returning the history rather than logging it means callers can assert on
    it, and means a single log line can carry the whole retry story instead of
    five interleaved lines you have to reassemble.
    """

    value: object = None
    attempts: list[Attempt] = field(default_factory=list)
    succeeded: bool = False
    final_error: BaseException | None = None

    def summary(self) -> str:
        path = " -> ".join(
            a.outcome if a.error is None else f"{a.outcome}({a.error})"
            for a in self.attempts
        )
        return f"{len(self.attempts)} attempt(s): {path}"
