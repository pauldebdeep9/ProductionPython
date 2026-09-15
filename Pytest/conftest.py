"""
tests/conftest.py — the root of the fixture hierarchy.

WHAT conftest.py IS
-------------------
A module pytest imports automatically for every test in its directory and all
subdirectories. Nothing imports it explicitly; that is the point. It is how
fixtures, hooks, and plugins are made available without an import statement in
every test file.

THE HIERARCHY, and why it matters:

    tests/conftest.py                 <- everything below sees these
      tests/unit/conftest.py          <- unit tests only
      tests/integration/conftest.py   <- integration tests only

A fixture defined in a lower conftest OVERRIDES one of the same name from
above, which is the mechanism for "the same test, against a different backend"
(see tests/contract/).

THE RULES WORTH FOLLOWING:
  1. Put a fixture in the NARROWEST conftest that needs it. A fixture at the
     root is loaded for every test in the suite; if only three tests use it,
     it belongs next to them.
  2. conftest.py is for FIXTURES AND HOOKS, not helper functions. Helpers go
     in an importable module — `from tests.helpers import build_chunk` is
     greppable, whereas a bare name injected by conftest is not.
  3. No test collection logic in a conftest you did not write yourself. It is
     the least discoverable file in the repo.
"""

from __future__ import annotations

import asyncio
import json
import time
import sys
from decimal import Decimal
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.service import (  # noqa: E402
    CORPUS,
    Chunk,
    Config,
    FakeClock,
    InMemoryRetriever,
    RagService,
    ScriptedModel,
    good_response,
)


# ===========================================================================
# 1. SCOPE
# ===========================================================================
# Five scopes, from widest to narrowest:
#
#   session   once per pytest run
#   package   once per package
#   module    once per test file
#   class     once per test class
#   function  once per test  (THE DEFAULT)
#
# THE RULE: default to `function`. Widen only when setup is genuinely
# expensive AND the fixture is immutable or resets itself.
#
# THE FAILURE MODE of a wide scope is test coupling: test A mutates the shared
# object, test B passes only because A ran first, and the suite breaks when
# someone reorders or runs a subset. That bug is expensive to find because the
# failing test is not the one with the bug.


@pytest.fixture(scope="session")
def corpus() -> list[Chunk]:
    """Session-scoped and SAFE, because Chunk is frozen.

    Immutability is what makes a wide scope safe. If this returned a mutable
    list of mutable objects, one test appending to it would change every
    subsequent test's view of the world.
    """
    return list(CORPUS)


@pytest.fixture(scope="session")
def golden_set() -> list[dict[str, object]]:
    """The evaluation cases, loaded once. See tests/eval/.

    Session scope is right here: reading and parsing a golden set on every
    test would dominate the runtime of an eval suite.
    """
    return [
        {"id": "qty_over_receipt", "question": "invoice 88 units billed",
         "groups": ["isc-all"], "expect_disposition": "adjust_quantity",
         "expect_citations": ["c3", "c4"]},
        {"id": "price_match", "question": "PO 1001 unit price",
         "groups": ["isc-all"], "expect_disposition": "approve",
         "expect_citations": ["c2"]},
        {"id": "no_receipt", "question": "goods receipt 55 units",
         "groups": ["isc-all"], "expect_disposition": "approve",
         "expect_citations": ["c4"]},
    ]


# ===========================================================================
# 2. FUNCTION-SCOPED BUILDING BLOCKS
# ===========================================================================

@pytest.fixture
def config() -> Config:
    """Fresh per test. Frozen, but a test may want a DIFFERENT config, and
    `dataclasses.replace` on a per-test instance is cleaner than mutating a
    shared one."""
    return Config()


@pytest.fixture
def clock() -> FakeClock:
    """A controllable clock, so no test ever sleeps for real.

    THE PAYOFF: a test asserting exponential backoff runs in microseconds and
    asserts on the EXACT sleep sequence, which a real sleep cannot do.
    """
    return FakeClock()


@pytest.fixture
def retriever(corpus: list[Chunk]) -> InMemoryRetriever:
    """Note this fixture DEPENDS ON another fixture by naming it as a
    parameter. That is pytest's dependency injection, and it is the feature
    that makes fixtures compose rather than nest."""
    return InMemoryRetriever(corpus)


@pytest.fixture
def model() -> ScriptedModel:
    """Defaults to one good response. Tests that need a specific sequence
    override it — see `make_service` below and tests/unit/test_repair.py."""
    return ScriptedModel([good_response()])


@pytest.fixture
def service(retriever: InMemoryRetriever, model: ScriptedModel,
            config: Config) -> RagService:
    """The assembled system under test.

    Three fixtures composed into one. A test that just needs "a working
    service" asks for `service` and gets it; a test that needs to inspect the
    retriever asks for both and gets THE SAME instance, because pytest caches
    fixtures per scope.
    """
    return RagService(retriever, model, config)


# ===========================================================================
# 3. FACTORY FIXTURES
# ===========================================================================
# A fixture that returns a FUNCTION rather than a value. Use one when the test
# needs to control construction — several instances, or parameters known only
# inside the test.
#
# This is the single most useful fixture pattern and the most under-used.


@pytest.fixture
def make_chunk():
    """Build a Chunk with sensible defaults.

    WHY THIS BEATS A PLAIN HELPER FUNCTION: it can depend on other fixtures
    (tmp_path, a database handle, a counter) without the test passing them
    through. And it can register cleanup for everything it built.
    """
    created: list[Chunk] = []

    def _make(chunk_id: str = "c1", *, doc_id: str = "doc-1",
              text: str = "some text", groups: set[str] | None = None,
              score: float = 1.0) -> Chunk:
        c = Chunk(chunk_id, doc_id, text,
                  frozenset(groups or {"isc-all"}), score)
        created.append(c)
        return c

    yield _make
    # Cleanup would go here. Recording what was built also lets a test assert
    # on it, which is occasionally exactly what you want.
    created.clear()


@pytest.fixture
def make_service(config: Config):
    """Build a service with a specific model script.

    THE PATTERN THIS REPLACES is a pile of near-identical fixtures:
    `service_with_bad_json`, `service_with_two_repairs`,
    `service_with_low_confidence`... each one a separate thing to maintain.
    One factory covers all of them and reads better at the call site.
    """
    built: list[RagService] = []

    def _make(responses: list[str] | None = None, *,
              corpus: list[Chunk] | None = None,
              trim: bool = True,
              config_overrides: dict[str, object] | None = None) -> RagService:
        import dataclasses
        cfg = (dataclasses.replace(config, **config_overrides)  # type: ignore[arg-type]
               if config_overrides else config)
        svc = RagService(
            InMemoryRetriever(corpus, trim=trim),
            ScriptedModel(responses or [good_response()]),
            cfg,
        )
        built.append(svc)
        return svc

    return _make


# ===========================================================================
# 4. HOOKS
# ===========================================================================

def pytest_collection_modifyitems(config: pytest.Config,
                                  items: list[pytest.Item]) -> None:
    """Auto-mark tests by their directory.

    WHY: relying on every author to remember `@pytest.mark.unit` fails within
    a month. Deriving the marker from location is mechanical and cannot drift.

    Directory layout then becomes meaningful:
        tests/unit/         -> unit
        tests/integration/  -> integration
        tests/eval/         -> eval + slow
        tests/contract/     -> contract
    """
    for item in items:
        path = str(item.fspath)
        if "/unit/" in path:
            item.add_marker(pytest.mark.unit)
        elif "/integration/" in path:
            item.add_marker(pytest.mark.integration)
        elif "/eval/" in path:
            item.add_marker(pytest.mark.eval)
            item.add_marker(pytest.mark.slow)
        elif "/contract/" in path:
            item.add_marker(pytest.mark.contract)


@pytest.fixture(autouse=True)
def _no_real_sleeping(monkeypatch: pytest.MonkeyPatch, request) -> None:
    """AUTOUSE: applies to every test without being requested.

    THIS ONE EARNS IT. A test that calls `asyncio.sleep(30)` for real turns a
    2-second suite into a 5-minute one, and nobody notices which test did it.
    Failing loudly is better.

    USE AUTOUSE SPARINGLY. It is invisible at the call site — a test that
    behaves oddly gives no hint that a fixture it never mentioned is involved.
    Reserve it for guardrails like this one, and for resetting global state
    that would otherwise leak between tests.

    Note the escape hatch: a test marked `slow` may sleep, because integration
    tests sometimes genuinely need to.
    """
    if request.node.get_closest_marker("slow"):
        return

    real_sleep = asyncio.sleep
    LIMIT = 0.05

    async def guarded(seconds: float, *args, **kwargs):
        # THE BUG IN THE FIRST VERSION OF THIS FIXTURE, worth keeping as a
        # comment because it is instructive: it raised on the REQUESTED
        # duration. That failed three legitimate tests in test_04_async.py —
        # tests that ask for `sleep(10)` and then cancel or time out
        # immediately. They never wait; they use a long sleep as "block until
        # something happens to me", which is the idiomatic way to test
        # cancellation.
        #
        # What we actually care about is ELAPSED time, not requested time. So
        # the guard now measures. A sleep that is cancelled costs nothing and
        # is allowed; a sleep that actually completes and burns wall-clock
        # time is what we want to catch.
        #
        # The general lesson: a guardrail that fires on INTENT rather than on
        # EFFECT will block correct code, and people will disable it.
        started = time.perf_counter()
        try:
            return await real_sleep(seconds, *args, **kwargs)
        finally:
            elapsed = time.perf_counter() - started
            if elapsed > LIMIT:
                raise AssertionError(
                    f"test slept for {elapsed:.2f}s of real time. Inject a "
                    f"FakeClock instead of sleeping, or mark the test "
                    f"@pytest.mark.slow."
                )

    monkeypatch.setattr(asyncio, "sleep", guarded)


# ===========================================================================
# 5. ASSERTION HELPERS
# ===========================================================================

def pytest_assertrepr_compare(op: str, left: object, right: object):
    """Custom failure output for domain types.

    pytest's default diff for two Proposals is a wall of dataclass repr. This
    makes the failure message name the FIELD that differs, which is the
    difference between a five-second and a two-minute diagnosis.

    Worth writing for the two or three types that appear in the most
    assertions. Not worth writing for everything.
    """
    from rag.service import Proposal

    if isinstance(left, Proposal) and isinstance(right, Proposal) and op == "==":
        lines = ["Proposal comparison failed:"]
        for f in ("disposition", "confidence", "reason", "evidence_ids"):
            lv, rv = getattr(left, f), getattr(right, f)
            mark = "  " if lv == rv else "->"
            lines.append(f"{mark} {f}: {lv!r} != {rv!r}"
                         if lv != rv else f"{mark} {f}: {lv!r}")
        return lines
    return None


def pytest_addoption(parser: pytest.Parser) -> None:
    """Custom CLI options.

    MUST live in the ROOT conftest (or a plugin) — pytest reads options before
    descending into subdirectories, so an option defined in a nested conftest
    is not registered in time and you get "unrecognized arguments".
    """
    parser.addoption(
        "--model-tier", action="store", default="cheap",
        choices=("cheap", "all"),
        help="Which model deployments the parametrised tier tests run against.",
    )
    parser.addoption(
        "--run-live", action="store_true", default=False,
        help="Run tests that call a REAL model endpoint. Costs money.",
    )
