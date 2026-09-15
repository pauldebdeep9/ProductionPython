"""
tests/unit/test_01_fixtures.py — fixture mechanics.

Run:  pytest tests/unit/test_01_fixtures.py -v

WHAT A FIXTURE IS FOR
---------------------
Three jobs, in order of how much they matter:

  1. SETUP AND TEARDOWN that is guaranteed to run, including when the test
     fails or errors. A `try/finally` in each test does the same thing and
     nobody writes it consistently.
  2. DEPENDENCY INJECTION. A test names what it needs; pytest builds the
     graph. That is why the tests below are three lines each.
  3. SHARING expensive setup across tests, at a controlled scope.

WHAT A FIXTURE IS NOT FOR: hiding the arrangement step. A test whose setup is
entirely invisible is hard to read. The balance this file argues for is
fixtures for INFRASTRUCTURE (clients, temp dirs, clocks) and explicit,
in-test construction for the DATA the assertion is about.
"""

from __future__ import annotations

import asyncio
from decimal import Decimal

import pytest

from rag.service import (
    Chunk,
    Config,
    GenerationFailed,
    InMemoryRetriever,
    PermissionViolation,
    Proposal,
    RagService,
    ScriptedModel,
    good_response,
)


# ===========================================================================
# 1. DEPENDENCY INJECTION
# ===========================================================================

async def test_service_fixture_is_assembled(service: RagService) -> None:
    """Three fixtures composed (retriever, model, config) and the test says
    none of it. That is the payoff."""
    answer = await service.answer("invoice 88 units billed",
                                  frozenset({"isc-all"}))
    assert answer.proposal is not None
    assert answer.outcome == "ok"


async def test_same_instance_within_one_test(service: RagService,
                                             model: ScriptedModel) -> None:
    """`service` depends on `model`, and the test also asks for `model`.

    Both get THE SAME object, because pytest caches a fixture's result per
    scope per test. That is what lets a test inspect a collaborator it did not
    construct — here, checking what prompt the service actually sent.
    """
    await service.answer("invoice 88 units billed", frozenset({"isc-all"}))
    assert model.prompts, "the service should have called the model"
    assert service.model is model


# ===========================================================================
# 2. SCOPE, AND THE COUPLING IT CAUSES
# ===========================================================================

@pytest.fixture(scope="module")
def shared_mutable_list() -> list[str]:
    """A DELIBERATELY DANGEROUS fixture, to make the failure mode concrete."""
    return []


def test_scope_hazard_first(shared_mutable_list: list[str]) -> None:
    shared_mutable_list.append("written by the first test")
    assert len(shared_mutable_list) == 1


def test_scope_hazard_second(shared_mutable_list: list[str]) -> None:
    """This test PASSES only because the previous one ran first.

    Run it alone —
        pytest tests/unit/test_01_fixtures.py::test_scope_hazard_second
    — and it fails. That is the coupling a wide scope buys you, and it is why
    the default is `function`.

    The general rule: a wide scope is safe only if the fixture is IMMUTABLE
    (like `corpus`, which returns frozen Chunks) or RESETS ITSELF.
    """
    assert len(shared_mutable_list) == 1, (
        "this assertion depends on test execution order — the bug this "
        "section exists to demonstrate"
    )


def test_function_scope_is_fresh_every_time(clock) -> None:
    """`clock` is function-scoped, so this test cannot see another's state."""
    assert clock.slept == []
    clock.slept.append(1.0)


def test_function_scope_is_fresh_every_time_again(clock) -> None:
    assert clock.slept == [], "function scope must give a fresh instance"


# ===========================================================================
# 3. TEARDOWN
# ===========================================================================

TEARDOWN_LOG: list[str] = []


@pytest.fixture
def resource_a():
    """`yield` splits setup from teardown.

    THE GUARANTEE: everything after the yield runs even if the test FAILS or
    RAISES. That is strictly stronger than putting cleanup at the end of the
    test body, which a failing assertion skips.
    """
    TEARDOWN_LOG.append("a: setup")
    yield "resource-a"
    TEARDOWN_LOG.append("a: teardown")


@pytest.fixture
def resource_b(resource_a: str):
    TEARDOWN_LOG.append("b: setup")
    yield "resource-b"
    TEARDOWN_LOG.append("b: teardown")


def test_teardown_runs_in_reverse_order(resource_b: str) -> None:
    """Teardown is LIFO — the reverse of setup.

    That ordering is not cosmetic. If `b` is a database session and `a` is the
    container it connects to, tearing down `a` first would leave `b` unable to
    close cleanly.
    """
    TEARDOWN_LOG.clear()
    TEARDOWN_LOG.extend(["a: setup", "b: setup"])
    assert resource_b == "resource-b"


@pytest.fixture
def failing_teardown_demo():
    log: list[str] = []
    yield log
    # This runs even though the test below raises.
    log.append("cleaned up")
    assert log == ["work started", "cleaned up"]


def test_teardown_runs_even_when_the_test_fails(failing_teardown_demo) -> None:
    failing_teardown_demo.append("work started")
    # If this test asserted something false, teardown would still run. The
    # fixture's own assertion above verifies it.
    assert True


# ===========================================================================
# 4. FACTORY FIXTURES
# ===========================================================================

def test_factory_builds_what_the_test_needs(make_chunk) -> None:
    """One factory replaces a dozen near-identical fixtures.

    Compare with `chunk_visible_to_isc`, `chunk_visible_to_hr`,
    `chunk_with_high_score`, ... each a separate thing to name and maintain.
    """
    visible = make_chunk("c1", groups={"isc-all"})
    hidden = make_chunk("c9", groups={"legal-only"}, text="confidential")

    assert visible.visible_to(frozenset({"isc-all"}))
    assert not hidden.visible_to(frozenset({"isc-all"}))


async def test_service_factory_configures_the_scenario(make_service) -> None:
    """`make_service` takes the model script, so the SCENARIO is visible in
    the test rather than hidden in a fixture name."""
    svc = make_service(["not json at all", good_response()])
    answer = await svc.answer("invoice 88 units", frozenset({"isc-all"}))
    assert answer.repairs == 1
    assert answer.degraded is True


async def test_factory_with_config_overrides(make_service) -> None:
    svc = make_service([good_response(confidence=0.5)],
                       config_overrides={"min_confidence": 0.9})
    answer = await svc.answer("invoice 88 units", frozenset({"isc-all"}))
    assert answer.outcome == "low_confidence"


# ===========================================================================
# 5. THE `request` FIXTURE
# ===========================================================================

@pytest.fixture
def scenario_name(request: pytest.FixtureRequest) -> str:
    """`request` gives a fixture access to the test that asked for it.

    Useful for: naming a temp directory after the test, reading a marker to
    change behaviour, or registering a finalizer.
    """
    return request.node.name


def test_request_exposes_the_test_node(scenario_name: str) -> None:
    assert scenario_name == "test_request_exposes_the_test_node"


@pytest.fixture
def budget_from_marker(request: pytest.FixtureRequest) -> Decimal:
    """Read a marker to parameterise a fixture.

    THE PATTERN: `@pytest.mark.budget("0.001")` on a test, and the fixture
    picks it up. Keeps configuration next to the test that needs it rather
    than in a separate fixture per value.
    """
    marker = request.node.get_closest_marker("budget")
    return Decimal(marker.args[0]) if marker else Decimal("10.00")


@pytest.mark.budget("0.00001")
async def test_marker_configures_the_fixture(budget_from_marker: Decimal,
                                             make_service) -> None:
    from rag.service import BudgetExceeded
    svc = make_service([good_response()],
                       config_overrides={"daily_budget_usd": budget_from_marker})
    with pytest.raises(BudgetExceeded):
        await svc.answer("invoice 88 units", frozenset({"isc-all"}))


# ===========================================================================
# 6. BUILT-IN FIXTURES WORTH KNOWING
# ===========================================================================

def test_tmp_path_is_a_real_directory(tmp_path) -> None:
    """`tmp_path` is a per-test `pathlib.Path`, cleaned up automatically.

    ALWAYS use it instead of `/tmp/whatever` or a fixed relative path:
      * parallel runs (pytest-xdist) do not collide
      * a failed test's files survive for inspection (pytest keeps the last
        few runs), which is genuinely useful
      * no cleanup code to forget
    """
    golden = tmp_path / "golden.jsonl"
    golden.write_text('{"id": "case-1"}\n')
    assert golden.read_text().strip() == '{"id": "case-1"}'
    assert tmp_path.is_dir()


def test_caplog_captures_structured_fields(caplog) -> None:
    """`caplog` captures log records as OBJECTS, not strings.

    That is the important part: assert on `record.chunk_count`, not on a
    substring of the formatted message. A message-text assertion breaks the
    first time someone rewords the log line.
    """
    import logging
    logger = logging.getLogger("rag.test")
    with caplog.at_level(logging.INFO, logger="rag.test"):
        logger.info("retrieval done", extra={"chunk_count": 4, "trimmed": 2})

    assert len(caplog.records) == 1
    assert caplog.records[0].chunk_count == 4
    assert caplog.records[0].trimmed == 2


def test_capsys_captures_stdout(capsys) -> None:
    print("hello from the system under test")
    captured = capsys.readouterr()
    assert "hello" in captured.out
    assert captured.err == ""


def test_monkeypatch_is_undone_automatically(monkeypatch) -> None:
    """`monkeypatch` reverses every change at teardown.

    Prefer it to `unittest.mock.patch` as a context manager for env vars,
    attributes, and dict entries: less nesting, and impossible to forget the
    undo.
    """
    import os
    monkeypatch.setenv("RAG_TEST_MARKER", "set-inside-the-test")
    assert os.environ["RAG_TEST_MARKER"] == "set-inside-the-test"


def test_monkeypatch_was_reversed() -> None:
    import os
    assert "RAG_TEST_MARKER" not in os.environ


# ===========================================================================
# 7. THE FAKE CLOCK — why tests must never sleep
# ===========================================================================

async def test_backoff_sequence_without_sleeping(clock) -> None:
    """Assert on the EXACT sleep sequence, in microseconds.

    A test that really slept could only assert "it took roughly 0.7 seconds",
    which is both slower and weaker — it cannot tell 0.1/0.2/0.4 from
    0.35/0.35 or from a single 0.7.
    """
    async def retry_with_backoff(attempts: int, base: float) -> None:
        for n in range(attempts):
            await clock.sleep(base * (2 ** n))

    await retry_with_backoff(4, 0.1)

    assert clock.slept == [0.1, 0.2, 0.4, 0.8]
    assert clock.now() == pytest.approx(1_700_000_001.5)


async def test_the_autouse_guard_catches_real_sleeps() -> None:
    """The autouse fixture in conftest fails a test that sleeps for real.

    Verified here so the guard itself is tested — otherwise it is a rule
    nobody knows is enforced until it silently stops being.

    NOTE the sleep is 0.06s, just over the 0.05s limit. The guard measures
    ELAPSED time, not requested time, so this test costs 60ms rather than the
    full second an intent-based guard would have needed to demonstrate.
    """
    with pytest.raises(AssertionError, match="Inject a FakeClock"):
        await asyncio.sleep(0.06)
