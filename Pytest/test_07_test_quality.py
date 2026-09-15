"""
tests/unit/test_07_test_quality.py — testing the tests.

Run:  pytest tests/unit/test_07_test_quality.py -v
      pytest --cov=rag --cov-report=term-missing

THE QUESTION NOBODY ASKS
------------------------
"Do we have tests?" is easy and useless. "Would our tests notice if the code
were wrong?" is the one that matters, and it has a mechanical answer:
BREAK THE CODE ON PURPOSE AND SEE IF THE SUITE GOES RED.

That is mutation testing. `mutmut` and `cosmic-ray` automate it; this file
does it by hand, in-process, so the mechanism is visible and so the tutorial's
own claims are demonstrated rather than asserted.

WHY IT MATTERS MORE THAN COVERAGE: coverage tells you a line EXECUTED. It
tells you nothing about whether anything was CHECKED. A test that calls every
function and asserts nothing achieves 100% coverage and catches zero bugs.
Part 1 demonstrates this concretely.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import re
from pathlib import Path
from typing import Callable

import pytest

from rag.service import (
    CORPUS,
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
# 1. COVERAGE LIES
# ===========================================================================

def test_100_percent_coverage_with_zero_assertions() -> None:
    """A test that executes every branch of `parse` and checks NOTHING.

    Coverage tooling reports these lines as covered. Every mutation of `parse`
    would leave this test green. It is worse than no test, because it makes
    the coverage number say the code is protected.

    THE SMELL TO GREP FOR: a test whose body contains no `assert` and no
    `pytest.raises`. It is nearly always either this, or a test that was
    abandoned half-written.
    """
    for raw in [good_response(), "not json", "[1,2]",
                '{"disposition": "bogus"}',
                '{"disposition": "approve", "confidence": 2.0}']:
        with contextlib.suppress(GenerationFailed):
            RagService.parse(raw)
    # No assertion. 100% coverage of the parse function. Zero bugs caught.


def test_assertion_strength_matters_more_than_coverage() -> None:
    """The same code path, three assertion strengths.

    WEAK    `is not None` — passes for any Proposal at all
    BETTER  the disposition — catches a wrong enum
    STRONG  every field — catches a wrong confidence, reason, or citation set

    Coverage is identical for all three. Mutation-resistance is not.
    """
    proposal = RagService.parse(good_response())

    assert proposal is not None                       # weak
    assert proposal.disposition == "adjust_quantity"  # better
    assert proposal == Proposal(                      # strong
        disposition="adjust_quantity",
        confidence=0.84,
        reason="Invoice billed 402 against a 400 unit receipt.",
        evidence_ids=("c3", "c4"),
    )


# ===========================================================================
# 2. MUTATION TESTING, IN PROCESS
# ===========================================================================

@contextlib.contextmanager
def mutate(target: type, attr: str, replacement):
    """Temporarily replace an attribute, restoring it afterwards.

    In a real repo you would run `mutmut run` rather than this. Doing it
    inline lets the tutorial PROVE its own claims about which tests bite,
    which is the same discipline the other tutorials in this series applied
    to their assertions.
    """
    original = getattr(target, attr)
    setattr(target, attr, replacement)
    try:
        yield
    finally:
        setattr(target, attr, original)


async def _run_the_security_test() -> bool:
    """The assertion from test_03, extracted so a mutation can be run against
    it. Returns True if the test PASSES."""
    svc = RagService(InMemoryRetriever(trim=False), ScriptedModel([good_response()]))
    try:
        await svc.answer("settlement terms confidential", frozenset({"isc-all"}))
    except PermissionViolation:
        return True          # the test expects this
    return False


async def test_the_permission_check_is_load_bearing() -> None:
    """MUTATION: remove the permission assertion from the service.

    If the suite still passes, the check is untested — which for a security
    invariant is the most expensive possible gap.

    THIS IS THE TEST TO WRITE FOR EVERY SECURITY CONTROL: not "does it work",
    but "would we notice if it stopped".
    """
    assert await _run_the_security_test(), "baseline: the check works"

    # Mutate `visible_to` so every chunk claims to be visible — the exact
    # shape of a bug someone could introduce while "simplifying" the ACL.
    with mutate(Chunk, "visible_to", lambda self, groups: True):
        survived = await _run_the_security_test()

    assert not survived, (
        "MUTATION SURVIVED: the permission check was disabled and the test "
        "still passed. The security assertion is not load-bearing."
    )


async def test_the_repair_loop_bound_is_load_bearing() -> None:
    """MUTATION: make max_repairs effectively unbounded.

    Verifies that a test actually pins the bound, rather than merely
    exercising the loop.
    """
    async def repairs_are_bounded() -> bool:
        svc = RagService(InMemoryRetriever(),
                         ScriptedModel(["bad"]),
                         Config(max_repairs=2))
        answer = await svc.answer("invoice 88 units", frozenset({"isc-all"}))
        return answer.repairs == 3 and answer.proposal is None

    assert await repairs_are_bounded(), "baseline"

    svc = RagService(InMemoryRetriever(), ScriptedModel(["bad"]),
                     Config(max_repairs=50))
    answer = await svc.answer("invoice 88 units", frozenset({"isc-all"}))
    assert answer.repairs == 51, (
        "changing max_repairs must change the observed behaviour — if it does "
        "not, no test is actually pinning the bound"
    )


@pytest.mark.parametrize("overrides,description", [
    pytest.param({"min_confidence": 0.0}, "confidence floor removed",
                 id="confidence_floor"),
    pytest.param({"top_k": 0}, "retrieval returns nothing", id="top_k_zero"),
])
async def test_config_mutations_are_detected(overrides: dict,
                                             description: str) -> None:
    """Each config value should CHANGE OBSERVABLE BEHAVIOUR.

    A config field that can be mutated with no test noticing is either
    untested or dead. Both are worth knowing — the second is a deletion
    opportunity, which is the cheapest maintenance there is.

    A LESSON FROM WRITING THIS TEST. The first version used
    `mutate(Config, "min_confidence", 0.0)` to patch the CLASS ATTRIBUTE, and
    the mutation "survived" — behaviour was identical.

    The reason is not a gap in the tests. `@dataclass` bakes field defaults
    into the generated `__init__` signature at CLASS CREATION time, so
    `Config()` never reads the class attribute afterwards. Patching it does
    nothing.

    That is worth knowing well beyond this test: monkeypatching a dataclass
    default is a no-op, and a test that relies on it passes while verifying
    nothing. Construct the object explicitly instead, as below.
    """
    import dataclasses

    async def observe(cfg: Config) -> tuple[str, int]:
        svc = RagService(InMemoryRetriever(),
                         ScriptedModel([good_response(confidence=0.3)]), cfg)
        a = await svc.answer("invoice 88 units", frozenset({"isc-all"}))
        return a.outcome, len(a.citations)

    baseline = await observe(Config())
    mutated = await observe(dataclasses.replace(Config(), **overrides))

    assert baseline != mutated, (
        f"MUTATION SURVIVED ({description}): the config change produced "
        f"identical behaviour {baseline}. Either it is untested or it is dead."
    )


# ===========================================================================
# 3. TEST SMELLS
# ===========================================================================

def test_smell_assertion_on_a_message_string() -> None:
    """SMELL: asserting on human-readable prose.

    `assert "not valid JSON" in str(e)` breaks the first time someone rewords
    the message, for a change with no behavioural effect. That teaches people
    that tests are noise.

    BETTER: assert on the EXCEPTION TYPE, and on structured fields if the
    exception carries any.

    `pytest.raises(..., match=...)` is a reasonable middle ground where the
    type alone is too coarse — as in test_02, where several different
    validation failures all raise GenerationFailed and the message is the only
    thing distinguishing them. Use a SHORT, STABLE fragment, never the whole
    sentence.
    """
    with pytest.raises(GenerationFailed) as exc:
        RagService.parse("not json")

    assert isinstance(exc.value, GenerationFailed)     # robust
    assert "JSON" in str(exc.value)                     # acceptable fragment
    # assert str(exc.value) == "not valid JSON: Expecting value: line 1..."
    #   ^ brittle: pins the exact wording AND json's error text


async def test_smell_multiple_unrelated_assertions() -> None:
    """SMELL: one test asserting several unrelated things.

    When it fails you get ONE failure for what may be several bugs, and
    pytest stops at the first assertion — hiding the rest.

    THE FIX is usually parametrisation, or separate tests. The exception is
    a test with a NARRATIVE, where the sequence is the point (see test_02's
    last test); there, several assertions about one scenario is correct.
    """
    svc = RagService(InMemoryRetriever(), ScriptedModel([good_response()]))
    answer = await svc.answer("invoice 88 units", frozenset({"isc-all"}))

    # These four are all about ONE outcome, so this is the acceptable case.
    assert answer.proposal is not None
    assert answer.outcome == "ok"
    assert answer.repairs == 0
    assert not answer.degraded


def test_smell_logic_in_the_test() -> None:
    """SMELL: an `if` in a test body.

    A conditional means the test does different things in different runs, so
    a passing result no longer tells you which path was verified. It also
    means the test can silently assert nothing:

        if condition:
            assert something     # and if not, the test passes vacuously

    THE FIX: parametrise, so each branch is its own named test.

    THE SAME APPLIES to `try/except` in a test body. `pytest.raises` is
    explicit about what is expected; a bare except can swallow the failure
    you were trying to detect.
    """
    for raw, expected in [(good_response(), "adjust_quantity"),
                          (good_response("approve"), "approve")]:
        assert RagService.parse(raw).disposition == expected
    # Even this loop is worse than @parametrize: one failure hides the rest.


# ===========================================================================
# 4. AUDITING THE SUITE ITSELF
# ===========================================================================

TESTS_ROOT = Path(__file__).resolve().parents[1]


def _test_functions(path: Path) -> list[tuple[str, str]]:
    """Crude extraction of test function bodies. Good enough for a lint."""
    import ast
    tree = ast.parse(path.read_text())
    out: list[tuple[str, str]] = []
    src = path.read_text().splitlines()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and \
                node.name.startswith("test_"):
            body = "\n".join(src[node.lineno - 1:node.end_lineno])
            out.append((node.name, body))
    return out


def test_every_test_asserts_something() -> None:
    """A LINT OVER THE SUITE: no test may be assertion-free.

    Cheap to run, and it catches the single most common way a test becomes
    decorative — someone comments out a failing assertion "temporarily".

    NOTE THE ALLOWLIST. `test_100_percent_coverage_with_zero_assertions` above
    is deliberately assertion-free because it demonstrates the problem. An
    allowlist with a REASON is honest; a lint with no escape hatch gets
    disabled entirely the first time it is wrong.
    """
    allowed = {
        "test_100_percent_coverage_with_zero_assertions",
        "test_every_test_asserts_something",
    }
    offenders: list[str] = []
    for path in TESTS_ROOT.rglob("test_*.py"):
        for name, body in _test_functions(path):
            if name in allowed:
                continue
            if "assert" not in body and "pytest.raises" not in body \
                    and "pytest.skip" not in body and "pytest.fail" not in body:
                offenders.append(f"{path.name}::{name}")
    assert not offenders, f"tests with no assertion: {offenders}"


def test_no_test_is_silently_skipped() -> None:
    """A bare `@pytest.mark.skip` with no reason is a test nobody will ever
    re-enable, because nobody knows why it was disabled.

    REQUIRE A REASON. It costs eight characters and it is the difference
    between a temporary skip and permanent dead code.
    """
    # SCAN DECORATOR LINES, not the raw file text. The first version of this
    # lint regex-matched the whole file and flagged ITSELF — because the
    # pattern string in this function's own source matched the pattern.
    #
    # A lint that fails on its own implementation is not a subtle bug; it is
    # a sign the lint is operating on the wrong representation. Decorators
    # start a line with `@`, so filtering to those lines is both simpler and
    # correct. (An AST walk would be more correct still, and is what a real
    # plugin should do.)
    offenders: list[str] = []
    for path in TESTS_ROOT.rglob("test_*.py"):
        for i, line in enumerate(path.read_text().splitlines(), 1):
            stripped = line.strip()
            if not stripped.startswith("@"):
                continue
            if re.match(r"@pytest\.mark\.skip(?!if)", stripped) and \
                    "reason" not in stripped:
                offenders.append(f"{path.name}:{i}")
    assert not offenders, (
        f"@pytest.mark.skip without a reason at: {offenders}"
    )


def test_no_nonstrict_xfail() -> None:
    """Every xfail must be strict, so it fails when the bug is fixed.

    A non-strict xfail stays green whether the code works or not — it is a
    comment with a decorator, and it rots. Script 02 has a worked example of
    a strict xfail catching a stale assumption on the day it was written.
    """
    offenders: list[str] = []
    for path in TESTS_ROOT.rglob("test_*.py"):
        lines = path.read_text().splitlines()
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            # Only DECORATOR lines. Without this filter the lint matches its
            # own source — the same self-matching bug as the skip lint above.
            # `marks=pytest.mark.xfail(...)` inside a pytest.param is the one
            # non-decorator form that counts, so allow it explicitly.
            is_decorator = stripped.startswith("@pytest.mark.xfail")
            is_param_mark = stripped.startswith("marks=pytest.mark.xfail")
            if not (is_decorator or is_param_mark):
                continue
            # An xfail's arguments may wrap over several lines.
            window = "\n".join(lines[i - 1:i + 4])
            if "strict=True" not in window:
                offenders.append(f"{path.name}:{i}")
    assert not offenders, f"non-strict xfail at: {offenders}"


def test_security_tests_are_never_skipped_or_xfailed() -> None:
    """THE MOST IMPORTANT LINT IN THIS FILE.

    A security test that is skipped or expected-to-fail is worse than no test:
    it appears in the report, it looks like coverage, and it verifies nothing.

    So: no `skip`, no `skipif`, no `xfail` on anything marked `security`.
    A security control that cannot be tested in CI is a control that needs to
    be redesigned, not annotated.
    """
    offenders: list[str] = []
    for path in TESTS_ROOT.rglob("test_*.py"):
        lines = path.read_text().splitlines()
        for i, line in enumerate(lines):
            if "pytest.mark.security" not in line:
                continue
            window = "\n".join(lines[max(0, i - 4):i + 5])
            if re.search(r"pytest\.mark\.(skip|skipif|xfail)", window):
                offenders.append(f"{path.name}:{i + 1}")
    assert not offenders, (
        f"a security test is skipped or xfailed at: {offenders}. "
        f"Fix the control or fix the test — do not annotate it away."
    )


# ===========================================================================
# 5. WHAT COVERAGE IS ACTUALLY GOOD FOR
# ===========================================================================

def test_coverage_is_a_floor_not_a_target() -> None:
    """A NOTE, because the point is about how to USE the number.

    COVERAGE ANSWERS ONE QUESTION WELL: "is there any code we never execute?"
    A file at 0% is genuinely untested. That is worth knowing and worth
    alerting on.

    IT ANSWERS NOTHING ELSE. 90% and 95% are not meaningfully different, and
    chasing the last few percent produces tests written to touch lines rather
    than to check behaviour — exactly the assertion-free tests this file opens
    with.

    HOW TO USE IT:
      * a RATCHET: coverage may not DECREASE. Cheap, mechanical, effective.
      * per-FILE floors on the code that matters (the parser, the ACL check),
        rather than one global number.
      * `--cov-report=term-missing` to find UNTOUCHED BRANCHES, which are
        genuinely useful — an `except` clause at 0% has never been exercised.

    HOW NOT TO:
      * a global "we require 85%" gate. It is met by adding weak tests, and
        the number stops meaning anything within two sprints.

    THE BETTER METRIC, if you want one: mutation score. It is slower and
    fiddlier, and it measures the thing you actually care about.

    A CONCRETE NUMBER FROM THIS REPO. Running
        pytest --cov=rag --cov-report=term-missing
    reports 99% coverage of rag/service.py — one uncovered line.

    That number is genuinely reassuring AND genuinely insufficient. The very
    first test in this file achieves full coverage of the parser while
    asserting nothing at all. 99% tells you the suite EXECUTES the code; only
    the mutation tests above tell you it CHECKS it.

    Use the 99% as a floor that may not fall. Use the mutation tests as the
    thing you actually trust.
    """
    assert True
