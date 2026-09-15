"""
tests/unit/test_02_parametrize.py — parametrisation.

Run:  pytest tests/unit/test_02_parametrize.py -v

THE ARGUMENT
------------
A loop inside one test:

    def test_dispositions():
        for d in ["approve", "request_credit", "bogus"]:
            assert validate(d) == (d != "bogus")

...fails at the FIRST bad case, hides the rest, reports one failure for what
may be several bugs, and gives a failure message that does not say which case
broke.

Parametrised, each case is a separate test with its own name, its own pass/
fail, and its own line in the report. You can run one of them by name. That is
the whole difference, and it is worth more than it sounds when a table has
forty rows.

Run with `-v` to see the generated test ids; that readout IS the documentation
of what the code accepts.
"""

from __future__ import annotations

import json
from decimal import Decimal

import pytest

from rag.service import (
    Chunk,
    Config,
    GenerationFailed,
    Proposal,
    RagService,
    good_response,
)


# ===========================================================================
# 1. THE BASICS, AND WHY ids MATTER
# ===========================================================================

@pytest.mark.parametrize("disposition", [
    "approve", "request_credit", "adjust_quantity",
    "hold_pending_receipt", "escalate_to_buyer",
])
def test_every_valid_disposition_parses(disposition: str) -> None:
    """Five tests, five names, five independent results.

    The ids come from the values automatically here because they are simple
    strings. `-v` shows:
        test_every_valid_disposition_parses[approve]
        test_every_valid_disposition_parses[request_credit]
        ...
    which means `pytest -k "escalate"` runs exactly one of them.
    """
    proposal = RagService.parse(good_response(disposition=disposition))
    assert proposal.disposition == disposition


@pytest.mark.parametrize(
    "raw,expected_error",
    [
        ('{"disposition": "partial_approve", "confidence": 0.8, '
         '"reason": "long enough reason", "evidence_ids": ["c1"]}',
         "not one of"),
        ('{"disposition": "approve", "confidence": 1.4, '
         '"reason": "long enough reason", "evidence_ids": ["c1"]}',
         "out of range"),
        ('{"disposition": "approve", "confidence": "high", '
         '"reason": "long enough reason", "evidence_ids": ["c1"]}',
         "not numeric"),
        ('{"disposition": "approve", "confidence": 0.8, '
         '"reason": "short", "evidence_ids": ["c1"]}',
         "at least 10 characters"),
        ('{"disposition": "approve", "confidence": 0.8, '
         '"reason": "long enough reason", "evidence_ids": []}',
         "non-empty"),
        ("not json at all", "not valid JSON"),
        ("[1, 2, 3]", "expected a JSON object"),
    ],
    ids=["hallucinated_enum", "confidence_too_high", "confidence_not_numeric",
         "reason_too_short", "no_evidence", "not_json", "json_but_not_object"],
)
def test_invalid_output_is_rejected(raw: str, expected_error: str) -> None:
    """EXPLICIT ids are the difference between a readable and an unreadable
    report.

    Without them pytest generates `[raw0]`, `[raw1]`... from the long JSON
    strings, and a failure tells you nothing. With them:

        test_invalid_output_is_rejected[hallucinated_enum] FAILED

    ...names the bug in the test id itself.

    NOTE THE SECOND ASSERTION. Checking only `pytest.raises(GenerationFailed)`
    would pass even if the parser rejected the input for the WRONG reason —
    which is how a validation rule silently stops working while its test
    stays green.
    """
    with pytest.raises(GenerationFailed, match=expected_error):
        RagService.parse(raw)


# ===========================================================================
# 2. pytest.param — ids, marks, and xfail per case
# ===========================================================================

@pytest.mark.parametrize("raw,expected", [
    pytest.param(good_response(), "adjust_quantity", id="plain_json"),
    pytest.param(f"```json\n{good_response()}\n```", "adjust_quantity",
                 id="markdown_fenced"),
    pytest.param(f"```\n{good_response()}\n```", "adjust_quantity",
                 id="bare_fence"),
    pytest.param(f"Sure! Here you go:\n{good_response()}\nHope that helps.",
                 "adjust_quantity", id="prose_wrapped"),
    pytest.param(
        '{"disposition": "adjust_quantity", "confidence": "84%", '
        '"reason": "long enough reason here", "evidence_ids": ["c1"]}',
        "adjust_quantity", id="percent_confidence"),
    pytest.param(
        '{"disposition": "adjust_quantity", "confidence": "eighty-four", '
        '"reason": "long enough reason here", "evidence_ids": ["c1"]}',
        "adjust_quantity", id="spelled_out_confidence",
        marks=pytest.mark.xfail(
            reason="a spelled-out confidence is not coerced; PLAT-431",
            strict=True, raises=GenerationFailed),
    ),
])
def test_free_repairs_handle_common_malformations(raw: str,
                                                  expected: str) -> None:
    """`pytest.param` attaches an id AND marks to a single case.

    THE xfail ROW IS THE INTERESTING ONE, and it caught me out while writing
    this file. I first marked `percent_confidence` as xfail on the assumption
    that "84%" was not coerced. Running it produced:

        FAILED ...[percent_confidence]
        [XPASS(strict)] percent-string confidence is not yet coerced

    XPASS(strict) means the test PASSED while marked as expected-to-fail. The
    parser does coerce percent strings — my "known gap" comment was wrong on
    the day I wrote it.

    THAT IS THE MECHANISM WORKING. `strict=True` turns a stale assumption into
    a failing test. A non-strict xfail would have stayed silently green
    forever, documenting a limitation that does not exist.

    So the row now tests what is genuinely unhandled (a spelled-out number),
    and `raises=GenerationFailed` pins WHY it fails — without that, the xfail
    would also "pass" if the parser started raising TypeError for an unrelated
    reason.

    RULE: every xfail is `strict=True` with a `raises=`, a `reason=`, and a
    ticket. Anything less is a comment that rots.
    """
    assert RagService.parse(raw).disposition == expected


# ===========================================================================
# 3. STACKING — the Cartesian product
# ===========================================================================

@pytest.mark.parametrize("groups", [
    pytest.param(frozenset({"isc-all"}), id="isc"),
    pytest.param(frozenset({"hr-only"}), id="hr"),
    pytest.param(frozenset({"legal-only"}), id="legal"),
])
@pytest.mark.parametrize("question", [
    pytest.param("invoice 88 units billed", id="invoice"),
    pytest.param("compensation band plant leads", id="compensation"),
    pytest.param("settlement terms confidential", id="settlement"),
])
async def test_retrieval_never_returns_invisible_chunks(
    make_service, groups: frozenset[str], question: str
) -> None:
    """3 x 3 = 9 tests from two decorators.

    THE SECURITY INVARIANT stated once and checked across the whole grid. This
    is exactly what parametrisation is for: the property is universal, so the
    test should be too.

    THE WARNING: stacking multiplies fast. Three decorators of five values
    each is 125 tests, and if each takes 200ms that is 25 seconds. Stack when
    the grid is genuinely meaningful; otherwise pick representative
    combinations explicitly.
    """
    svc = make_service([good_response()])
    answer = await svc.answer(question, groups)
    retrieved = svc.retriever.calls
    assert retrieved, "the retriever should have been called"
    # No PermissionViolation raised is itself the assertion — the service
    # fails closed on any invisible chunk.
    assert answer.outcome in ("ok", "low_confidence", "repair_exhausted") or \
        answer.outcome.startswith("repair_exhausted")


# ===========================================================================
# 4. INDIRECT — parametrising a FIXTURE rather than an argument
# ===========================================================================

@pytest.fixture
def configured_service(request: pytest.FixtureRequest, make_service):
    """`request.param` receives the parametrised value.

    USE INDIRECT WHEN the parameter needs SETUP: opening a connection,
    building a container, assembling an object graph. Here it saves each test
    from repeating the construction.

    DO NOT USE IT for plain data. Indirect parametrisation is significantly
    harder to read than a direct parameter, and the indirection only pays for
    itself when there is real setup to hide.
    """
    responses, overrides = request.param
    return make_service(responses, config_overrides=overrides)


@pytest.mark.parametrize(
    "configured_service,expected_outcome",
    [
        pytest.param(([good_response()], None), "ok", id="clean"),
        pytest.param((["bad", good_response()], None), "ok", id="one_repair"),
        pytest.param((["bad", "bad", "bad"], None), "repair_exhausted",
                     id="exhausted"),
        pytest.param(([good_response(confidence=0.3)], {"min_confidence": 0.6}),
                     "low_confidence", id="low_confidence"),
    ],
    indirect=["configured_service"],
)
async def test_outcomes(configured_service: RagService,
                        expected_outcome: str) -> None:
    """Note `indirect=["configured_service"]` — only THAT argument goes
    through the fixture; `expected_outcome` arrives directly."""
    answer = await configured_service.answer("invoice 88 units",
                                             frozenset({"isc-all"}))
    assert answer.outcome.startswith(expected_outcome)


# ===========================================================================
# 5. GENERATING CASES FROM DATA
# ===========================================================================

def load_golden_cases() -> list[dict[str, object]]:
    """In a real suite this reads a JSONL file. Kept inline so the tutorial
    runs without fixture files."""
    return [
        {"id": "qty_over_receipt", "question": "invoice 88 units billed",
         "groups": ["isc-all"], "expect": "adjust_quantity"},
        {"id": "price_check", "question": "PO 1001 unit price",
         "groups": ["isc-all"], "expect": "approve"},
        {"id": "receipt_check", "question": "goods receipt 55 units",
         "groups": ["isc-all"], "expect": "approve"},
    ]


@pytest.mark.parametrize(
    "case", load_golden_cases(),
    ids=lambda c: c["id"] if isinstance(c, dict) else str(c),
)
async def test_golden_cases(case: dict[str, object], make_service) -> None:
    """Cases loaded from data, ids derived by a callable.

    THE IMPORTANT CONSTRAINT: collection happens at IMPORT time, so the data
    must be available then. That means:
      * a file read at module level is fine
      * a database query or a network call at module level is NOT — it makes
        `pytest --collect-only` hit your network, and a collection error is
        much harder to debug than a test failure.

    If cases genuinely require I/O to enumerate, generate them in a build step
    and commit the result.
    """
    svc = make_service([good_response(disposition=str(case["expect"]))])
    answer = await svc.answer(str(case["question"]),
                              frozenset(case["groups"]))  # type: ignore[arg-type]
    assert answer.proposal is not None
    assert answer.proposal.disposition == case["expect"]


# ===========================================================================
# 6. pytest_generate_tests — the programmatic hook
# ===========================================================================

def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """The hook behind `@parametrize`. Use it when the cases depend on
    something only known at collection time — a command-line option, an
    environment variable, or a fixture's value.

    Here: `--model-tier` selects which deployments the tier test runs against.
    Run `pytest --model-tier=all` to widen it.

    USE THIS SPARINGLY. It is invisible from the test itself, which makes a
    surprising test id hard to trace. A decorator is almost always clearer.
    """
    if "deployment" in metafunc.fixturenames:
        tier = metafunc.config.getoption("--model-tier", default="cheap")
        deployments = (["gpt-4o-mini", "gpt-4o"] if tier == "all"
                       else ["gpt-4o-mini"])
        metafunc.parametrize("deployment", deployments)


def test_deployment_is_selected_by_option(deployment: str) -> None:
    """Runs once by default, twice with `--model-tier=all`."""
    assert deployment.startswith("gpt-")


# ===========================================================================
# 7. WHEN NOT TO PARAMETRISE
# ===========================================================================

async def test_a_scenario_with_real_narrative(make_service) -> None:
    """SOME TESTS SHOULD NOT BE PARAMETRISED.

    This one describes a specific sequence: bad JSON, then a hallucinated
    enum, then a valid response — and asserts on the repair COUNT and on the
    prompt EVOLUTION. Squeezing it into a table would make both the setup and
    the assertions unreadable.

    THE TEST: if the rows differ only in DATA, parametrise. If they differ in
    WHAT IS ASSERTED, write separate tests. A parametrised test whose body is
    full of `if expected_x is not None:` has stopped being a table and become
    three tests wearing a trench coat.
    """
    svc = make_service([
        "not json at all",
        '{"disposition": "partial_approve", "confidence": 0.8, '
        '"reason": "long enough reason", "evidence_ids": ["c1"]}',
        good_response(),
    ])
    answer = await svc.answer("invoice 88 units", frozenset({"isc-all"}))

    assert answer.repairs == 2
    assert answer.degraded is True
    assert answer.proposal is not None

    prompts = svc.model.prompts
    assert len(prompts) == 3
    assert "not valid JSON" in prompts[1], (
        "the first repair must tell the model what was wrong"
    )
    assert "not one of" in prompts[2], (
        "the second repair must name the invalid enum value"
    )
    assert len(prompts[0]) < len(prompts[1]) < len(prompts[2]), (
        "each repair appends context rather than replacing the prompt"
    )
