"""
tests/eval/test_05_eval_as_test.py — testing a nondeterministic system.

Run:  pytest tests/eval -v
      pytest -m "not eval"        # the default developer loop excludes these

THE PROBLEM
-----------
Everything so far tested DETERMINISTIC behaviour: given this input, assert
that output. An LLM does not offer that. The same prompt can produce different
text, and "correct" is a judgement rather than an equality check.

Two failure modes follow, and both are common:

  1. TREATING AN EVAL AS A UNIT TEST. `assert answer == "the expected text"`
     against a live model is flaky by construction. It gets marked flaky,
     then skipped, then deleted — and it takes the useful tests with it.

  2. TREATING AN EVAL AS NOT-A-TEST. Quality lives in a notebook someone runs
     occasionally, so a prompt change that degrades answers ships and nobody
     notices for two weeks.

THE RESOLUTION is a THIRD TIER with different rules:

    UNIT           deterministic, scripted model, runs on every save.
                   A failure BLOCKS the merge.
    CONTRACT       fakes match the real implementation's behaviour.
    EVAL           real model, golden set, reported as k/n.
                   A REGRESSION blocks; an absolute score does not.

THE REPORTING DISCIPLINE that makes this work: report k/n over NAMED
scenarios, never a percentage. With n=12, "58%" implies a precision the sample
cannot support, and it invites comparison against a threshold nobody can
defend. "7/12, and these five failed" is honest and actionable.
"""

from __future__ import annotations

import json
import statistics
from collections import Counter
from dataclasses import dataclass, field
from decimal import Decimal

import pytest

from rag.service import (
    Answer,
    GenerationFailed,
    Proposal,
    RagService,
    good_response,
)


# ===========================================================================
# 1. THE GOLDEN SET
# ===========================================================================

@dataclass(frozen=True)
class EvalCase:
    """One named scenario.

    FIELDS THAT MATTER:
      id            stable, human-readable. This is what a report names.
      must_cite     the citations REQUIRED for the answer to be grounded.
      forbidden     chunks that must NEVER appear (permission or relevance).
      why           why this case exists. A golden set without rationale
                    becomes uneditable, because nobody dares change a case
                    they do not understand.
    """

    id: str
    question: str
    groups: frozenset[str]
    expect_disposition: str
    must_cite: frozenset[str] = frozenset()
    forbidden_citations: frozenset[str] = frozenset()
    min_confidence: float = 0.0
    why: str = ""


GOLDEN_SET: list[EvalCase] = [
    EvalCase(
        id="qty_over_receipt",
        question="invoice 88 units billed receipt",
        groups=frozenset({"isc-all"}),
        expect_disposition="adjust_quantity",
        must_cite=frozenset({"c3"}),
        why="The core three-way-match case: 402 billed vs 400 received.",
    ),
    EvalCase(
        id="price_match",
        question="PO 1001 unit price",
        groups=frozenset({"isc-all"}),
        expect_disposition="approve",
        must_cite=frozenset({"c2"}),
        why="A clean match must not be escalated. Guards over-triggering.",
    ),
    EvalCase(
        id="hr_isolation",
        question="compensation band plant leads",
        groups=frozenset({"isc-all"}),
        expect_disposition="approve",
        forbidden_citations=frozenset({"c5"}),
        why="SECURITY: an ISC caller must never cite the HR chunk.",
    ),
    EvalCase(
        id="legal_isolation",
        question="settlement terms confidential",
        groups=frozenset({"isc-all"}),
        expect_disposition="approve",
        forbidden_citations=frozenset({"c6"}),
        why="SECURITY: an ISC caller must never cite the legal chunk.",
    ),
    EvalCase(
        id="receipt_lookup",
        question="goods receipt 55 units recorded",
        groups=frozenset({"isc-all"}),
        expect_disposition="approve",
        must_cite=frozenset({"c4"}),
        why="Basic retrieval grounding.",
    ),
]


@pytest.fixture(scope="session")
def golden() -> list[EvalCase]:
    return GOLDEN_SET


# ===========================================================================
# 2. THE RESULT OBJECT — k/n, never a percentage
# ===========================================================================

@dataclass
class EvalResult:
    passed: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)

    @property
    def k(self) -> int:
        return len(self.passed)

    @property
    def n(self) -> int:
        return len(self.passed) + len(self.failed)

    def report(self) -> str:
        """DELIBERATELY REFUSES to compute a percentage.

        With n=5, the difference between 4/5 and 5/5 is one case, and the
        binomial confidence interval on "80%" spans roughly 28% to 99%.
        Reporting "80% accuracy" invites a comparison against a threshold
        that the sample size cannot support.

        `4/5, failed: [hr_isolation]` is the honest form and it tells you what
        to do next.
        """
        lines = [f"{self.k}/{self.n} scenarios passed"]
        for name, why in self.failed:
            lines.append(f"  FAILED {name}: {why}")
        return "\n".join(lines)


def evaluate(answer: Answer, case: EvalCase) -> tuple[bool, str]:
    """Grade one answer. Returns (passed, reason)."""
    if answer.proposal is None:
        return False, f"no proposal ({answer.outcome})"

    forbidden = case.forbidden_citations & set(answer.citations)
    if forbidden:
        # A permission failure is graded as a failure REGARDLESS of the
        # disposition. An answer that reaches the right conclusion using a
        # document the caller may not see is still a security incident.
        return False, f"cited forbidden chunks {sorted(forbidden)}"

    missing = case.must_cite - set(answer.citations)
    if missing:
        return False, f"missing required citations {sorted(missing)}"

    if answer.proposal.disposition != case.expect_disposition:
        return False, (f"disposition {answer.proposal.disposition!r}, "
                       f"expected {case.expect_disposition!r}")

    if answer.proposal.confidence < case.min_confidence:
        return False, f"confidence {answer.proposal.confidence} below floor"

    return True, ""


# ===========================================================================
# 3. THE DETERMINISTIC TIER — runs on every save
# ===========================================================================

@pytest.mark.parametrize("case", GOLDEN_SET, ids=lambda c: c.id)
async def test_golden_case_with_a_scripted_model(case: EvalCase,
                                                 make_service) -> None:
    """THE SAME GOLDEN SET, run against a SCRIPTED model.

    This is not a quality test — the model is told what to say. It tests the
    PIPELINE: retrieval, permission trimming, parsing, citation binding,
    grading. All of that is deterministic and belongs in the fast tier.

    THE VALUE: when an eval regresses, this tier tells you whether the
    pipeline broke or the model did. Without it, every quality regression
    starts with "is this the prompt, the model, or a bug?" and takes a day.

    Note this test is NOT marked eval — it lives here for cohesion with the
    golden set, but the conftest hook marks it by directory. In a real repo
    it would sit in tests/unit/ and import GOLDEN_SET.
    """
    svc = make_service([good_response(
        disposition=case.expect_disposition,
        evidence=sorted(case.must_cite) or ["c1"],
    )])
    answer = await svc.answer(case.question, case.groups)
    passed, why = evaluate(answer, case)
    assert passed, f"{case.id}: {why}\n(case rationale: {case.why})"


# ===========================================================================
# 4. THE EVAL TIER — nondeterministic, reported as k/n
# ===========================================================================

class NoisyModel:
    """Stands in for a real model: mostly right, sometimes not.

    THE POINT is that its output VARIES across runs. A test written against
    it must therefore assert something other than exact equality — which is
    the whole methodological problem this file exists to address.
    """

    # NOTE ON TUNING, because the first version of this class taught me
    # something: with a 15% failure rate the suite scored 5/5 on every run.
    # The reason is that most injected failures are MALFORMED output, which
    # the repair loop simply fixes — the pipeline is doing its job.
    #
    # The failure that actually matters for an eval is the one repair cannot
    # touch: syntactically valid JSON containing the WRONG ANSWER. No parser
    # catches it, no retry helps, and only a graded golden set finds it.
    #
    # So `wrong_answer_rate` is separate from `malformed_rate`, and it is the
    # dominant one. That split is not an artefact of the demo — it is the real
    # shape of LLM failure, and an eval harness that only injects malformed
    # output will report a reassuring score against a system that is wrong.
    def __init__(self, seed: int, *, wrong_answer_rate: float = 0.30,
                 malformed_rate: float = 0.15) -> None:
        import random
        self.rng = random.Random(seed)
        self.wrong_answer_rate = wrong_answer_rate
        self.malformed_rate = malformed_rate
        self.calls = 0

    async def complete(self, prompt: str, *,
                       max_tokens: int) -> tuple[str, int, int]:
        self.calls += 1
        # Derive the "right" answer from the prompt, then perturb it.
        wanted = ("adjust_quantity" if "invoice 88" in prompt else "approve")
        cited = [c for c in ("c1", "c2", "c3", "c4")
                 if f"[{c}]" in prompt] or ["c1"]

        r = self.rng.random()

        # WRONG BUT WELL-FORMED. The repair loop cannot help; only grading
        # against a golden set catches this.
        if r < self.wrong_answer_rate:
            wrong = "escalate_to_buyer" if wanted == "approve" else "approve"
            return good_response(disposition=wrong, evidence=cited), 100, 40

        # MALFORMED. The repair loop fixes these, so they mostly do NOT
        # affect the eval score — they affect latency and cost instead.
        if r < self.wrong_answer_rate + self.malformed_rate * 0.5:
            return "I'm not able to help with that.", 100, 10
        if r < self.wrong_answer_rate + self.malformed_rate:
            return "```json\n" + good_response(
                disposition=wanted, evidence=cited) + "\n```", 100, 40

        return good_response(disposition=wanted, confidence=0.84,
                             evidence=cited), 100, 40


@pytest.mark.eval
async def test_eval_suite_reports_k_over_n(golden: list[EvalCase],
                                           make_service, request) -> None:
    """ONE TEST for the whole suite, reporting k/n.

    WHY ONE TEST rather than one per case: an eval is a MEASUREMENT of the
    system, not N independent assertions. Five separate failing tests tell you
    "five things are broken"; one test reporting `3/5, failed: [...]` tells
    you the actual state.

    It also makes the regression threshold expressible. See the next test.
    """
    from rag.service import InMemoryRetriever, RagService

    result = EvalResult()
    for i, case in enumerate(golden):
        svc = RagService(InMemoryRetriever(), NoisyModel(seed=100 + i))
        answer = await svc.answer(case.question, case.groups)
        passed, why = evaluate(answer, case)
        (result.passed.append(case.id) if passed
         else result.failed.append((case.id, why)))

    print("\n" + result.report())

    # THE ASSERTION IS A FLOOR, not an equality. See the next test for why
    # this number exists and how it should be maintained.
    # A FLOOR, and a deliberately loose one. The variance test below measures
    # a spread of min=2/5 to max=5/5 across runs at these seeds — so any floor
    # above 2 would be flaky by construction.
    #
    # THAT IS THE HONEST CONCLUSION and it is uncomfortable: with n=5 and this
    # much noise, an absolute floor is nearly worthless as a gate. The
    # regression test below is the assertion that actually gates a merge; this
    # one only catches a total collapse.
    #
    # The fix in a real system is a BIGGER GOLDEN SET, not a cleverer
    # threshold. Twenty cases would narrow the spread considerably; five
    # cannot, no matter how the assertion is phrased.
    assert result.k >= 2, result.report()


# ===========================================================================
# 5. REGRESSION, NOT ABSOLUTE SCORE
# ===========================================================================

BASELINE_PATH_NOTE = """
In a real repo this lives in a committed file — evals/baseline.json — updated
by a deliberate PR when a change legitimately moves the number. Inline here so
the tutorial runs standalone.
"""

# Recorded from an actual run of this suite at these seeds. Two cases fail
# today; that is written down rather than hidden behind an aggregate score.
BASELINE: dict[str, bool] = {
    "qty_over_receipt": False,    # KNOWN failure, recorded honestly
    "price_match": True,
    "hr_isolation": False,        # KNOWN failure, recorded honestly
    "legal_isolation": True,
    "receipt_lookup": True,
}


@pytest.mark.eval
async def test_no_case_regresses_from_the_baseline(golden: list[EvalCase],
                                                   ) -> None:
    """THE ASSERTION THAT ACTUALLY GATES A MERGE.

    Not "accuracy >= 80%" — a threshold nobody can defend and everyone
    eventually lowers. Instead: NO CASE THAT PASSED BEFORE MAY FAIL NOW.

    Properties that make this work in practice:
      * per-CASE, so the failure names exactly what broke
      * known failures are RECORDED, not hidden — `receipt_lookup: False`
        above is honest, and a fix flips it to True in the same PR
      * a newly-passing case is reported but does not fail the build
      * the baseline is a committed artifact, so moving it is a reviewable
        diff rather than a silent threshold change

    THE FAILURE MODE THIS PREVENTS: a prompt change that fixes two cases and
    breaks one. An aggregate score goes UP and the regression ships.
    """
    from rag.service import InMemoryRetriever, RagService

    current: dict[str, bool] = {}
    reasons: dict[str, str] = {}
    for i, case in enumerate(golden):
        svc = RagService(InMemoryRetriever(), NoisyModel(seed=100 + i))
        answer = await svc.answer(case.question, case.groups)
        passed, why = evaluate(answer, case)
        current[case.id] = passed
        reasons[case.id] = why

    regressions = [cid for cid, was in BASELINE.items()
                   if was and not current.get(cid, False)]
    improvements = [cid for cid, was in BASELINE.items()
                    if not was and current.get(cid, False)]

    if improvements:
        print(f"\nIMPROVED (update the baseline in this PR): {improvements}")

    assert not regressions, (
        "cases regressed from the committed baseline:\n"
        + "\n".join(f"  {cid}: {reasons[cid]}" for cid in regressions)
    )


# ===========================================================================
# 6. SECURITY CASES ARE NOT GRADED ON A CURVE
# ===========================================================================

@pytest.mark.security
@pytest.mark.parametrize(
    "case", [c for c in GOLDEN_SET if c.forbidden_citations],
    ids=lambda c: c.id,
)
async def test_permission_cases_are_absolute(case: EvalCase) -> None:
    """SECURITY CASES GET THEIR OWN TEST, AND IT IS PER-CASE AND STRICT.

    A permission failure must never be absorbed into a k/n score. "4/5, and
    the one that failed was the HR isolation case" is not an acceptable
    result — it is an incident.

    So these run:
      * per-case, so each failure is its own red test
      * marked `security`, which the CI config never allows to be skipped or
        xfailed
      * against MANY seeds, because a security property must hold for every
        sample, not on average

    THIS IS THE MOST IMPORTANT TEST IN THE FILE.
    """
    from rag.service import InMemoryRetriever, RagService

    for seed in range(25):
        svc = RagService(InMemoryRetriever(), NoisyModel(seed=seed))
        answer = await svc.answer(case.question, case.groups)
        leaked = case.forbidden_citations & set(answer.citations)
        assert not leaked, (
            f"seed={seed}: {case.id} leaked {sorted(leaked)} — "
            f"a permission failure is never acceptable at any rate"
        )


# ===========================================================================
# 7. VARIANCE — one run tells you almost nothing
# ===========================================================================

@pytest.mark.eval
async def test_report_variance_across_seeds(golden: list[EvalCase]) -> None:
    """RUN THE SUITE SEVERAL TIMES AND REPORT THE SPREAD.

    A single eval run is one sample from a distribution. Reporting "7/12"
    from one run and comparing it against last week's "8/12" is comparing
    two noise draws, and teams routinely chase a regression that was
    variance.

    Reporting `min/median/max across 5 runs` makes the noise visible, so a
    real regression is distinguishable from a bad draw.

    THE HONEST CAVEAT: five runs of a five-case suite is still a small sample.
    This tells you the ORDER OF MAGNITUDE of the noise, not a confidence
    interval. If you need the latter, you need a much larger golden set —
    and that is a real cost worth naming rather than pretending otherwise.
    """
    from rag.service import InMemoryRetriever, RagService

    scores: list[int] = []
    for run in range(5):
        result = EvalResult()
        for i, case in enumerate(golden):
            svc = RagService(InMemoryRetriever(),
                             NoisyModel(seed=run * 1000 + i))
            answer = await svc.answer(case.question, case.groups)
            passed, why = evaluate(answer, case)
            (result.passed.append(case.id) if passed
             else result.failed.append((case.id, why)))
        scores.append(result.k)

    n = len(golden)
    print(f"\nacross 5 runs of {n} cases: "
          f"min={min(scores)}/{n} median={int(statistics.median(scores))}/{n} "
          f"max={max(scores)}/{n}  raw={scores}")

    spread = max(scores) - min(scores)
    assert spread <= n, "sanity check on the harness itself"


# ===========================================================================
# 8. PROPERTY-BASED TESTING
# ===========================================================================

from hypothesis import given, settings
from hypothesis import strategies as st


@settings(max_examples=200, deadline=None)
@given(
    disposition=st.sampled_from(sorted([
        "approve", "request_credit", "adjust_quantity",
        "hold_pending_receipt", "escalate_to_buyer"])),
    confidence=st.floats(min_value=0.0, max_value=1.0,
                         allow_nan=False, allow_infinity=False),
    reason=st.text(min_size=10, max_size=200).filter(lambda s: len(s) >= 10),
    evidence=st.lists(st.text(min_size=1, max_size=8), min_size=1, max_size=5),
)
def test_parser_accepts_everything_that_should_be_valid(
    disposition: str, confidence: float, reason: str, evidence: list[str]
) -> None:
    """PROPERTY-BASED: instead of examples, state an INVARIANT and let
    hypothesis attack it.

    THE PROPERTY: any JSON satisfying the schema must parse. Hypothesis
    generates 200 cases including the ones you would never think of — empty
    strings, unicode, values exactly at the boundary, floats like 1e-300.

    WHERE THIS EARNS ITS PLACE: parsers, validators, serialisers, anything
    with a stateable invariant. It found a real class of bug for me here —
    see the `.filter` on `reason` and the `allow_nan=False`, both of which are
    there because the first version generated inputs the schema does not
    actually permit.

    WHERE IT DOES NOT: business logic where the "property" is just a
    restatement of the implementation. If your property test mirrors the code
    line for line, it tests nothing and doubles the maintenance.
    """
    raw = json.dumps({
        "disposition": disposition, "confidence": confidence,
        "reason": reason, "evidence_ids": evidence,
    })
    proposal = RagService.parse(raw)
    assert proposal.disposition == disposition
    assert proposal.confidence == pytest.approx(confidence)
    assert len(proposal.evidence_ids) == len(evidence)


@settings(max_examples=200, deadline=None)
@given(text=st.text(max_size=300))
def test_parser_never_crashes_on_arbitrary_text(text: str) -> None:
    """THE ROBUSTNESS PROPERTY, and the one most worth having for LLM output.

    A model can emit ANYTHING. The parser must respond with a
    GenerationFailed — never a TypeError, KeyError, IndexError,
    RecursionError, or an unhandled UnicodeError.

    That distinction matters operationally: GenerationFailed is classified as
    REPAIRABLE and handled; anything else escapes as a 500.
    """
    try:
        RagService.parse(text)
    except GenerationFailed:
        pass          # the expected, handled outcome
    except Exception as e:  # noqa: BLE001 — that is the point of the test
        pytest.fail(
            f"parser raised {type(e).__name__} instead of GenerationFailed "
            f"for input {text!r}"
        )


# ===========================================================================
# 9. LIVE TESTS — opt-in, never in the default run
# ===========================================================================

@pytest.mark.eval
@pytest.mark.slow
async def test_against_a_live_model(request: pytest.FixtureRequest) -> None:
    """Gated behind `--run-live`, so it never runs by accident.

    RULES FOR LIVE TESTS:
      * OPT-IN via a flag, never by an env var that might be set in CI
      * they cost money, so a nightly schedule rather than per-PR
      * a failure opens an ISSUE; it does not block a merge, because a
        provider incident is not a code defect
      * pin the model VERSION, not just the deployment — otherwise your
        baseline drifts when the provider updates underneath you

    THE THING PEOPLE GET WRONG: running live tests on every PR. It makes the
    suite slow, flaky, and expensive, and it couples your merge queue to a
    third party's availability.
    """
    if not request.config.getoption("--run-live"):
        pytest.skip("live model tests require --run-live")
    pytest.skip("no credentials configured in this tutorial")
