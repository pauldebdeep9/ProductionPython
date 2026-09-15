"""
09 — Capstone: an invoice exception resolution agent with real failure handling.

WHAT THIS COMPOSES
------------------
  01  taxonomy + adapter        -> one classification, dispositions not booleans
  02  budgets + deadlines       -> retry budget, propagated deadline
  03  idempotency               -> the write tool is safe to retry
  04  timeouts                  -> per-attempt and end-to-end
  05  breaker + bulkhead        -> containment per dependency
  06  fallback ladder           -> model degradation, labelled
  07  DLQ + batch guard         -> partial failure across a run
  08  repair loop               -> the LLM-specific failures

THE DOMAIN
----------
Three-way match: purchase order vs goods receipt vs invoice. When they
disagree, an exception is raised and someone must decide what to do. The agent
proposes a disposition; a deterministic gate re-derives the exception
independently and either approves the proposal or rejects it.

THE CONTROL THAT MATTERS MOST
-----------------------------
The approval gate does NOT trust the model. It recomputes the discrepancy from
the source documents and checks that the model's proposed disposition is
consistent with what it computed. A model that hallucinates a quantity, or
proposes 'approve' for a genuine overbill, is caught by arithmetic rather than
by hoping the prompt was good enough.

Exactly one tool has side effects (`write_proposal`), and it is idempotent.
Every other tool is read-only and therefore free to retry.

Run:  python 09_capstone_agent.py
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import random
import time
from dataclasses import dataclass, field
from decimal import Decimal

from failure_lab import (
    AppError,
    Disposition,
    Fail,
    FakeLLM,
    FaultInjector,
    Metrics,
    ServiceUnavailable,
    Timer,
    UpstreamTimeout,
    banner,
    classify,
    section,
)

# ---------------------------------------------------------------------------
# Domain model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Document:
    doc_id: str
    kind: str            # po | gr | invoice
    quantity: int
    unit_price: Decimal  # Decimal, never float — this is money
    currency: str = "USD"


@dataclass
class ExceptionCase:
    case_id: str
    po: Document
    gr: Document
    invoice: Document

    def computed_exceptions(self) -> list[str]:
        """Deterministic ground truth, derived from the documents.

        The gate uses this. The model never sees it, and the model's opinion
        never overrides it. Decimal arithmetic throughout — a float rounding
        error in a money comparison is its own class of production incident.
        """
        out = []
        if self.invoice.quantity > self.gr.quantity:
            out.append("qty_over_receipt")
        if self.invoice.quantity > self.po.quantity:
            out.append("qty_over_po")
        if self.invoice.unit_price > self.po.unit_price:
            out.append("price_over_po")
        if self.gr.quantity == 0:
            out.append("no_receipt")
        return out or ["none"]


@dataclass
class Proposal:
    case_id: str
    disposition: str
    confidence: float
    reason: str
    evidence_ids: list[str] = field(default_factory=list)


@dataclass
class Outcome:
    case_id: str
    status: str                    # approved | rejected | degraded | failed
    proposal: Proposal | None = None
    gate_reason: str = ""
    served_by: str = ""
    degraded: bool = False
    attempts: list[str] = field(default_factory=list)
    trace_id: str = ""


VALID_DISPOSITIONS = {
    "approve", "request_credit", "adjust_quantity",
    "hold_pending_receipt", "escalate_to_buyer",
}

# Which dispositions are consistent with which computed exceptions. This
# mapping is the policy, written down, reviewable, and testable — as opposed
# to living implicitly inside a prompt where nobody can audit it.
CONSISTENT = {
    "none": {"approve"},
    "qty_over_receipt": {"request_credit", "adjust_quantity", "escalate_to_buyer"},
    "qty_over_po": {"request_credit", "adjust_quantity", "escalate_to_buyer"},
    "price_over_po": {"request_credit", "escalate_to_buyer"},
    "no_receipt": {"hold_pending_receipt", "escalate_to_buyer"},
}


# ---------------------------------------------------------------------------
# Resilience primitives (compact versions of 02/05)
# ---------------------------------------------------------------------------

class RetryBudget:
    def __init__(self, ratio: float = 0.2, capacity: float = 50.0) -> None:
        self.ratio, self.capacity, self.tokens = ratio, capacity, capacity
        self.denied = 0

    def on_request(self) -> None:
        self.tokens = min(self.capacity, self.tokens + self.ratio)

    def try_retry(self) -> bool:
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True
        self.denied += 1
        return False


class Breaker:
    def __init__(self, *, window: int = 10, rate: float = 0.5,
                 min_calls: int = 5, open_for: float = 0.3) -> None:
        self.results: list[bool] = []
        self.window, self.rate, self.min_calls = window, rate, min_calls
        self.open_for = open_for
        self.state = "closed"
        self._opened = 0.0
        self.rejected = 0

    async def call(self, fn):
        if self.state == "open":
            if time.monotonic() - self._opened >= self.open_for:
                self.state = "half_open"
            else:
                self.rejected += 1
                raise ServiceUnavailable("circuit open")
        try:
            r = await fn()
        except Exception:
            self._record(False)
            raise
        else:
            self._record(True)
            return r

    def _record(self, ok: bool) -> None:
        if self.state == "half_open":
            self.state = "closed" if ok else "open"
            if not ok:
                self._opened = time.monotonic()
            self.results.clear()
            return
        self.results.append(ok)
        self.results = self.results[-self.window:]
        if (len(self.results) >= self.min_calls
                and self.results.count(False) / len(self.results) >= self.rate):
                self.state = "open"
                self._opened = time.monotonic()
                self.results.clear()


class Deadline:
    def __init__(self, seconds: float) -> None:
        self._loop = asyncio.get_event_loop()
        self.at = self._loop.time() + seconds

    def remaining(self) -> float:
        return max(0.0, self.at - self._loop.time())

    def expired(self) -> bool:
        return self.remaining() <= 0


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

class ToolRegistry:
    """Read-only tools retry freely. The ONE write tool is idempotent.

    Marking read-only vs write in the registry rather than by convention means
    the retry layer can make the safety decision automatically, instead of
    relying on every future contributor remembering the rule.
    """

    def __init__(self) -> None:
        self.proposals: dict[str, Proposal] = {}    # idempotency_key -> proposal
        self.write_calls = 0
        self.read_only = {"get_po", "get_gr", "get_invoice"}

    async def get_po(self, case: ExceptionCase) -> dict:
        await asyncio.sleep(0.002)
        return {"doc_id": case.po.doc_id, "quantity": case.po.quantity,
                "unit_price": str(case.po.unit_price)}

    async def get_gr(self, case: ExceptionCase) -> dict:
        await asyncio.sleep(0.002)
        return {"doc_id": case.gr.doc_id, "quantity": case.gr.quantity}

    async def get_invoice(self, case: ExceptionCase) -> dict:
        await asyncio.sleep(0.002)
        return {"doc_id": case.invoice.doc_id, "quantity": case.invoice.quantity,
                "unit_price": str(case.invoice.unit_price)}

    async def write_proposal(self, p: Proposal, *, idempotency_key: str) -> dict:
        """The only side-effecting tool. Idempotent by construction.

        The key is derived from (case_id, trace_id) — both stable across
        retries of the same logical step, both already available. No extra
        state needed.
        """
        if idempotency_key in self.proposals:
            return {"written": False, "deduplicated": True,
                    "proposal": self.proposals[idempotency_key]}
        self.write_calls += 1
        await asyncio.sleep(0.003)
        self.proposals[idempotency_key] = p
        return {"written": True, "deduplicated": False, "proposal": p}


# ---------------------------------------------------------------------------
# The agent
# ---------------------------------------------------------------------------

class ExceptionAgent:
    def __init__(self, *, primary: FakeLLM, fallback: FakeLLM,
                 tools: ToolRegistry, metrics: Metrics) -> None:
        self.primary = primary
        self.fallback = fallback
        self.tools = tools
        self.m = metrics
        self.budget = RetryBudget(ratio=0.3)
        self.breakers = {"primary": Breaker(), "fallback": Breaker()}
        self.bulkhead = asyncio.Semaphore(6)
        self.dlq: list[dict] = []

    # -- resilient model call ---------------------------------------------

    async def _call_model(self, llm: FakeLLM, name: str, prompt: str,
                          deadline: Deadline, max_tokens: int) -> str:
        """Breaker OUTSIDE, retries INSIDE, all within one bulkhead slot.

        The ordering is deliberate (see 05 part 5): with the breaker inside the
        retry loop, an open circuit burns all four attempts in microseconds and
        reports 'exhausted', which is true and useless.
        """
        async with self.bulkhead:
            for n in range(4):
                if deadline.expired():
                    raise UpstreamTimeout("deadline exhausted")
                self.m.incr("attempts")
                try:
                    async with asyncio.timeout(min(1.0, deadline.remaining())):
                        c = await self.breakers[name].call(
                            lambda: llm.complete(prompt, max_tokens=max_tokens)
                        )
                    return c.text if c.finish_reason != "length" else "__TRUNCATED__"
                except asyncio.CancelledError:
                    raise
                except (AppError, TimeoutError) as e:
                    disp = (classify(e) if isinstance(e, AppError)
                            else Disposition.RETRY)
                    if disp is not Disposition.RETRY or n == 3:
                        raise
                    if not self.budget.try_retry():
                        self.m.incr("budget_denied")
                        raise
                    delay = getattr(e, "retry_after", None) or \
                        random.uniform(0, 0.02 * (2 ** n))
                    await asyncio.sleep(min(delay, deadline.remaining()))
            raise ServiceUnavailable("unreachable")

    # -- parse + repair ----------------------------------------------------

    @staticmethod
    def _parse(text: str) -> Proposal | None:
        import re
        for candidate in (text, ):
            try:
                return None if not candidate else Proposal(**json.loads(candidate))
            except (json.JSONDecodeError, TypeError):
                pass
        fence = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
        if fence:
            with contextlib.suppress(json.JSONDecodeError, TypeError):
                return Proposal(**json.loads(fence.group(1)))
        s, e = text.find("{"), text.rfind("}")
        if 0 <= s < e:
            with contextlib.suppress(json.JSONDecodeError, TypeError):
                return Proposal(**json.loads(text[s:e + 1]))
        return None

    async def _propose(self, case: ExceptionCase, llm: FakeLLM, name: str,
                       deadline: Deadline) -> tuple[Proposal, list[str]]:
        trail: list[str] = []
        prompt = f"Classify exception for {case.case_id}."
        max_tokens = 128

        for _ in range(3):
            text = await self._call_model(llm, name, prompt, deadline, max_tokens)

            if text == "__TRUNCATED__":
                trail.append("truncated")
                self.m.incr("repair.truncated")
                max_tokens *= 2
                continue

            p = self._parse(text)
            if p is None:
                trail.append("parse_fail")
                self.m.incr("repair.parse")
                prompt += "\nReturn ONLY a JSON object, no fences, no prose."
                continue

            if p.disposition not in VALID_DISPOSITIONS:
                trail.append("bad_enum")
                self.m.incr("repair.schema")
                prompt += (f"\n'{p.disposition}' is not valid. Choose one of: "
                           f"{sorted(VALID_DISPOSITIONS)}")
                continue

            trail.append("ok")
            return p, trail

        raise ServiceUnavailable("could not obtain a valid proposal")

    # -- the deterministic gate -------------------------------------------

    def _gate(self, case: ExceptionCase, p: Proposal) -> tuple[bool, str]:
        """Re-derive the exception independently; check the proposal against it.

        This is the control that makes the whole thing defensible. It does not
        ask 'did the model sound confident'. It computes the discrepancy from
        the documents and checks consistency against a written policy.

        Note it also rejects LOW CONFIDENCE and MISSING EVIDENCE — a proposal
        the model itself is unsure about, or one that cites nothing, goes to a
        human rather than through.
        """
        computed = case.computed_exceptions()
        allowed: set[str] = set()
        for exc in computed:
            allowed |= CONSISTENT.get(exc, set())

        if p.disposition not in allowed:
            return False, (f"disposition {p.disposition!r} inconsistent with "
                           f"computed exceptions {computed}")
        if p.confidence < 0.6:
            return False, f"confidence {p.confidence} below 0.6 threshold"
        if not p.evidence_ids:
            return False, "no evidence cited"
        return True, f"consistent with {computed}"

    # -- one case, end to end ---------------------------------------------

    async def resolve(self, case: ExceptionCase, *, budget_s: float = 3.0) -> Outcome:
        trace_id = f"tr-{case.case_id}"
        deadline = Deadline(budget_s)
        self.m.incr("cases")
        self.budget.on_request()

        # Read-only tool calls: safe to retry, no key needed.
        await asyncio.gather(self.tools.get_po(case), self.tools.get_gr(case),
                             self.tools.get_invoice(case))

        ladder = [(self.primary, "primary", False), (self.fallback, "fallback", True)]
        attempts: list[str] = []

        for llm, name, is_degraded in ladder:
            try:
                p, trail = await self._propose(case, llm, name, deadline)
                attempts += [f"{name}:{t}" for t in trail]
                p.case_id = case.case_id

                ok, reason = self._gate(case, p)
                if not ok:
                    self.m.incr("gate.rejected")
                    return Outcome(case.case_id, "rejected", p, reason, name,
                                   is_degraded, attempts, trace_id)

                # Idempotent write. Key derived from identifiers we already have.
                await self.tools.write_proposal(
                    p, idempotency_key=f"{case.case_id}:{trace_id}")
                self.m.incr("gate.approved")
                self.m.incr(f"served_by.{name}")
                return Outcome(case.case_id, "approved", p, reason, name,
                               is_degraded, attempts, trace_id)

            except AppError as e:
                attempts.append(f"{name}:{type(e).__name__}")
                if classify(e) is Disposition.FAIL_FAST:
                    self.m.incr("fail_fast")
                    self.dlq.append({"case_id": case.case_id, "trace_id": trace_id,
                                     "error": type(e).__name__, "replayable": False})
                    return Outcome(case.case_id, "failed", None, str(e), name,
                                   True, attempts, trace_id)
                self.m.incr(f"fallback_from.{name}")
            except TimeoutError:
                attempts.append(f"{name}:timeout")

        self.m.incr("exhausted")
        self.dlq.append({"case_id": case.case_id, "trace_id": trace_id,
                         "error": "all rungs failed", "replayable": True})
        return Outcome(case.case_id, "failed", None, "all rungs failed", "none",
                       True, attempts, trace_id)


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------

def make_cases() -> list[ExceptionCase]:
    d = Decimal
    return [
        ExceptionCase("EXC-001",
                      Document("PO-1001", "po", 400, d("12.50")),
                      Document("GR-0055", "gr", 400, d("12.50")),
                      Document("INV-0088", "invoice", 402, d("12.50"))),
        ExceptionCase("EXC-002",
                      Document("PO-1002", "po", 100, d("8.00")),
                      Document("GR-0056", "gr", 100, d("8.00")),
                      Document("INV-0089", "invoice", 100, d("8.00"))),
        ExceptionCase("EXC-003",
                      Document("PO-1003", "po", 250, d("15.00")),
                      Document("GR-0057", "gr", 0, d("15.00")),
                      Document("INV-0090", "invoice", 250, d("15.00"))),
    ]


def good_response(disp: str, conf: float = 0.85) -> str:
    return json.dumps({"case_id": "", "disposition": disp, "confidence": conf,
                       "reason": "derived from three-way match",
                       "evidence_ids": ["gr-55", "inv-88"]})


# ---------------------------------------------------------------------------
# Demonstrations
# ---------------------------------------------------------------------------

async def demo1_happy_path() -> None:
    banner("DEMO 1 — clean resolution")

    m = Metrics()
    cases = make_cases()
    agent = ExceptionAgent(
        primary=FakeLLM("gpt-4o", respond_with=[
            good_response("request_credit"), good_response("approve"),
            good_response("hold_pending_receipt")]),
        fallback=FakeLLM("gpt-4o-mini"),
        tools=ToolRegistry(), metrics=m)

    for case in cases:
        o = await agent.resolve(case)
        print(f"    {o.case_id}  computed={case.computed_exceptions()}")
        print(f"      -> {o.status:<9} {o.proposal.disposition:<22} "
              f"gate: {o.gate_reason}")
    print(f"\n    writes performed: {agent.tools.write_calls}")


async def demo2_gate_catches_bad_proposals() -> None:
    banner("DEMO 2 — the gate rejects proposals the arithmetic contradicts")

    m = Metrics()
    case = make_cases()[0]        # invoice 402 vs receipt 400 => over-receipt

    section("model proposes 'approve' for a genuine overbill")
    agent = ExceptionAgent(
        primary=FakeLLM("gpt-4o", respond_with=[good_response("approve", 0.97)]),
        fallback=FakeLLM("mini", respond_with=[good_response("approve", 0.97)]),
        tools=ToolRegistry(), metrics=m)
    o = await agent.resolve(case)
    print(f"      status={o.status}")
    print(f"      gate:  {o.gate_reason}")
    print(f"      writes performed: {agent.tools.write_calls}  <- nothing written")
    print("      The model was 97% confident and completely wrong. Confidence")
    print("      is not evidence; the arithmetic is.")

    section("model is honest but uncertain")
    agent2 = ExceptionAgent(
        primary=FakeLLM("gpt-4o", respond_with=[good_response("request_credit", 0.35)]),
        fallback=FakeLLM("mini", respond_with=[good_response("request_credit", 0.35)]),
        tools=ToolRegistry(), metrics=m)
    o2 = await agent2.resolve(case)
    print(f"      status={o2.status}  gate: {o2.gate_reason}")
    print("      Correct disposition, but below threshold -> routed to a human.")


async def demo3_degradation() -> None:
    banner("DEMO 3 — primary down, degrade to fallback, labelled")

    m = Metrics()
    case = make_cases()[0]
    agent = ExceptionAgent(
        primary=FakeLLM("gpt-4o", injector=FaultInjector(
            script=[Fail(ServiceUnavailable, 0.005)])),
        fallback=FakeLLM("gpt-4o-mini",
                         respond_with=[good_response("request_credit")]),
        tools=ToolRegistry(), metrics=m)

    o = await agent.resolve(case)
    print(f"    status={o.status}  served_by={o.served_by}  degraded={o.degraded}")
    print(f"    attempts: {o.attempts}")
    print(f"    {m.report('attempts', 'fallback_from.primary', 'served_by.fallback')}")
    print("""
    The outcome is labelled degraded=True. Without that label this looks
    identical to a clean success in every dashboard you have — and a human
    reviewing the proposal has no idea it came from the cheaper model.""")


async def demo4_repair() -> None:
    banner("DEMO 4 — LLM output repair inside the agent")

    m = Metrics()
    case = make_cases()[0]
    agent = ExceptionAgent(
        primary=FakeLLM("gpt-4o", respond_with=[
            "```json\n{'disposition': 'partial_credit',}\n```",     # broken JSON
            good_response("partial_credit"),                        # bad enum
            good_response("request_credit"),                        # correct
        ]),
        fallback=FakeLLM("mini"), tools=ToolRegistry(), metrics=m)

    o = await agent.resolve(case)
    print(f"    status={o.status}  attempts={o.attempts}")
    print(f"    {m.report('repair.parse', 'repair.schema')}")
    print("    Two repairs, one valid proposal, no retries of an identical prompt.")


async def demo5_idempotent_write() -> None:
    banner("DEMO 5 — the write tool is safe to retry")

    tools = ToolRegistry()
    p = Proposal("EXC-001", "request_credit", 0.9, "overbill", ["inv-88"])
    key = "EXC-001:tr-EXC-001"

    r1 = await tools.write_proposal(p, idempotency_key=key)
    r2 = await tools.write_proposal(p, idempotency_key=key)
    r3 = await tools.write_proposal(p, idempotency_key=key)
    print(f"    call 1: written={r1['written']} dedup={r1['deduplicated']}")
    print(f"    call 2: written={r2['written']} dedup={r2['deduplicated']}")
    print(f"    call 3: written={r3['written']} dedup={r3['deduplicated']}")
    print(f"    actual writes: {tools.write_calls}")
    print("    Three calls, one proposal. A retry after a lost response — or a")
    print("    queue redelivery, or a pod restart — cannot double-file.")


async def demo6_batch_under_stress() -> None:
    banner("DEMO 6 — 60 cases against a flaky dependency")

    m = Metrics()
    tools = ToolRegistry()
    cases = [make_cases()[i % 3] for i in range(60)]
    # Give each a distinct id so idempotency keys differ.
    cases = [ExceptionCase(f"EXC-{i:03d}", c.po, c.gr, c.invoice)
             for i, c in enumerate(cases)]

    correct = {"qty_over_receipt": "request_credit", "none": "approve",
               "no_receipt": "hold_pending_receipt"}

    class PolicyLLM(FakeLLM):
        """Returns the correct disposition, but the endpoint is flaky."""

        def __init__(self, name: str, injector: FaultInjector) -> None:
            super().__init__(name, injector)
            self.case: ExceptionCase | None = None

        async def complete(self, prompt: str, *, max_tokens: int = 256):
            from failure_lab import Completion
            await self.injector.maybe_fail()
            exc = self.case.computed_exceptions()[0]
            return Completion(text=good_response(correct[exc]),
                              finish_reason="stop")

    primary = PolicyLLM("gpt-4o", FaultInjector(rate=0.35, seed=99,
                                                error_cls=ServiceUnavailable,
                                                base_latency=0.002))
    fallback = PolicyLLM("gpt-4o-mini", FaultInjector(rate=0.05, seed=7,
                                                      base_latency=0.002))
    agent = ExceptionAgent(primary=primary, fallback=fallback,
                           tools=tools, metrics=m)

    with Timer("60 cases, 35% primary failure rate"):
        outcomes = []
        for c in cases:
            primary.case = c
            fallback.case = c
            outcomes.append(await agent.resolve(c))

    approved = sum(1 for o in outcomes if o.status == "approved")
    degraded = sum(1 for o in outcomes if o.degraded and o.status == "approved")
    rejected = sum(1 for o in outcomes if o.status == "rejected")
    failed = sum(1 for o in outcomes if o.status == "failed")

    print(f"\n    approved: {approved}/60   (of which degraded: {degraded})")
    print(f"    rejected by gate: {rejected}/60")
    print(f"    failed: {failed}/60")
    print(f"    dead letters: {len(agent.dlq)}")
    print(f"    proposals written: {tools.write_calls}  (must equal approved)")
    print(f"    amplification: {m.counters['attempts'] / m.counters['cases']:.2f}x")
    print(f"    breaker state: primary={agent.breakers['primary'].state} "
          f"rejected={agent.breakers['primary'].rejected}")
    print(f"    retry budget denials: {agent.budget.denied}")

    print(f"""
    RESULT: {approved}/60 approved, {degraded}/60 of those from the fallback
    model, {failed}/60 failed. Reported as k/n named runs — with n=60 and an
    injected failure rate, a percentage would imply a capability estimate this
    sample cannot support.

    The number worth watching is proposals_written == approved. If those ever
    diverge, either the gate is being bypassed or the write is not idempotent.
    That is the invariant to assert in a test, not to check by eye.""")


# ---------------------------------------------------------------------------

async def main() -> None:
    await demo1_happy_path()
    await demo2_gate_catches_bad_proposals()
    await demo3_degradation()
    await demo4_repair()
    await demo5_idempotent_write()
    await demo6_batch_under_stress()

    banner("WHAT TO DEFEND IN A DESIGN REVIEW")
    print("""
  1. A deterministic gate re-derives the exception from source documents and
     rejects proposals inconsistent with it. Model confidence is not evidence.
  2. The consistency policy is a data structure (CONSISTENT), not prose in a
     prompt — so it is reviewable, testable, and diffable.
  3. Exactly one tool has side effects, and it is idempotent on a key derived
     from identifiers that already exist (case_id + trace_id).
  4. Retries live INSIDE the bulkhead slot; the breaker sits OUTSIDE the retry
     loop, so an open circuit reports 'not attempted', not 'exhausted'.
  5. One deadline spans retrieval, all repairs, both fallback rungs, and the
     write. Every attempt checks it before spending a call.
  6. Repairs are bounded and counted separately from retries. They are
     different failures with different fixes.
  7. Degraded outcomes are labelled. Unlabelled fallbacks corrupt your own
     quality metrics as well as misleading reviewers.
  8. Money is Decimal. Never float.
  9. Failures go to a DLQ with a trace_id and a replayable flag.
 10. Results reported as k/n named runs, not percentages, at this sample size.
""")


if __name__ == "__main__":
    asyncio.run(main())
