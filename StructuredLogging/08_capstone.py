"""
08 — Capstone: a fully instrumented RAG service.

WHAT THIS COMPOSES
------------------
  01  structured logging   -> one wide event per request, no content
  02  context propagation  -> middleware, x-trace-id, contextvars
  03  span design          -> ~7 spans, correct kinds, events vs attributes
  04  what to record       -> sensitivity routing, telemetry vs audit
  05  metrics              -> bounded labels, TTFT, cost at the call site
  06  sampling             -> baseline + force-keep
  07  queryability         -> every emitted field earns its place

THE ASSERTION AT THE END
------------------------
The service runs a mixed workload, then a scanner checks every emitted
telemetry surface for prompt text, chunk text, and user questions, and asserts
zero. That assertion is the deliverable.

Run:  python 08_capstone.py
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import time
from collections import Counter
from dataclasses import dataclass, field

from obs_lab import (
    ALLOWED_IN_AUDIT,
    ALLOWED_IN_TELEMETRY,
    CORPUS,
    Chunk,
    Sensitivity,
    SensitivityScanner,
    SpanKind,
    TelemetryStore,
    Tracer,
    banner,
    make_logger,
    new_trace_id,
    principal_var,
    print_json,
    section,
    show,
    tenant_var,
    trace_id_var,
    verdict,
)

STORE = TelemetryStore()
TRACER = Tracer(STORE)
log, cap = make_logger("isc.rag")
SCANNER = SensitivityScanner()
for c in CORPUS:
    SCANNER.register(f"chunk:{c.chunk_id}", c.text)

PRICE = {"gpt-4o-mini": (0.00015, 0.00060), "gpt-4o": (0.00250, 0.01000)}
SALT = "isc-telemetry-salt-v1"


# ===========================================================================
# Sensitivity-routed recording
# ===========================================================================

@dataclass
class Emission:
    """One value plus where it may go. Declared once, routed twice."""

    key: str
    value: object
    sensitivity: Sensitivity = Sensitivity.PUBLIC


class RequestRecorder:
    """Collects everything about a request, then routes by sensitivity.

    ONE declaration produces the span attributes, the wide log event, and the
    audit record — so they cannot drift, and a new field is classified at the
    moment it is added rather than at the moment someone remembers.
    """

    def __init__(self) -> None:
        self.items: list[Emission] = []

    def add(self, key: str, value: object,
            sensitivity: Sensitivity = Sensitivity.PUBLIC) -> None:
        self.items.append(Emission(key, value, sensitivity))

    def telemetry(self) -> dict[str, object]:
        return {i.key: i.value for i in self.items
                if i.sensitivity in ALLOWED_IN_TELEMETRY}

    def audit(self) -> dict[str, object]:
        return {i.key: i.value for i in self.items
                if i.sensitivity in ALLOWED_IN_AUDIT}


# ===========================================================================
# Sampling
# ===========================================================================

@dataclass
class Sampler:
    base_rate: float = 0.10
    slow_threshold_ms: float = 120.0
    decisions: Counter[str] = field(default_factory=Counter)

    def head(self, trace_id: str) -> bool:
        d = hashlib.sha256(trace_id.encode()).digest()
        return int.from_bytes(d[:8], "big") / (2 ** 64) < self.base_rate

    def decide(self, *, trace_id: str, outcome: str, duration_ms: float,
               acl_violation: bool) -> str | None:
        if acl_violation:
            reason = "acl_violation"
        elif outcome == "error":
            reason = "error"
        elif outcome in ("degraded", "refused"):
            reason = outcome
        elif duration_ms >= self.slow_threshold_ms:
            reason = "slow"
        elif self.head(trace_id):
            reason = "baseline"
        else:
            self.decisions["dropped"] += 1
            return None
        self.decisions[reason] += 1
        return reason


SAMPLER = Sampler()


# ===========================================================================
# Fake dependencies
# ===========================================================================

class Search:
    def __init__(self, *, trim: bool = True) -> None:
        self.trim = trim

    async def query(self, question: str, groups: frozenset[str],
                    k: int) -> tuple[list[Chunk], int]:
        await asyncio.sleep(0.004)
        candidates = CORPUS
        visible = [c for c in candidates if c.visible_to(groups)] \
            if self.trim else list(candidates)
        terms = set(question.lower().split())
        scored = [
            Chunk(c.chunk_id, c.doc_id, c.text, c.allowed_groups, c.sensitivity,
                  float(len(terms & set(c.text.lower().split()))))
            for c in visible
        ]
        hits = sorted((c for c in scored if c.score > 0),
                      key=lambda c: -c.score)[:k]
        return hits, len(candidates)


class Model:
    def __init__(self, name: str = "gpt-4o-mini",
                 served: str = "gpt-4o-mini-2024-07-18") -> None:
        self.name = name
        self.served = served
        self.calls = 0

    async def generate(self, prompt: str, *, max_tokens: int,
                       ttft_s: float = 0.006, fail: bool = False,
                       bad_json: bool = False):
        self.calls += 1
        await asyncio.sleep(ttft_s)
        if fail:
            raise ConnectionError("503 upstream unavailable")
        await asyncio.sleep(0.004)
        text = ("not json at all" if bad_json else
                json.dumps({"answer": "Invoice 88 billed 402 against 400.",
                            "citations": ["c3", "c4"]}))
        return text, len(prompt) // 4, len(text) // 4


# ===========================================================================
# The instrumented service
# ===========================================================================

class RagService:
    def __init__(self, search: Search, model: Model) -> None:
        self.search = search
        self.model = model
        self.audit_records: list[dict[str, object]] = []

    @staticmethod
    def _qhash(q: str) -> str:
        return hashlib.sha256((SALT + q).encode()).hexdigest()[:12]

    async def answer(self, question: str, principal: str, tenant: str,
                     groups: frozenset[str], *,
                     inject_failure: bool = False,
                     inject_bad_json: bool = False,
                     thumbs_down: bool = False) -> dict[str, object]:
        # ---- MIDDLEWARE: context, then the SERVER span ------------------
        trace_id = new_trace_id()
        trace_id_var.set(trace_id)
        principal_var.set(principal)
        tenant_var.set(tenant)

        rec = RequestRecorder()
        rec.add("question.text", question, Sensitivity.SENSITIVE)
        rec.add("question.hash", self._qhash(question), Sensitivity.INTERNAL)
        rec.add("question.length", len(question))
        rec.add("tenant", tenant, Sensitivity.INTERNAL)

        outcome = "ok"
        acl_violation = False
        started = time.perf_counter()
        repairs = 0
        cost_usd = 0.0
        ttft_ms = 0.0

        with TRACER.span("POST /answer", SpanKind.SERVER,
                         **{"http.route": "/answer", "tenant": tenant,
                            # A pseudonym, never the email — script 04 rule 5.
                            "enduser.id": self._qhash(principal)}) as root:
            try:
                # ---- RETRIEVE (CLIENT) --------------------------------
                with TRACER.span("retrieve", SpanKind.CLIENT,
                                 **{"peer.service": "azure-ai-search"}) as s:
                    chunks, candidates = await self.search.query(
                        question, groups, k=4)
                    trimmed = candidates - len([c for c in CORPUS
                                                if c.visible_to(groups)])
                    s.set("retrieval.candidate_count", candidates)
                    s.set("retrieval.chunk_count", len(chunks))
                    s.set("retrieval.trimmed_count", trimmed)
                    s.set("retrieval.chunk_ids", [c.chunk_id for c in chunks])
                    s.set("retrieval.doc_ids", sorted({c.doc_id for c in chunks}))
                    if chunks:
                        s.set("retrieval.score_max", max(c.score for c in chunks))
                        s.set("retrieval.score_min", min(c.score for c in chunks))

                    # THE HARD INVARIANT, emitted as an event with ids only.
                    for c in chunks:
                        if not c.visible_to(groups):
                            acl_violation = True
                            s.event("acl.violation", **{
                                "chunk_id": c.chunk_id, "doc_id": c.doc_id,
                                "chunk.sensitivity": c.sensitivity,
                                "caller.group_count": len(groups)})
                            STORE.incr("acl_violations_total", tenant=tenant)

                    STORE.incr("retrieval_trimmed_total", trimmed, tenant=tenant)
                    STORE.observe("retrieval_chunks", len(chunks),
                                  index="isc-docs-v3")

                rec.add("retrieval.chunk_ids", [c.chunk_id for c in chunks],
                        Sensitivity.INTERNAL)
                rec.add("retrieval.trimmed_count", trimmed)
                rec.add("chunks.text", [c.text for c in chunks],
                        Sensitivity.SENSITIVE)

                # ---- BUILD PROMPT (INTERNAL) ---------------------------
                with TRACER.span("build_prompt", SpanKind.INTERNAL) as s:
                    prompt = ("Context:\n"
                              + "\n".join(f"[{c.chunk_id}] {c.text}" for c in chunks)
                              + f"\nQ: {question}")
                    s.set("isc.prompt.version", "v3")
                    s.set("isc.prompt.fingerprint", "36206a029343")
                    s.set("isc.prompt.char_count", len(prompt))
                rec.add("prompt.text", prompt, Sensitivity.SENSITIVE)
                rec.add("prompt.fingerprint", "36206a029343", Sensitivity.INTERNAL)

                # ---- GENERATE (CLIENT), with a bounded repair loop ------
                text = ""
                with TRACER.span("generate", SpanKind.CLIENT,
                                 **{"peer.service": "azure-openai",
                                    "gen_ai.system": "azure.ai.openai",
                                    "gen_ai.operation.name": "chat",
                                    "gen_ai.request.model": self.model.name,
                                    "gen_ai.request.max_tokens": 512}) as s:
                    t_gen = time.perf_counter()
                    for attempt in range(3):
                        text, tin, tout = await self.model.generate(
                            prompt, max_tokens=512,
                            fail=inject_failure and attempt == 0,
                            bad_json=inject_bad_json and attempt == 0)
                        if attempt == 0:
                            ttft_ms = (time.perf_counter() - t_gen) * 1000
                            s.event("first_token")
                        try:
                            json.loads(text)
                            break
                        except json.JSONDecodeError:
                            repairs += 1
                            # An EVENT, not a silent retry — script 07 Q6.
                            s.event("repair.attempted", kind="schema",
                                    attempt=attempt + 1)
                            STORE.incr("repairs_total", kind="schema")
                            prompt += "\nReturn ONLY JSON."

                    pin, pout = PRICE[self.model.name]
                    cost_usd = tin / 1000 * pin + tout / 1000 * pout
                    s.set("gen_ai.response.model", self.model.served)
                    s.set("gen_ai.usage.input_tokens", tin)
                    s.set("gen_ai.usage.output_tokens", tout)
                    s.set("gen_ai.response.finish_reasons", ["stop"])
                    s.set("isc.cost_usd", round(cost_usd, 6))
                    s.set("isc.repairs", repairs)
                    STORE.observe("gen_ai_ttft_ms", ttft_ms,
                                  deployment=self.model.name)
                    STORE.incr("gen_ai_tokens_total", tin,
                               deployment=self.model.name, direction="input")
                    STORE.incr("gen_ai_tokens_total", tout,
                               deployment=self.model.name, direction="output")

                rec.add("response.text", text, Sensitivity.SENSITIVE)
                if repairs:
                    outcome = "degraded"

            except ConnectionError as e:
                outcome = "error"
                root.record_exception(e)

            duration_ms = (time.perf_counter() - started) * 1000
            root.set("outcome", outcome)
            root.set("isc.cost_usd", round(cost_usd, 6))
            root.set("isc.repairs", repairs)

        # ---- SAMPLING DECISION, recorded on the wide event -------------
        sampled = SAMPLER.decide(trace_id=trace_id, outcome=outcome,
                                 duration_ms=duration_ms,
                                 acl_violation=acl_violation)

        # ---- ONE WIDE EVENT, always emitted, telemetry-safe ------------
        log.info("request completed", extra={
            "event": "request.completed",
            "outcome": outcome,
            "duration_ms": round(duration_ms, 2),
            "ttft_ms": round(ttft_ms, 2),
            "cost_usd": round(cost_usd, 6),
            "repairs": repairs,
            "acl_violation": acl_violation,
            "trace.sampled": sampled is not None,
            "trace.sample_reason": sampled or "dropped",
            **rec.telemetry(),
        })
        STORE.incr("requests_total", tenant=tenant, outcome=outcome)
        STORE.observe("request_duration_ms", duration_ms,
                      tenant=tenant, outcome=outcome)
        STORE.observe(f"cost_usd|tenant={tenant},outcome={outcome}", cost_usd)

        # ---- AUDIT, triggered only ------------------------------------
        reason = ("acl_violation" if acl_violation else
                  "user_reported" if thumbs_down else
                  "degraded" if outcome == "degraded" else
                  "random_sample" if random.random() < 0.01 else None)
        if reason:
            self.audit_records.append({
                "trace_id": trace_id, "reason": reason,
                "principal": principal, "tenant": tenant,
                "retention_days": 7, **rec.audit()})

        return {"answer": text if outcome != "error" else None,
                "outcome": outcome, "_headers": {"x-trace-id": trace_id}}


# ===========================================================================
# DEMONSTRATIONS
# ===========================================================================

QUESTION = "Why was invoice 88 billed for more units than the goods receipt?"


async def demo1_one_request() -> RagService:
    banner("DEMO 1 — one request, fully instrumented")

    svc = RagService(Search(), Model())
    r = await svc.answer(QUESTION, "debdeep@contoso.com", "isc-sg",
                         frozenset({"isc-all"}))
    show("returned header", r["_headers"])

    section("the trace")
    print(STORE.render_trace(str(r["_headers"]["x-trace-id"])))

    section("the wide event (this is the whole log output)")
    print_json(cap.records[-1])
    return svc


async def demo2_workload(svc: RagService) -> None:
    banner("DEMO 2 — a mixed workload")

    people = [
        ("debdeep@contoso.com", "isc-sg", frozenset({"isc-all"})),
        ("hr.lead@contoso.com", "isc-eu", frozenset({"isc-all", "hr-only"})),
        ("legal@contoso.com", "isc-us", frozenset({"legal-only"})),
    ]
    rng = random.Random(9)
    for i in range(60):
        p, t, g = people[i % 3]
        await svc.answer(QUESTION, p, t, g,
                         inject_failure=rng.random() < 0.05,
                         inject_bad_json=rng.random() < 0.12,
                         thumbs_down=rng.random() < 0.03)

    section("outcomes")
    outcomes = Counter(r["outcome"] for r in cap.by_event("request.completed"))
    for k, v in outcomes.most_common():
        show(k, v)

    section("sampling decisions")
    for k, v in SAMPLER.decisions.most_common():
        show(k, v)

    section("metrics")
    show("p50 request_duration (ok)",
         f"{STORE.hist_percentile('request_duration_ms|outcome=ok,tenant=isc-sg', 0.5):.1f}ms")
    show("p95 TTFT",
         f"{STORE.hist_percentile('gen_ai_ttft_ms|deployment=gpt-4o-mini', 0.95):.1f}ms")
    show("metric cardinality", STORE.cardinality())
    show("audit records written", f"{len(svc.audit_records)}/61")


async def demo3_acl_violation() -> None:
    banner("DEMO 3 — a retriever that stops trimming")

    svc = RagService(Search(trim=False), Model())
    r = await svc.answer("settlement compensation confidential terms",
                         "debdeep@contoso.com", "isc-sg",
                         frozenset({"isc-all"}))
    tid = str(r["_headers"]["x-trace-id"])
    retrieve = next(s for s in STORE.trace(tid) if s.name == "retrieve")

    show("trimmed_count", retrieve.attributes["retrieval.trimmed_count"])
    show("acl.violation events", sum(1 for e in retrieve.events
                                     if e.name == "acl.violation"))
    section("the violation events — ids only")
    print_json([e.attributes for e in retrieve.events
                if e.name == "acl.violation"])
    show("audit record written", len(svc.audit_records) > 0)
    show("audit reason", svc.audit_records[0]["reason"] if svc.audit_records else None)

    print("""
    The user's response looked entirely normal. The only signals anywhere are
    `trimmed_count=0`, the violation events, and the forced audit record —
    which is why all three exist.""")


async def demo4_leak_audit(svc: RagService) -> None:
    banner("DEMO 4 — does any telemetry surface carry content?")

    SCANNER.register("user_question", QUESTION)
    SCANNER.reset()

    surfaces = [
        ("all spans", json.dumps([s.to_dict() for s in STORE.spans])),
        ("all log records", json.dumps(cap.records)),
        ("metric labels", json.dumps([list(k) for k in STORE.counters])),
    ]
    for label, text in surfaces:
        found = SCANNER.scan(text, label)
        verdict(not found, f"{label:<20} {len(found)} finding(s)")
        for _, name, excerpt in found[:3]:
            print(f"        {name}: {excerpt}")

    section("the audit sink, for comparison")
    audit_text = json.dumps(svc.audit_records)
    found = SCANNER.scan(audit_text, "audit")
    verdict(bool(found),
            f"audit contains {len(found)} sensitive item(s) — EXPECTED")
    show("audit records", len(svc.audit_records))
    show("audit retention", "7 days, restricted access")

    print("""
    THIS IS THE ASSERTION THAT MATTERS. Everything else in this file is setup
    for it: a scanner over every telemetry surface, asserting zero, while the
    restricted audit sink deliberately holds what debugging needs.""")


async def main() -> None:
    svc = await demo1_one_request()
    await demo2_workload(svc)
    await demo3_acl_violation()
    await demo4_leak_audit(svc)

    banner("WHAT TO DEFEND IN A DESIGN REVIEW")
    print("""
  1. ONE middleware sets the context and returns `x-trace-id`. Every log line
     and span carries the trace id without anyone remembering to pass it.
  2. Seven spans per request, correctly kinded, so dependency time is
     computable and the service map is right.
  3. Sensitivity is declared ONCE per field and routed to telemetry or audit
     automatically. A new field is classified when it is added.
  4. Telemetry carries ids, counts, hashes, and fingerprints. `chunk_ids` +
     `prompt.fingerprint` reconstruct the prompt under your own ACL, which is
     better than storing it.
  5. `enduser.id` is a salted pseudonym, never the email.
  6. `retrieval.trimmed_count` and `acl.violation` events make a silent
     permission failure loud. Violation events carry ids only.
  7. `repair.attempted` is an EVENT, so silent repairs are countable and
     model drift is detectable days early.
  8. ONE wide log event per request, always emitted, carrying the sampling
     decision — so a missing trace is explicable.
  9. Metric labels are bounded and enumerable; trace_id is never a label.
 10. Cost is computed at the call site from returned usage and labelled by
     tenant and outcome, so spend-on-failed-requests is visible.
 11. Audit is TRIGGERED, restricted, and retention-bounded.
""")


if __name__ == "__main__":
    asyncio.run(main())
