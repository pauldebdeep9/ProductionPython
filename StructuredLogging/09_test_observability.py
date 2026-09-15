"""
09 — Testing observability.

Run:  pytest 09_test_observability.py -v

WHY TEST TELEMETRY AT ALL
-------------------------
Because it fails silently. A broken metric does not raise; it just stops
appearing on a dashboard nobody was looking at until the incident. A dropped
trace_id does not error; it produces a partial trace six weeks later. A field
that starts carrying prompt text does not warn you; it quietly accumulates
document content in a store with the wrong access model.

Telemetry is production code that nothing else exercises. That makes it
exactly the kind of code that needs tests.

THE FOUR THINGS WORTH ASSERTING
-------------------------------
  1. NO CONTENT LEAKS. The highest-value test in this file, and the only way
     "we don't log prompts" becomes a control rather than an intention.
  2. TRACE CONTINUITY. Every span in a request shares one trace id, and the
     parent-child structure is what you think it is.
  3. THE FIELDS QUERIES DEPEND ON EXIST. A dashboard is a consumer of your
     telemetry schema; renaming a field breaks it silently.
  4. CARDINALITY IS BOUNDED. One unbounded label is a cost incident.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import pathlib
import sys

import pytest

from obs_lab import (
    CORPUS,
    SensitivityScanner,
    SpanKind,
    StatusCode,
    make_logger,
    new_span_id,
    new_trace_id,
    principal_var,
    span_id_var,
    tenant_var,
    trace_id_var,
)

_spec = importlib.util.spec_from_file_location(
    "capstone", pathlib.Path(__file__).parent / "08_capstone.py")
assert _spec and _spec.loader
capstone = importlib.util.module_from_spec(_spec)
sys.modules["capstone"] = capstone
_spec.loader.exec_module(capstone)

QUESTION = capstone.QUESTION


@pytest.fixture(autouse=True)
def clean_state():
    """Telemetry state is global by nature. Reset it around every test, or
    assertions depend on execution order — the same discipline the config
    tutorial applied to os.environ."""
    capstone.STORE.reset()
    capstone.cap.reset()
    capstone.SCANNER.reset()
    capstone.SAMPLER.decisions.clear()
    capstone.svc_audit = None
    trace_id_var.set("")
    span_id_var.set("")
    principal_var.set("")
    tenant_var.set("")
    yield


@pytest.fixture
def service():
    return capstone.RagService(capstone.Search(), capstone.Model())


@pytest.fixture
def scanner() -> SensitivityScanner:
    s = SensitivityScanner()
    s.register("question", QUESTION)
    for c in CORPUS:
        s.register(f"chunk:{c.chunk_id}", c.text)
    return s


# ===========================================================================
# 1. CONTENT LEAKS — the tests that matter most
# ===========================================================================

@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ["spans", "logs", "metric_labels"])
async def test_no_content_in_telemetry(service, scanner, surface) -> None:
    """Parametrised by SURFACE so a failure names which one broke.

    "A secret leaked somewhere" and "the span exporter started including
    chunk text" are very different investigations.
    """
    await service.answer(QUESTION, "debdeep@contoso.com", "isc-sg",
                         frozenset({"isc-all"}))

    if surface == "spans":
        text = json.dumps([s.to_dict() for s in capstone.STORE.spans])
    elif surface == "logs":
        text = json.dumps(capstone.cap.records)
    else:
        text = json.dumps([list(k) for k in capstone.STORE.counters])

    findings = scanner.scan(text, surface)
    assert not findings, scanner.report()


@pytest.mark.asyncio
async def test_scanner_would_catch_a_real_leak(service, scanner) -> None:
    """THE NEGATIVE CONTROL, and the most important test here.

    Without it, the three tests above could pass because the scanner is
    broken, because nothing was registered, or because the fixture's values
    do not match what the service actually used.
    """
    leaky = json.dumps({"prompt": f"Context: {CORPUS[0].text}\nQ: {QUESTION}"})
    findings = scanner.scan(leaky, "deliberate")
    assert findings, "the scanner failed to detect plain content"
    names = {n for _, n, _ in findings}
    assert "question" in names or "chunk:c1" in names


@pytest.mark.asyncio
async def test_no_email_addresses_in_telemetry(service, scanner) -> None:
    """Regression test for a bug this tutorial shipped: the logging filter
    injected `principal` verbatim, putting a user's email on every log line.

    Personal identifiers must be pseudonymous in telemetry (script 04, rule 5).
    """
    await service.answer(QUESTION, "debdeep@contoso.com", "isc-sg",
                         frozenset({"isc-all"}))
    text = json.dumps(capstone.cap.records) + json.dumps(
        [s.to_dict() for s in capstone.STORE.spans])
    assert "debdeep@contoso.com" not in text, (
        "a raw email address reached telemetry; it must be hashed"
    )
    assert any("enduser_id" in r for r in capstone.cap.records), (
        "the pseudonymous id should still be present — it is what makes "
        "per-user grouping possible without storing the identity"
    )


@pytest.mark.asyncio
async def test_audit_sink_deliberately_holds_content(service) -> None:
    """The complement: the restricted sink MUST contain what debugging needs,
    or the split has cost you the content without buying anything."""
    await service.answer(QUESTION, "debdeep@contoso.com", "isc-sg",
                         frozenset({"isc-all"}), thumbs_down=True)
    assert service.audit_records, "a thumbs-down must trigger an audit record"
    rec = service.audit_records[0]
    assert QUESTION in json.dumps(rec)
    assert rec["retention_days"] <= 30, "audit retention must be bounded"
    assert rec["reason"] == "user_reported"


# ===========================================================================
# 2. TRACE CONTINUITY
# ===========================================================================

@pytest.mark.asyncio
async def test_all_spans_share_one_trace_id(service) -> None:
    r = await service.answer(QUESTION, "a@contoso.com", "isc-sg",
                             frozenset({"isc-all"}))
    expected = r["_headers"]["x-trace-id"]
    ids = {s.trace_id for s in capstone.STORE.spans}
    assert ids == {expected}, f"trace fragmented across {len(ids)} ids"


@pytest.mark.asyncio
async def test_span_tree_has_exactly_one_root(service) -> None:
    """Two roots means a span was created outside the request context —
    usually a background task that lost its parent."""
    await service.answer(QUESTION, "a@contoso.com", "isc-sg",
                         frozenset({"isc-all"}))
    roots = [s for s in capstone.STORE.spans if s.parent_id is None]
    assert len(roots) == 1, f"expected 1 root span, got {len(roots)}"
    assert roots[0].kind is SpanKind.SERVER


@pytest.mark.asyncio
async def test_every_child_has_a_real_parent(service) -> None:
    """A dangling parent_id renders as a broken tree in every backend."""
    await service.answer(QUESTION, "a@contoso.com", "isc-sg",
                         frozenset({"isc-all"}))
    ids = {s.span_id for s in capstone.STORE.spans}
    for s in capstone.STORE.spans:
        if s.parent_id is not None:
            assert s.parent_id in ids, (
                f"span {s.name} references a parent that was never exported"
            )


@pytest.mark.asyncio
async def test_concurrent_requests_do_not_share_context() -> None:
    """The failure this catches is severe: one user's principal attached to
    another user's trace. It only appears under concurrency, so a sequential
    test will never find it."""
    svc = capstone.RagService(capstone.Search(), capstone.Model())
    results = await asyncio.gather(*[
        svc.answer(QUESTION, f"user{i}@contoso.com", f"tenant-{i}",
                   frozenset({"isc-all"}))
        for i in range(6)
    ])
    trace_ids = [r["_headers"]["x-trace-id"] for r in results]
    assert len(set(trace_ids)) == 6, "trace ids collided across requests"

    by_trace: dict[str, set[str]] = {}
    for s in capstone.STORE.spans:
        by_trace.setdefault(s.trace_id, set()).add(
            str(s.attributes.get("tenant", "")))
    for tid, tenants in by_trace.items():
        real = {t for t in tenants if t}
        assert len(real) <= 1, f"trace {tid[:8]} mixed tenants: {real}"


@pytest.mark.asyncio
async def test_spans_exported_on_the_failure_path(service) -> None:
    """A span exported only on success gives you a trace with a hole exactly
    where the failure was — the least useful possible outcome."""
    svc = capstone.RagService(capstone.Search(), capstone.Model())
    await svc.answer(QUESTION, "a@contoso.com", "isc-sg",
                     frozenset({"isc-all"}), inject_failure=True)
    names = {s.name for s in capstone.STORE.spans}
    assert "POST /answer" in names
    assert "generate" in names, "the failing span must still be exported"


# ===========================================================================
# 3. THE FIELDS QUERIES DEPEND ON
# ===========================================================================

REQUIRED_SPAN_ATTRS = {
    "retrieve": ["retrieval.chunk_count", "retrieval.trimmed_count",
                 "retrieval.chunk_ids", "peer.service"],
    "generate": ["gen_ai.request.model", "gen_ai.response.model",
                 "gen_ai.usage.input_tokens", "gen_ai.usage.output_tokens",
                 "isc.cost_usd", "peer.service"],
    "POST /answer": ["outcome", "tenant", "enduser.id"],
}


@pytest.mark.asyncio
@pytest.mark.parametrize("span_name", sorted(REQUIRED_SPAN_ATTRS))
async def test_required_attributes_present(service, span_name) -> None:
    """A SCHEMA TEST for telemetry.

    Every one of these fields backs a query from script 07. Renaming or
    dropping one breaks a dashboard silently — the query returns zero rows,
    which looks like "the problem stopped".

    Treating the telemetry schema as an API with tests is the only thing that
    prevents that.
    """
    await service.answer(QUESTION, "a@contoso.com", "isc-sg",
                         frozenset({"isc-all"}))
    spans = [s for s in capstone.STORE.spans if s.name == span_name]
    assert spans, f"no span named {span_name!r} was emitted"
    for attr in REQUIRED_SPAN_ATTRS[span_name]:
        assert attr in spans[0].attributes, (
            f"{span_name} is missing {attr!r}, which a saved query depends on"
        )


@pytest.mark.asyncio
async def test_wide_event_has_the_analytical_fields(service) -> None:
    await service.answer(QUESTION, "a@contoso.com", "isc-sg",
                         frozenset({"isc-all"}))
    events = capstone.cap.by_event("request.completed")
    assert len(events) == 1, "exactly one wide event per request"
    for field in ("outcome", "duration_ms", "cost_usd", "repairs",
                  "trace_id", "tenant", "trace.sampled"):
        assert field in events[0], f"wide event missing {field!r}"


@pytest.mark.asyncio
async def test_sampling_decision_is_recorded(service) -> None:
    """So that "no trace for this id" is distinguishable from "the trace is
    missing" — two very different investigations."""
    await service.answer(QUESTION, "a@contoso.com", "isc-sg",
                         frozenset({"isc-all"}))
    ev = capstone.cap.by_event("request.completed")[0]
    assert isinstance(ev["trace.sampled"], bool)
    assert ev["trace.sample_reason"]


# ===========================================================================
# 4. SECURITY SIGNALS
# ===========================================================================

@pytest.mark.asyncio
async def test_trimmed_count_is_emitted_every_request(service) -> None:
    """This integer is the ONLY externally visible signal that permission
    trimming is still running. It must be on every request, always."""
    for groups in (frozenset({"isc-all"}), frozenset({"hr-only"}),
                   frozenset({"legal-only"})):
        capstone.STORE.reset()
        await service.answer(QUESTION, "a@contoso.com", "isc-sg", groups)
        retrieve = [s for s in capstone.STORE.spans if s.name == "retrieve"]
        assert retrieve
        assert "retrieval.trimmed_count" in retrieve[0].attributes


@pytest.mark.asyncio
async def test_acl_violation_emits_an_event_with_ids_only(scanner) -> None:
    """The violation event must not contain the leaked chunk's text —
    logging it would complete the leak you just detected."""
    svc = capstone.RagService(capstone.Search(trim=False), capstone.Model())
    await svc.answer("settlement compensation confidential terms",
                     "a@contoso.com", "isc-sg", frozenset({"isc-all"}))

    retrieve = next(s for s in capstone.STORE.spans if s.name == "retrieve")
    violations = [e for e in retrieve.events if e.name == "acl.violation"]
    assert violations, "an untrimmed retriever must produce violation events"

    for e in violations:
        assert "chunk_id" in e.attributes and "doc_id" in e.attributes
        findings = scanner.scan(json.dumps(e.attributes), "acl.violation")
        assert not findings, (
            f"the violation event leaked the content it detected: {findings}"
        )


@pytest.mark.asyncio
async def test_acl_violation_forces_an_audit_record() -> None:
    svc = capstone.RagService(capstone.Search(trim=False), capstone.Model())
    await svc.answer("settlement compensation confidential terms",
                     "a@contoso.com", "isc-sg", frozenset({"isc-all"}))
    assert svc.audit_records
    assert svc.audit_records[0]["reason"] == "acl_violation"


@pytest.mark.asyncio
async def test_acl_violation_is_never_sampled_away() -> None:
    """A security event sampled at 1% is a security event you do not have."""
    svc = capstone.RagService(capstone.Search(trim=False), capstone.Model())
    capstone.SAMPLER.base_rate = 0.0        # drop everything not forced
    try:
        await svc.answer("settlement compensation confidential",
                         "a@contoso.com", "isc-sg", frozenset({"isc-all"}))
        assert capstone.SAMPLER.decisions["acl_violation"] == 1
        assert capstone.SAMPLER.decisions["dropped"] == 0
    finally:
        capstone.SAMPLER.base_rate = 0.10


# ===========================================================================
# 5. CARDINALITY
# ===========================================================================

@pytest.mark.asyncio
async def test_metric_labels_are_bounded() -> None:
    """One unbounded label is a cost incident. Assert the bound directly."""
    svc = capstone.RagService(capstone.Search(), capstone.Model())
    for i in range(40):
        await svc.answer(f"question number {i}", f"user{i}@contoso.com",
                         f"tenant-{i % 3}", frozenset({"isc-all"}))

    card = capstone.STORE.cardinality()
    for metric, n in card.items():
        assert n <= 24, (
            f"metric {metric!r} has {n} label combinations after only 40 "
            f"requests — a label is unbounded"
        )


@pytest.mark.asyncio
async def test_no_high_cardinality_label_names() -> None:
    """A denylist as a second line of defence. The bound test above is the
    real check; this one names the specific mistake so the failure message
    tells you what you did."""
    svc = capstone.RagService(capstone.Search(), capstone.Model())
    await svc.answer(QUESTION, "a@contoso.com", "isc-sg",
                     frozenset({"isc-all"}))
    forbidden = {"trace_id", "span_id", "user_id", "principal",
                 "question", "question_hash", "chunk_id"}
    for (_metric, labels) in capstone.STORE.counters:
        for key, _ in labels:
            assert key not in forbidden, (
                f"{key!r} is a metric label; it belongs on a span or a log "
                f"line, not in a metric"
            )


# ===========================================================================
# 6. STATUS DISCIPLINE
# ===========================================================================

@pytest.mark.asyncio
async def test_repaired_request_is_not_an_error() -> None:
    """A repaired generation SUCCEEDED. Marking it ERROR makes your error rate
    count handled failures, and then nobody trusts it."""
    svc = capstone.RagService(capstone.Search(), capstone.Model())
    r = await svc.answer(QUESTION, "a@contoso.com", "isc-sg",
                         frozenset({"isc-all"}), inject_bad_json=True)
    gen = next(s for s in capstone.STORE.spans if s.name == "generate")
    assert gen.status is not StatusCode.ERROR, (
        "a repaired generation is not an error"
    )
    assert any(e.name == "repair.attempted" for e in gen.events), (
        "the repair must be visible as an event, or it is invisible entirely"
    )
    assert r["outcome"] == "degraded"


@pytest.mark.asyncio
async def test_real_failure_is_an_error() -> None:
    """The negative control for the test above."""
    svc = capstone.RagService(capstone.Search(), capstone.Model())
    await svc.answer(QUESTION, "a@contoso.com", "isc-sg",
                     frozenset({"isc-all"}), inject_failure=True)
    root = next(s for s in capstone.STORE.spans if s.kind is SpanKind.SERVER)
    assert root.status is StatusCode.ERROR


# ===========================================================================
# 7. THE HARNESS ITSELF
# ===========================================================================

def test_trace_ids_are_w3c_shaped() -> None:
    """A malformed traceparent is dropped SILENTLY by downstream services."""
    for _ in range(20):
        t = new_trace_id()
        assert len(t) == 32 and all(c in "0123456789abcdef" for c in t)
        s = new_span_id()
        assert len(s) == 16 and all(c in "0123456789abcdef" for c in s)


def test_context_filter_injects_on_every_record() -> None:
    log, cap = make_logger("isc.test.filter")
    trace_id_var.set(new_trace_id())
    log.info("", extra={"event": "test"})
    assert cap.records[0]["trace_id"], (
        "the filter must attach the trace id even when the caller forgot"
    )
