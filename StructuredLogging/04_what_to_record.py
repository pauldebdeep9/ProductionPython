"""
04 — What to record: prompts, chunks, and the telemetry/audit split.

THE TENSION
-----------
Debugging a bad answer needs the prompt, the retrieved chunks, and the output.
Those are exactly the three things that must not go into your general
telemetry stream.

This is not a rule someone invented to be difficult. Query access to
Application Insights is handed out for debugging; access to the SharePoint
documents behind your index is granted per-document, per-group, and audited.
Writing chunk text into telemetry copies document content into a store with a
LARGER READER SET, a DIFFERENT RETENTION PERIOD, and NO PERMISSION TRIMMING.

It also breaks the property your whole retrieval design exists to guarantee.
A permission-trimmed pipeline that logs the chunks it retrieved has an
untrimmed copy of the corpus in Log Analytics.

THE RESOLUTION is two sinks, not one:

    TELEMETRY  broad access, 30-90 day retention, high volume.
               Ids, counts, hashes, durations, fingerprints. NO content.

    AUDIT      restricted access, short retention, explicit legal basis,
               low volume, sampled or on-demand.
               Full content, for the cases that genuinely need it.

Run:  python 04_what_to_record.py
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass

from obs_lab import (
    ALLOWED_IN_AUDIT,
    ALLOWED_IN_TELEMETRY,
    CORPUS,
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
    trace_id_var,
    verdict,
)

STORE = TelemetryStore()
TRACER = Tracer(STORE)
log, cap = make_logger("isc.record")

USER_QUESTION = ("Why was invoice 88 billed for more units than the goods "
                 "receipt shows, and who approved the variance?")
SCANNER = SensitivityScanner()
SCANNER.register("user_question", USER_QUESTION)
SCANNER.register("hr_chunk", CORPUS[4].text)
SCANNER.register("legal_chunk", CORPUS[5].text)


# ---------------------------------------------------------------------------
# PART 1 — the naive instrumentation
# ---------------------------------------------------------------------------

def part1_naive() -> None:
    banner("PART 1 — the instrumentation everyone writes first")

    STORE.reset()
    SCANNER.reset()
    trace_id_var.set(new_trace_id())
    principal_var.set("debdeep@contoso.com")

    chunks = [CORPUS[0], CORPUS[2], CORPUS[3]]
    prompt = ("You are an ISC analyst.\nContext:\n"
              + "\n".join(f"[{c.chunk_id}] {c.text}" for c in chunks)
              + f"\nQuestion: {USER_QUESTION}")

    with TRACER.span("generate", SpanKind.CLIENT) as s:
        # Every one of these is a line someone wrote to make debugging easier.
        s.set("prompt", prompt)
        s.set("question", USER_QUESTION)
        s.set("chunks", [c.text for c in chunks])
        s.set("response", "Invoice 88 billed 402 units against a 400 unit receipt.")

    findings = SCANNER.scan_obj(STORE.spans[0].to_dict(), "span")
    verdict(not findings, f"{len(findings)} sensitive item(s) in the span")
    for _where, name, excerpt in findings:
        print(f"        {name}: {excerpt}...")

    print("""
    Four attributes, and the span now contains the user's question and three
    documents' worth of content. Multiply by your request rate and 90 days of
    retention.

    AND THE PART THAT IS EASY TO MISS: `chunks` here happened to be
    ISC-visible. The instrumentation does not know that. The same line, on a
    request from someone with `hr-only`, writes compensation data into
    telemetry — where the ISC team can read it.""")


# ---------------------------------------------------------------------------
# PART 2 — the sensitivity-tagged field
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Field_:
    """A value that knows where it may be emitted.

    Same argument as SecretStr in the config tutorial: the decision belongs at
    DECLARATION, not at every emission site. A new field is protected the
    moment it is declared, rather than the moment someone remembers.
    """

    name: str
    value: object
    sensitivity: Sensitivity

    def for_sink(self, allowed: set[Sensitivity]) -> object | None:
        return self.value if self.sensitivity in allowed else None


class Recorder:
    """Routes fields to the sinks their sensitivity permits.

    ONE call site produces BOTH the telemetry attributes and the audit record,
    from one declaration. That is what stops the two drifting — the common
    failure being that someone adds a debugging attribute to the span and
    forgets that the span goes somewhere the audit record does not.
    """

    def __init__(self) -> None:
        self.fields: list[Field_] = []

    def add(self, name: str, value: object,
            sensitivity: Sensitivity = Sensitivity.PUBLIC) -> Recorder:
        self.fields.append(Field_(name, value, sensitivity))
        return self

    def telemetry(self) -> dict[str, object]:
        return {f.name: f.value for f in self.fields
                if f.sensitivity in ALLOWED_IN_TELEMETRY}

    def audit(self) -> dict[str, object]:
        return {f.name: f.value for f in self.fields
                if f.sensitivity in ALLOWED_IN_AUDIT}

    def dropped(self) -> list[str]:
        return [f.name for f in self.fields
                if f.sensitivity not in ALLOWED_IN_TELEMETRY]


def part2_tagged_fields() -> None:
    banner("PART 2 — tag the field, not the call site")

    chunks = [CORPUS[0], CORPUS[2], CORPUS[3]]
    prompt = "…" + USER_QUESTION

    r = (Recorder()
         .add("model.deployment", "gpt-4o-mini-prod", Sensitivity.PUBLIC)
         .add("gen.prompt_tokens", 1840, Sensitivity.PUBLIC)
         .add("gen.finish_reason", "stop", Sensitivity.PUBLIC)
         .add("retrieval.chunk_ids", [c.chunk_id for c in chunks],
              Sensitivity.INTERNAL)
         .add("retrieval.doc_ids", [c.doc_id for c in chunks],
              Sensitivity.INTERNAL)
         .add("acl.group_count", 2, Sensitivity.INTERNAL)
         # The three that must never reach telemetry.
         .add("prompt.text", prompt, Sensitivity.SENSITIVE)
         .add("question.text", USER_QUESTION, Sensitivity.SENSITIVE)
         .add("chunks.text", [c.text for c in chunks], Sensitivity.SENSITIVE)
         .add("auth.token", "eyJ-FAKE-TOKEN-abc", Sensitivity.SECRET))

    section("telemetry sink")
    print_json(r.telemetry())
    section("dropped before telemetry")
    show("fields", r.dropped())
    section("audit sink (restricted access)")
    show("field names", sorted(r.audit()))
    show("includes the token?", "auth.token" in r.audit())

    SCANNER.reset()
    findings = SCANNER.scan_obj(r.telemetry(), "telemetry")
    verdict(not findings, f"telemetry sink: {len(findings)} sensitive item(s)")
    findings2 = SCANNER.scan_obj(r.audit(), "audit")
    verdict(bool(findings2),
            f"audit sink: {len(findings2)} sensitive item(s) — EXPECTED, "
            f"that is its purpose")
    SCANNER.reset()

    print("""
    Note the SECRET tier reaches NEITHER sink. A bearer token has no debugging
    value that justifies persisting it anywhere.""")


# ---------------------------------------------------------------------------
# PART 3 — what to record instead of content
# ---------------------------------------------------------------------------

def question_hash(q: str) -> str:
    """Group identical questions without storing any of them.

    A SALT is required. Without one, an attacker with the telemetry and a
    dictionary of plausible questions can confirm which were asked — the hash
    space for "questions people ask a supply chain bot" is small enough to
    enumerate. Salt per deployment, keep it with your secrets.
    """
    salt = "isc-telemetry-salt-v1"
    return hashlib.sha256((salt + q).encode()).hexdigest()[:12]


def part3_substitutes() -> None:
    banner("PART 3 — the substitutes, and what each buys")

    substitutes = [
        ("question.hash", question_hash(USER_QUESTION),
         "group repeats; find the top 20 questions; detect a spike"),
        ("question.length", len(USER_QUESTION),
         "detect truncation, abuse, and prompt-injection attempts by size"),
        ("question.lang", "en", "routing and quality analysis"),
        ("retrieval.chunk_ids", ["c1", "c3", "c4"],
         "WHICH documents — reconstruct under your own ACL when needed"),
        ("retrieval.doc_ids", ["po-1001", "inv-88", "gr-55"],
         "which sources; detect a document dominating answers"),
        ("retrieval.score_min/max", "0.41 / 0.88",
         "detect a retrieval-quality regression"),
        ("retrieval.trimmed_count", 2,
         "how many chunks the ACL removed — a SECURITY signal"),
        ("prompt.fingerprint", "36206a029343",
         "which template version produced this"),
        ("prompt.token_count", 1840, "size without content"),
        ("response.finish_reason", "stop", "how generation ended"),
        ("response.citation_count", 3, "did it cite anything at all?"),
        ("response.grounded", True,
         "did every claim map to a retrieved chunk?"),
    ]
    print(f"      {'field':<30} {'example':<26} why")
    print(f"      {'-' * 30} {'-' * 26} {'-' * 30}")
    for name, example, why in substitutes:
        print(f"      {name:<30} {str(example)[:26]:<26} {why[:42]}")

    print("""
    THE KEY INSIGHT: `retrieval.chunk_ids` plus `prompt.fingerprint` lets you
    RECONSTRUCT the exact prompt — by fetching those chunks from your index,
    under your own permission checks, at the moment you need them. You get
    full debuggability without a persistent untrimmed copy.

    The cost is that reconstruction is only faithful while those chunks still
    exist and are unchanged. For a system where answers must be auditable
    years later, that is not enough, and you need the audit sink in PART 5.""")

    section("hashing groups without storing")
    variants = [USER_QUESTION, USER_QUESTION, "What is the PO quantity?"]
    for q in variants:
        show(question_hash(q), f"len={len(q)}")
    print("      Two identical questions share a hash; the third does not.")
    print("      You can count, rank, and spot spikes — and cannot read any.")


# ---------------------------------------------------------------------------
# PART 4 — the ACL signals that belong in telemetry
# ---------------------------------------------------------------------------

def part4_acl_signals() -> None:
    banner("PART 4 — telemetry is how a silent permission failure becomes loud")

    print("""
    Permission-trimmed retrieval fails SILENTLY and UPWARD: the user gets a
    fluent, well-cited answer built from a document they were never allowed to
    see, and nothing in the response indicates a problem.

    There is no error to alert on. So telemetry is the ONLY place this becomes
    visible, which makes these fields load-bearing rather than nice-to-have:

      acl.group_count            resolved groups. A sudden 0 means the ACL
                                 resolver is failing open somewhere.
      retrieval.candidate_count  before trimming
      retrieval.chunk_count      after trimming
      retrieval.trimmed_count    the difference. A sudden drop to 0 across
                                 all requests means trimming stopped running.
      acl.violation              a chunk survived the post-retrieval assertion
                                 that should not have. Should ALWAYS be zero.

    THE ALERT WORTH HAVING: `trimmed_count == 0` sustained across many
    requests, when it is normally non-zero. That is what "someone deployed a
    retriever that ignores groups" looks like from the outside — and it looks
    like nothing else.""")

    section("normal operation")
    STORE.reset()
    for principal, groups, _expected in [
        ("debdeep@contoso.com", frozenset({"isc-all"}), 4),
        ("hr.lead@contoso.com", frozenset({"isc-all", "hr-only"}), 5),
        ("legal@contoso.com", frozenset({"legal-only"}), 1),
    ]:
        trace_id_var.set(new_trace_id())
        principal_var.set(principal)
        visible = [c for c in CORPUS if c.visible_to(groups)]
        with TRACER.span("retrieve", SpanKind.CLIENT) as s:
            s.set("acl.group_count", len(groups))
            s.set("retrieval.candidate_count", len(CORPUS))
            s.set("retrieval.chunk_count", len(visible))
            s.set("retrieval.trimmed_count", len(CORPUS) - len(visible))
            s.set("retrieval.chunk_ids", [c.chunk_id for c in visible])
        STORE.incr("retrieval.trimmed", len(CORPUS) - len(visible))
        show(f"{principal:<22}", f"visible={len(visible)} "
                                 f"trimmed={len(CORPUS) - len(visible)}")

    section("a retriever that stopped trimming")
    trace_id_var.set(new_trace_id())
    principal_var.set("debdeep@contoso.com")
    with TRACER.span("retrieve", SpanKind.CLIENT) as s:
        s.set("acl.group_count", 1)
        s.set("retrieval.candidate_count", len(CORPUS))
        s.set("retrieval.chunk_count", len(CORPUS))    # nothing removed
        s.set("retrieval.trimmed_count", 0)
        s.set("retrieval.chunk_ids", [c.chunk_id for c in CORPUS])

    zero_trim = list(STORE.where(**{"retrieval.trimmed_count": 0}))
    show("spans with trimmed_count == 0", len(zero_trim))
    print("""
      The response to that request looked completely normal. The only
      indication anywhere in the system that a leak occurred is this one
      integer — which is why it has to be emitted on every request, and why
      an alert on it is worth more than most of your dashboard.""")

    section("the assertion, as a span event")
    STORE.reset()
    trace_id_var.set(new_trace_id())
    groups = frozenset({"isc-all"})
    returned = [CORPUS[0], CORPUS[5]]     # c6 is legal-only — a violation
    with TRACER.span("retrieve") as s:
        violations = [c for c in returned if not c.visible_to(groups)]
        for c in violations:
            # An EVENT, with ids only. Never the text of the leaked chunk —
            # logging it would complete the leak you just detected.
            s.event("acl.violation", **{
                "chunk_id": c.chunk_id, "doc_id": c.doc_id,
                "chunk.sensitivity": c.sensitivity,
                "caller.group_count": len(groups),
            })
            STORE.incr("acl.violation", doc_id=c.doc_id)
    print_json(STORE.spans[0].to_dict()["events"])
    SCANNER.reset()
    findings = SCANNER.scan_obj(STORE.spans[0].to_dict(), "violation span")
    verdict(not findings,
            "the violation event contains ids, not the leaked content")


# ---------------------------------------------------------------------------
# PART 5 — the audit sink
# ---------------------------------------------------------------------------

@dataclass
class AuditRecord:
    """A full-content record, going to a restricted sink.

    THE PROPERTIES that make this acceptable where telemetry is not:
      * ACCESS: a small named group, not everyone with a debugging need
      * RETENTION: short and enforced, e.g. 7-30 days
      * VOLUME: sampled, or triggered, not every request
      * BASIS: an explicit reason it exists, written down
      * IMMUTABLE and itself audited — reads are logged
    """

    trace_id: str
    ts: float
    principal: str
    tenant: str
    reason: str                # WHY this record exists
    question: str
    chunk_ids: list[str]
    chunk_texts: list[str]
    prompt: str
    response: str
    retention_days: int = 7


class AuditSink:
    def __init__(self) -> None:
        self.records: list[AuditRecord] = []
        self.reads: list[tuple[str, str]] = []

    def write(self, rec: AuditRecord) -> None:
        self.records.append(rec)

    def read(self, trace_id: str, by: str) -> AuditRecord | None:
        # READS ARE LOGGED. An audit store nobody can read without a record
        # of having done so is what makes the restricted access real rather
        # than nominal.
        self.reads.append((trace_id, by))
        return next((r for r in self.records if r.trace_id == trace_id), None)


def should_audit(*, thumbs_down: bool, acl_violation: bool,
                 degraded: bool, sample_rate: float, roll: float) -> str | None:
    """WHEN to write a full-content record.

    Not every request. The triggers below are the ones with a genuine
    debugging or compliance need; everything else gets telemetry only.
    """
    if acl_violation:
        return "acl_violation"        # always — this is a security incident
    if thumbs_down:
        return "user_reported"        # the user asked us to look
    if degraded:
        return "degraded_answer"
    if roll < sample_rate:
        return "random_sample"        # a baseline for quality review
    return None


def part5_audit() -> None:
    banner("PART 5 — the audit sink: full content, restricted, triggered")

    sink = AuditSink()
    scenarios = [
        ("normal request", {"thumbs_down": False, "acl_violation": False,
                            "degraded": False, "roll": 0.9}),
        ("user thumbs-down", {"thumbs_down": True, "acl_violation": False,
                              "degraded": False, "roll": 0.9}),
        ("degraded answer", {"thumbs_down": False, "acl_violation": False,
                             "degraded": True, "roll": 0.9}),
        ("ACL violation", {"thumbs_down": False, "acl_violation": True,
                           "degraded": False, "roll": 0.9}),
        ("random sample", {"thumbs_down": False, "acl_violation": False,
                           "degraded": False, "roll": 0.005}),
    ]
    for label, kw in scenarios:
        reason = should_audit(sample_rate=0.01, **kw)  # type: ignore[arg-type]
        show(label, reason or "telemetry only")
        if reason:
            sink.write(AuditRecord(
                trace_id=new_trace_id(), ts=time.time(),
                principal="debdeep@contoso.com", tenant="isc-sg",
                reason=reason, question=USER_QUESTION,
                chunk_ids=["c1", "c3"], chunk_texts=[CORPUS[0].text, CORPUS[2].text],
                prompt="…", response="…"))

    show("audit records written", len(sink.records))
    show("of 5 requests", f"{len(sink.records)}/5")

    section("reading it is itself audited")
    rec = sink.read(sink.records[0].trace_id, by="debdeep@contoso.com")
    show("read returned", rec.reason if rec else None)
    show("read log", sink.reads)

    print("""
    THE SAMPLING RATE IS THE WHOLE COST CONTROL. At 1% you get a quality
    baseline for a hundredth of the storage and a hundredth of the exposure.
    The triggered cases — thumbs-down, degraded, ACL violation — are exactly
    the ones where you would otherwise wish you had the content, and they are
    rare.

    WHAT TO WRITE DOWN BEFORE BUILDING THIS, because it is a data-protection
    decision and not an engineering one:
      * who may read it, by name or by group
      * how long it is kept, enforced by the store rather than by intention
      * whether user content may be used for evaluation or fine-tuning
      * what happens on a deletion request
      * whether the region it is stored in satisfies residency requirements

    In an export-controlled or regulated environment, get that agreed before
    the first record is written. Retrofitting a retention policy onto a store
    that already has two years of prompts in it is not a technical problem.""")


# ---------------------------------------------------------------------------
# PART 6 — the checklist
# ---------------------------------------------------------------------------

def part6_checklist() -> None:
    banner("PART 6 — the emission checklist")

    print("""
    FOR EVERY ATTRIBUTE, LOG FIELD, OR SPAN EVENT, ask:

      1. Could this contain USER-AUTHORED text?
         question, prompt, chat history, an uploaded filename
         -> SENSITIVE. Hash, measure, or drop.

      2. Could this contain DOCUMENT content?
         chunk text, a summary, a snippet, a highlighted excerpt
         -> SENSITIVE. Ids only.
         NOTE: search "highlights" are document content wearing a different
         name, and they arrive pre-formatted for display, which makes them
         very easy to log by accident.

      3. Could this contain MODEL OUTPUT?
         -> SENSITIVE. It is derived from the two above.

      4. Is it a CREDENTIAL?
         -> SECRET. Never, anywhere.

      5. Is it a personal IDENTIFIER?
         email, UPN, employee id
         -> Hash it, or use an opaque id. `enduser.id` in OTel conventions
         should be a stable pseudonym, not an email address.

      6. Is it UNBOUNDED?
         a list that grows with input size, a raw error body
         -> Cap the length and record the true size separately. A single
         span attribute holding a 200KB error body is a real incident.

    IF IT SURVIVES ALL SIX, it is a count, an id, a duration, an enum, or a
    fingerprint — which is exactly the set of things telemetry should carry.""")

    section("running the checklist over a real attribute set")
    candidates = [
        ("model.deployment", "gpt-4o-mini-prod", "OK"),
        ("gen.prompt_tokens", 1840, "OK"),
        ("retrieval.chunk_ids", ["c1", "c3"], "OK"),
        ("enduser.id", "debdeep@contoso.com", "HASH IT — personal identifier"),
        ("search.highlights", ["...402 units..."], "DROP — document content"),
        ("error.body", "<8KB of JSON>", "CAP — unbounded"),
        ("question", USER_QUESTION, "DROP — user-authored"),
        ("prompt", "…", "DROP — contains both of the above"),
    ]
    for name, _value, verdict_text in candidates:
        marker = "ok " if verdict_text == "OK" else "-> "
        print(f"      {marker} {name:<24} {verdict_text}")


def main() -> None:
    part1_naive()
    part2_tagged_fields()
    part3_substitutes()
    part4_acl_signals()
    part5_audit()
    part6_checklist()

    banner("SUMMARY")
    print("""
  * Telemetry and audit are DIFFERENT SINKS with different reader sets,
    retention, and volume. One decision, two destinations.
  * Tag sensitivity at DECLARATION, not at the emission site — the same
    argument as SecretStr for secrets.
  * Never emit prompt text, chunk text, the user's question, or model output
    to telemetry. Emit salted hashes, ids, counts, and fingerprints.
  * chunk_ids + prompt.fingerprint lets you RECONSTRUCT the prompt under your
    own permission checks, which is better than storing it.
  * `retrieval.trimmed_count` is a SECURITY signal. A silent permission
    failure is invisible everywhere else, so alert on it dropping to zero.
  * An `acl.violation` event carries ids only — logging the leaked chunk
    completes the leak you just detected.
  * Audit is TRIGGERED (thumbs-down, degraded, violation) plus a small random
    sample. Reads are logged. Retention is enforced, not intended.
  * Search `highlights` are document content under another name.
""")


if __name__ == "__main__":
    main()
