"""
obs_lab.py — shared harness for the observability tutorial.

WHAT THIS PROVIDES
------------------
  * A JSON logger that captures records in memory, so tests can assert on
    FIELDS rather than on string matching.
  * A minimal span implementation with the OpenTelemetry data model — trace
    id, span id, parent, kind, attributes, events, status. Enough to be
    faithful without needing the SDK or a collector.
  * A QUERYABLE store. This is the piece that matters: telemetry you cannot
    query is decoration. Script 07 answers real incident questions against it.
  * A sensitivity scanner, so "we don't log prompt content" becomes a test.
  * A cardinality tracker, because the metric that kills your bill is usually
    one label nobody thought about.

WHY NOT THE REAL OTEL SDK
-------------------------
The concepts transfer exactly and the SDK adds an exporter, a collector, and a
backend to the setup. Everything here maps one-to-one onto
`opentelemetry-sdk`: `Span` -> `trace.Span`, `attributes` -> `set_attribute`,
`events` -> `add_event`, `Status` -> `trace.Status`. Where the real API differs
in a way that matters, the scripts say so.
"""

from __future__ import annotations

import contextvars
import hashlib
import json
import logging
import re
import time
import uuid
from collections import Counter, defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar

# ===========================================================================
# 1. CONTEXT
# ===========================================================================
# These are the values that must follow a request across every await, task,
# and thread boundary. contextvars is the mechanism; script 02 covers the
# edges where it does and does not propagate.

trace_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "trace_id", default="")
span_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "span_id", default="")
principal_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "principal", default="")
tenant_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "tenant", default="")


def new_trace_id() -> str:
    """128-bit, hex, no dashes — the W3C trace-context format.

    NOT a uuid4 string with dashes. The `traceparent` header requires exactly
    32 lowercase hex characters, and a backend that parses it will silently
    drop a malformed one, which presents as "our traces are not linking up"
    with no error anywhere.
    """
    return uuid.uuid4().hex


def new_span_id() -> str:
    """64-bit, hex. Half the width of a trace id."""
    return uuid.uuid4().hex[:16]


# ===========================================================================
# 2. SENSITIVITY
# ===========================================================================

class Sensitivity(Enum):
    """What may be emitted where.

    THE CENTRAL IDEA of script 04: telemetry and audit are DIFFERENT SINKS
    with different reader sets and different retention. A field's sensitivity
    determines which sinks may receive it, and that decision belongs at the
    point of declaration rather than at each emission site.
    """

    PUBLIC = "public"          # ids, counts, durations, model names
    INTERNAL = "internal"      # query shape, chunk ids, group names
    SENSITIVE = "sensitive"    # prompt text, chunk text, user questions
    SECRET = "secret"          # tokens, keys — never emitted anywhere


ALLOWED_IN_TELEMETRY = {Sensitivity.PUBLIC, Sensitivity.INTERNAL}
ALLOWED_IN_AUDIT = {Sensitivity.PUBLIC, Sensitivity.INTERNAL,
                    Sensitivity.SENSITIVE}


class SensitivityScanner:
    """Detects content that should never have reached a telemetry sink.

    Two detection strategies, because neither alone is sufficient:
      1. Registered VERBATIM strings (a specific prompt, a specific chunk).
         Precise, but only finds what you told it about.
      2. Shape-based patterns (emails, long free text). Catches things you
         did not register, at the cost of false positives.

    Both are used in the tests. Verbatim registration is what makes the
    assertions deterministic; the patterns are what catch the case someone
    added last week.
    """

    PATTERNS: ClassVar[dict[str, re.Pattern[str]]] = {
        "email": re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),
        "bearer_token": re.compile(r"eyJ[A-Za-z0-9_\-.]{10,}"),
        "long_prose": re.compile(r"[A-Za-z ,.']{160,}"),
    }

    def __init__(self) -> None:
        self.registered: dict[str, str] = {}
        self.findings: list[tuple[str, str, str]] = []

    def register(self, name: str, value: str) -> None:
        if len(value) >= 12:
            self.registered[name] = value

    def scan(self, text: str, where: str) -> list[tuple[str, str, str]]:
        found: list[tuple[str, str, str]] = []
        for name, value in self.registered.items():
            if value in text:
                found.append((where, name, value[:40]))
        for name, pattern in self.PATTERNS.items():
            m = pattern.search(text)
            if m:
                found.append((where, f"pattern:{name}", m.group()[:40]))
        self.findings.extend(found)
        return found

    def scan_obj(self, obj: Any, where: str) -> list[tuple[str, str, str]]:
        return self.scan(json.dumps(obj, default=str), where)

    def report(self) -> str:
        if not self.findings:
            return "no sensitive content found in telemetry"
        lines = [f"{len(self.findings)} FINDING(S):"]
        lines += [f"    {w}: {n} -> {v!r}" for w, n, v in self.findings]
        return "\n".join(lines)

    def reset(self) -> None:
        self.findings.clear()


# ===========================================================================
# 3. STRUCTURED LOGGING
# ===========================================================================

class ContextFilter(logging.Filter):
    """Injects the request context into every record automatically.

    THIS IS THE POINT of using a filter rather than passing trace_id to every
    log call: you cannot forget it. A log line written by a library, by a
    framework, or by someone who did not know about your conventions still
    carries the trace id.
    """

    # A per-process salt. In production this belongs with your secrets, and
    # must be STABLE across replicas or the same user hashes differently on
    # each pod and grouping breaks.
    SALT = "isc-telemetry-salt-v1"

    def filter(self, record: logging.LogRecord) -> bool:
        record.trace_id = trace_id_var.get()          # type: ignore[attr-defined]
        record.span_id = span_id_var.get()            # type: ignore[attr-defined]
        record.tenant = tenant_var.get()              # type: ignore[attr-defined]

        # THE BUG THIS FIXES, which I shipped in the first version of this
        # file: the filter emitted `principal` verbatim, so every log line
        # carried a user's email address — violating the "hash personal
        # identifiers" rule from script 04, in the very harness that teaches
        # it. The capstone's own scanner caught it via the email pattern.
        #
        # It is a good illustration of why the automatic-injection filter is
        # both the right design AND a place to be careful: it applies to every
        # log line in the process, so one careless field there is a leak
        # everywhere at once.
        principal = principal_var.get()
        record.enduser_id = (                          # type: ignore[attr-defined]
            hashlib.sha256((self.SALT + principal).encode()).hexdigest()[:12]
            if principal else ""
        )
        return True


RESERVED = {
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
    "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated", "thread", "threadName",
    "processName", "process", "taskName", "message", "asctime",
}


class JsonFormatter(logging.Formatter):
    """Emits one JSON object per line.

    FIELD DISCIPLINE, which script 01 argues for:
      * `event` is a STABLE, LOW-CARDINALITY name you can group by.
      * `message` is optional human prose. Never parse it.
      * everything else is a typed field, not interpolated into a string.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "event": getattr(record, "event", record.getMessage()),
            "trace_id": getattr(record, "trace_id", ""),
            "span_id": getattr(record, "span_id", ""),
        }
        for key, value in record.__dict__.items():
            if key not in RESERVED and key not in payload and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else "",
                "message": str(record.exc_info[1]),
            }
        return json.dumps(payload, default=str)


class CapturingHandler(logging.Handler):
    """Keeps records in memory as dicts, so tests assert on fields."""

    def __init__(self) -> None:
        super().__init__()
        self.records: list[dict[str, Any]] = []
        self.setFormatter(JsonFormatter())

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(json.loads(self.format(record)))

    def by_event(self, event: str) -> list[dict[str, Any]]:
        return [r for r in self.records if r.get("event") == event]

    def reset(self) -> None:
        self.records.clear()


def make_logger(name: str = "isc", level: int = logging.INFO
                ) -> tuple[logging.Logger, CapturingHandler]:
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(level)
    logger.propagate = False
    handler = CapturingHandler()
    handler.addFilter(ContextFilter())
    logger.addHandler(handler)
    return logger, handler


# ===========================================================================
# 4. SPANS
# ===========================================================================

class SpanKind(Enum):
    """Mirrors OTel. The kind changes how a backend renders and aggregates."""

    INTERNAL = "internal"
    SERVER = "server"      # you received a request
    CLIENT = "client"      # you called something
    PRODUCER = "producer"  # you enqueued
    CONSUMER = "consumer"  # you dequeued


class StatusCode(Enum):
    UNSET = "unset"
    OK = "ok"
    ERROR = "error"


@dataclass
class SpanEvent:
    """A timestamped point WITHIN a span.

    Use an event, not a child span, when the thing has no meaningful duration:
    a cache hit, a retry attempt, a validation failure, a token arriving.
    Creating a child span for every one of those is how a trace becomes
    unreadable and expensive.
    """

    name: str
    ts: float
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class Span:
    name: str
    trace_id: str
    span_id: str
    parent_id: str | None
    kind: SpanKind = SpanKind.INTERNAL
    start: float = field(default_factory=time.perf_counter)
    end: float | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[SpanEvent] = field(default_factory=list)
    status: StatusCode = StatusCode.UNSET
    status_message: str = ""

    @property
    def duration_ms(self) -> float:
        return ((self.end or time.perf_counter()) - self.start) * 1000

    def set(self, key: str, value: Any) -> Span:
        self.attributes[key] = value
        return self

    def event(self, name: str, **attrs: Any) -> Span:
        self.events.append(SpanEvent(name, time.perf_counter(), attrs))
        return self

    def record_exception(self, exc: BaseException) -> Span:
        """Record the exception TYPE and message, never the object's repr.

        A repr can pull in locals under some formatters, and in a config or
        request object that means secrets and prompt text. Type plus message
        is what you actually need to triage.
        """
        self.event("exception", **{
            "exception.type": type(exc).__name__,
            "exception.message": str(exc)[:200],
        })
        self.status = StatusCode.ERROR
        self.status_message = f"{type(exc).__name__}: {str(exc)[:120]}"
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "trace_id": self.trace_id,
            "span_id": self.span_id, "parent_id": self.parent_id,
            "kind": self.kind.value, "duration_ms": round(self.duration_ms, 2),
            "status": self.status.value, "status_message": self.status_message,
            "attributes": self.attributes,
            "events": [{"name": e.name, "attributes": e.attributes}
                       for e in self.events],
        }


class Tracer:
    """Creates spans, maintains parent-child relationships via contextvars."""

    def __init__(self, store: TelemetryStore) -> None:
        self.store = store

    @contextmanager
    def span(self, name: str, kind: SpanKind = SpanKind.INTERNAL,
             **attributes: Any) -> Iterator[Span]:
        trace_id = trace_id_var.get() or new_trace_id()
        parent = span_id_var.get() or None
        span = Span(name=name, trace_id=trace_id, span_id=new_span_id(),
                    parent_id=parent, kind=kind, attributes=dict(attributes))

        t_token = trace_id_var.set(trace_id)
        s_token = span_id_var.set(span.span_id)
        try:
            yield span
            if span.status is StatusCode.UNSET:
                span.status = StatusCode.OK
        except BaseException as exc:
            span.record_exception(exc)
            raise
        finally:
            span.end = time.perf_counter()
            # ALWAYS export, including on the exception path. A span that is
            # only exported on success gives you a trace with a hole exactly
            # where the failure was.
            self.store.add_span(span)
            span_id_var.reset(s_token)
            trace_id_var.reset(t_token)


# ===========================================================================
# 5. THE QUERYABLE STORE
# ===========================================================================

class TelemetryStore:
    """Spans, logs, and metrics in one place, with a query surface.

    The methods below are deliberately shaped like the questions you ask
    during an incident, not like a generic database. Script 07 uses them to
    answer real questions — which is the only test of whether your telemetry
    was worth emitting.
    """

    def __init__(self) -> None:
        self.spans: list[Span] = []
        self.counters: Counter[tuple[str, tuple[tuple[str, str], ...]]] = Counter()
        self.histograms: dict[str, list[float]] = defaultdict(list)
        self.dropped_by_sampling = 0

    # -- ingestion ---------------------------------------------------------

    def add_span(self, span: Span) -> None:
        self.spans.append(span)

    def incr(self, name: str, value: int = 1, **labels: str) -> None:
        self.counters[(name, tuple(sorted(labels.items())))] += value

    def observe(self, name: str, value: float, **labels: str) -> None:
        key = name + ("|" + ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
                      if labels else "")
        self.histograms[key].append(value)

    # -- queries -----------------------------------------------------------

    def trace(self, trace_id: str) -> list[Span]:
        return sorted((s for s in self.spans if s.trace_id == trace_id),
                      key=lambda s: s.start)

    def traces(self) -> list[str]:
        seen: list[str] = []
        for s in self.spans:
            if s.trace_id not in seen:
                seen.append(s.trace_id)
        return seen

    def root_spans(self) -> list[Span]:
        return [s for s in self.spans if s.parent_id is None]

    def where(self, **conditions: Any) -> list[Span]:
        """Filter spans by attribute equality. `where(**{"model.name": "x"})`."""
        def matches(s: Span) -> bool:
            for k, v in conditions.items():
                if k in ("name", "status", "kind"):
                    actual = getattr(s, k)
                    actual = actual.value if isinstance(actual, Enum) else actual
                    if actual != v:
                        return False
                elif s.attributes.get(k) != v:
                    return False
            return True

        return [s for s in self.spans if matches(s)]

    def errors(self) -> list[Span]:
        return [s for s in self.spans if s.status is StatusCode.ERROR]

    def percentile(self, name: str, p: float) -> float:
        vals = sorted(s.duration_ms for s in self.spans if s.name == name)
        if not vals:
            return 0.0
        return vals[min(len(vals) - 1, int(p * len(vals)))]

    def hist_percentile(self, key: str, p: float) -> float:
        vals = sorted(self.histograms.get(key, []))
        if not vals:
            return 0.0
        return vals[min(len(vals) - 1, int(p * len(vals)))]

    def render_trace(self, trace_id: str, indent: str = "      ") -> str:
        """A waterfall view — the thing you actually stare at."""
        spans = self.trace(trace_id)
        if not spans:
            return f"{indent}(no spans for {trace_id})"
        by_parent: dict[str | None, list[Span]] = defaultdict(list)
        for s in spans:
            by_parent[s.parent_id].append(s)
        t0 = min(s.start for s in spans)
        lines: list[str] = []

        def walk(parent: str | None, depth: int) -> None:
            for s in by_parent.get(parent, []):
                offset = (s.start - t0) * 1000
                mark = {"ok": " ", "error": "!", "unset": "?"}[s.status.value]
                lines.append(
                    f"{indent}{mark} {'  ' * depth}{s.name:<28} "
                    f"+{offset:6.1f}ms  {s.duration_ms:6.1f}ms"
                )
                for e in s.events:
                    lines.append(f"{indent}    {'  ' * depth}· {e.name}")
                walk(s.span_id, depth + 1)

        walk(None, 0)
        return "\n".join(lines)

    def cardinality(self) -> dict[str, int]:
        """Distinct label-combinations per metric name.

        The number that predicts your bill. See script 05.
        """
        out: dict[str, set[tuple[tuple[str, str], ...]]] = defaultdict(set)
        for (name, labels) in self.counters:
            out[name].add(labels)
        return {k: len(v) for k, v in out.items()}

    def reset(self) -> None:
        self.spans.clear()
        self.counters.clear()
        self.histograms.clear()
        self.dropped_by_sampling = 0


# ===========================================================================
# 6. FAKE DOMAIN
# ===========================================================================

@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    allowed_groups: frozenset[str]
    sensitivity: str = "internal"
    score: float = 0.0

    def visible_to(self, groups: frozenset[str]) -> bool:
        return bool(self.allowed_groups & groups)


CORPUS: list[Chunk] = [
    Chunk("c1", "po-1001", "PO 1001 quantity 400 units", frozenset({"isc-all"})),
    Chunk("c2", "po-1001", "PO 1001 unit price 12.50 USD", frozenset({"isc-all"})),
    Chunk("c3", "inv-88", "Invoice 88 billed 402 units", frozenset({"isc-all"})),
    Chunk("c4", "gr-55", "Goods receipt 55 recorded 400 units", frozenset({"isc-all"})),
    Chunk("c5", "hr-comp", "Compensation band for plant leads is 95000 to 130000",
          frozenset({"hr-only"}), "confidential"),
    Chunk("c6", "legal-1", "Settlement terms with supplier remain confidential",
          frozenset({"legal-only"}), "restricted"),
]


# ===========================================================================
# 7. OUTPUT HELPERS
# ===========================================================================

def banner(title: str) -> None:
    print(f"\n{'=' * 74}\n{title}\n{'=' * 74}")


def section(title: str) -> None:
    print(f"\n  --- {title} ---")


def show(label: str, value: Any) -> None:
    print(f"    {label:<46} {value}")


def verdict(ok: bool, text: str) -> None:
    print(f"    {'PASS' if ok else '*** FAIL ***':<14} {text}")


def print_json(obj: Any, limit: int = 0) -> None:
    text = json.dumps(obj, indent=2, default=str)
    if limit:
        text = text[:limit]
    for line in text.splitlines():
        print(f"      {line}")
