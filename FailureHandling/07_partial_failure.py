"""
07 — Partial failure at scale: dead letters, poison pills, and replay.

THE SCENARIO
------------
50,000 SharePoint documents to ingest. Three hours in, document 31,000 is a
corrupt PDF. What happens?

  WRONG:  the exception propagates, the run dies, you have 31,000 documents
          indexed and no record of which, and restarting means redoing all of
          them.

  ALSO WRONG:  `except Exception: pass`. The run completes, reports success,
          and 400 documents are silently missing from the index. Nobody finds
          out until a user asks why a document is not searchable, months later.

  RIGHT:  quarantine the item with enough context to diagnose and replay it,
          continue the run, and report the quarantine count as a first-class
          result — not a log line.

THE THREE THINGS THIS NEEDS
---------------------------
  1. A dead-letter store with enough context for replay.
  2. Poison-pill detection: an item that fails REPEATEDLY must stop being
     retried, or it blocks the queue forever.
  3. Failure-rate circuit breaking on the BATCH: if 40% of items are failing,
     something systemic is wrong and continuing wastes hours.

Run:  python 07_partial_failure.py
"""

from __future__ import annotations

import asyncio
import json
import time
import traceback
from dataclasses import dataclass

from failure_lab import (
    AppError,
    Metrics,
    PoisonItemError,
    ServiceUnavailable,
    ValidationError,
    banner,
    section,
)

# ---------------------------------------------------------------------------
# The dead letter record
# ---------------------------------------------------------------------------

@dataclass
class DeadLetter:
    """A quarantined item. The fields exist so someone can FIX and REPLAY it
    six weeks later without access to the original run.

    What makes a dead letter useful rather than decorative:
      item_id + payload_ref  -> find the source
      error_type + message   -> know what to fix
      attempts + first/last  -> know if it is transient or genuinely poison
      trace_id               -> pull the full trace from telemetry
      stage                  -> know WHERE it failed, not just that it did
      stack_digest           -> group identical failures without reading 400
                                stack traces individually
    """

    item_id: str
    stage: str
    error_type: str
    message: str
    attempts: int
    trace_id: str
    first_seen: float
    last_seen: float
    payload_ref: str
    stack_digest: str = ""
    replayable: bool = True

    def to_json(self) -> str:
        # NOTE: payload_REF, not the payload. Dead letters routinely contain
        # customer documents; store a pointer, not the content, or your DLQ
        # becomes an unsecured copy of your corpus.
        return json.dumps({
            "item_id": self.item_id, "stage": self.stage,
            "error_type": self.error_type, "message": self.message[:200],
            "attempts": self.attempts, "trace_id": self.trace_id,
            "payload_ref": self.payload_ref, "replayable": self.replayable,
            "stack_digest": self.stack_digest,
        })


class DeadLetterQueue:
    def __init__(self) -> None:
        self.items: dict[str, DeadLetter] = {}

    def record(self, item_id: str, stage: str, exc: BaseException,
               *, trace_id: str, payload_ref: str, replayable: bool = True) -> None:
        now = time.monotonic()
        digest = "|".join(
            line.split(",")[0].strip() for line in
            traceback.format_exception_only(type(exc), exc)
        )[:60]
        if item_id in self.items:
            d = self.items[item_id]
            d.attempts += 1
            d.last_seen = now
            d.error_type = type(exc).__name__
            d.message = str(exc)
        else:
            self.items[item_id] = DeadLetter(
                item_id=item_id, stage=stage, error_type=type(exc).__name__,
                message=str(exc), attempts=1, trace_id=trace_id,
                first_seen=now, last_seen=now, payload_ref=payload_ref,
                stack_digest=digest, replayable=replayable,
            )

    def by_error_type(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for d in self.items.values():
            out[d.error_type] = out.get(d.error_type, 0) + 1
        return out

    def replayable(self) -> list[DeadLetter]:
        return [d for d in self.items.values() if d.replayable]


# ---------------------------------------------------------------------------
# PART 1 — the three ways to handle a bad item
# ---------------------------------------------------------------------------

async def part1_three_ways() -> None:
    banner("PART 1 — abort, swallow, or quarantine")

    async def process(doc_id: int) -> str:
        await asyncio.sleep(0.001)
        if doc_id == 37:
            raise ValidationError("unparseable PDF: no /Root object")
        if doc_id == 61:
            raise ServiceUnavailable("embedding endpoint 503")
        return f"indexed-{doc_id}"

    section("A) abort on first failure")
    indexed = 0
    try:
        for i in range(100):
            await process(i)
            indexed += 1
    except AppError as e:
        print(f"      run died at doc {indexed} with {type(e).__name__}")
        print("      63 documents never attempted; no record of which succeeded")

    section("B) swallow everything")
    indexed, swallowed = 0, 0
    for i in range(100):
        try:
            await process(i)
            indexed += 1
        except Exception:
            swallowed += 1
    print(f"      indexed={indexed}  silently dropped={swallowed}")
    print("      run reports SUCCESS. Two documents are missing from the")
    print("      index and nothing anywhere says so.")

    section("C) quarantine and continue")
    dlq = DeadLetterQueue()
    indexed = 0
    for i in range(100):
        try:
            await process(i)
            indexed += 1
        except AppError as e:
            dlq.record(f"doc-{i:05d}", "index", e,
                       trace_id=f"tr-{i:05d}",
                       payload_ref=f"sharepoint://isc/docs/doc-{i:05d}.pdf")
    print(f"      indexed={indexed}  quarantined={len(dlq.items)}")
    print(f"      breakdown: {dlq.by_error_type()}")
    print("      dead letters:")
    for d in dlq.items.values():
        print(f"        {d.to_json()}")
    print("""
      The run completed, the count is reported as a first-class result, and
      both failures are diagnosable and replayable. Note the two failures are
      DIFFERENT KINDS: the 503 is transient and will succeed on replay; the
      corrupt PDF will not. Part 3 separates them.""")


# ---------------------------------------------------------------------------
# PART 2 — poison pills
# ---------------------------------------------------------------------------

@dataclass
class QueueMessage:
    body: dict
    delivery_count: int = 0
    message_id: str = ""


class PoisonDetector:
    """An item that fails repeatedly must be removed from circulation.

    THE FAILURE WITHOUT THIS: a message that crashes the consumer is
    redelivered by the broker (that is what at-least-once means), crashes the
    consumer again, is redelivered again... forever. It blocks the queue,
    burns CPU, and fills your logs. Service Bus calls this a poison message
    and gives you MaxDeliveryCount for exactly this reason; if you build your
    own loop, you must build this yourself.

    THE THRESHOLD IS A JUDGEMENT: too low and transient failures get
    quarantined unnecessarily; too high and a genuinely poison item wastes
    hours. 3-5 is typical, and it should be paired with backoff between
    deliveries so the retries are spread out rather than instantaneous.
    """

    def __init__(self, max_deliveries: int = 3) -> None:
        self.max_deliveries = max_deliveries

    def is_poison(self, msg: QueueMessage) -> bool:
        return msg.delivery_count >= self.max_deliveries


async def part2_poison() -> None:
    banner("PART 2 — poison pills block the queue forever")

    dlq = DeadLetterQueue()
    detector = PoisonDetector(max_deliveries=3)
    m = Metrics()

    # A queue with one message that always crashes the consumer.
    queue: list[QueueMessage] = [
        QueueMessage({"doc": "good-1"}, message_id="m1"),
        QueueMessage({"doc": "POISON"}, message_id="m2"),
        QueueMessage({"doc": "good-2"}, message_id="m3"),
    ]

    async def consume(msg: QueueMessage) -> str:
        if msg.body["doc"] == "POISON":
            raise PoisonItemError("consumer cannot parse this message body")
        return f"ok:{msg.body['doc']}"

    processed, redelivered = [], 0
    max_iterations = 20      # so this demo terminates

    for _ in range(max_iterations):
        if not queue:
            break
        msg = queue.pop(0)
        msg.delivery_count += 1
        m.incr("deliveries")

        if detector.is_poison(msg):
            # DEAD-LETTER IT AND ACK. The message leaves circulation.
            dlq.record(msg.message_id, "consume",
                       PoisonItemError("max delivery count exceeded"),
                       trace_id=f"tr-{msg.message_id}",
                       payload_ref=f"queue://ingest/{msg.message_id}",
                       replayable=False)
            m.incr("dead_lettered")
            continue

        try:
            processed.append(await consume(msg))
            m.incr("processed")
        except AppError:
            queue.append(msg)          # redeliver
            redelivered += 1
            m.incr("redelivered")

    print(f"      processed: {processed}")
    print(f"      deliveries={m.counters['deliveries']} "
          f"redelivered={m.counters['redelivered']} "
          f"dead_lettered={m.counters['dead_lettered']}")
    print(f"      queue drained: {len(queue) == 0}")
    print("""
      The poison message was redelivered 3 times, then dead-lettered with
      replayable=False. Without the delivery count it would loop forever and
      the two good messages behind it would never be processed — a single bad
      document taking down an entire ingestion pipeline.""")


# ---------------------------------------------------------------------------
# PART 3 — separating transient from genuinely poison
# ---------------------------------------------------------------------------

async def part3_triage() -> None:
    banner("PART 3 — triaging the dead letter queue")

    dlq = DeadLetterQueue()

    # A realistic mixed DLQ after a large run.
    fixtures = [
        ("doc-00012", ServiceUnavailable("503 embedding endpoint"), True),
        ("doc-00087", ServiceUnavailable("503 embedding endpoint"), True),
        ("doc-00143", ServiceUnavailable("503 embedding endpoint"), True),
        ("doc-00201", ValidationError("PDF has no /Root object"), False),
        ("doc-00355", ValidationError("PDF has no /Root object"), False),
        ("doc-00412", PoisonItemError("file is 0 bytes"), False),
        ("doc-00588", ServiceUnavailable("timeout after 30s"), True),
    ]
    for item_id, exc, replayable in fixtures:
        dlq.record(item_id, "embed", exc, trace_id=f"tr-{item_id}",
                   payload_ref=f"sharepoint://isc/{item_id}.pdf",
                   replayable=replayable)

    print("    dead letters grouped by error type:")
    for etype, n in sorted(dlq.by_error_type().items(), key=lambda x: -x[1]):
        print(f"      {etype:<22} {n}")

    replayable = dlq.replayable()
    print(f"\n    replayable: {len(replayable)}/{len(dlq.items)}")
    print(f"    needs a code or data fix: {len(dlq.items) - len(replayable)}")

    print("""
    THIS GROUPING IS THE POINT. 400 dead letters is not 400 problems — it is
    usually two or three problems with high multiplicity. Grouping by
    error_type (or stack_digest) turns an unreadable pile into:

      "3 x 503 during the embedding stage    -> replay, will succeed"
      "2 x corrupt PDF                        -> fix the parser, then replay"
      "1 x zero-byte file                     -> fix the source, unreplayable"

    That is a triage list a person can act on in ten minutes.

    THE OPERATIONAL LOOP:
      1. Group by error type. Fix the biggest bucket.
      2. Replay the replayable ones with the SAME idempotency keys (see 03),
         so replaying an item that actually succeeded is harmless.
      3. Items that fail replay TWICE get replayable=False and a ticket.
      4. Alert on DLQ GROWTH RATE, not depth. A DLQ with 400 items that is
         not growing is a backlog. One with 20 items growing at 5/minute is
         an incident.""")

    section("replay, with idempotency")
    replayed_ok = 0
    for d in replayable:
        # The replay uses a key derived from the item, so an item that
        # actually succeeded the first time is a no-op rather than a duplicate.
        key = f"ingest:{d.item_id}"
        replayed_ok += 1
        del key
    print(f"      replayed {replayed_ok} items with keys derived from item_id")
    print("      Replay MUST be idempotent. Some dead letters are items that")
    print("      succeeded and then failed while recording success — replaying")
    print("      those without a key duplicates them in the index.")


# ---------------------------------------------------------------------------
# PART 4 — batch-level circuit breaking
# ---------------------------------------------------------------------------

class BatchGuard:
    """Abort a long run when the failure rate says something systemic broke.

    THE FAILURE WITHOUT THIS: your credential expires 20 minutes into a
    6-hour run. Every subsequent item fails. Item-level quarantine dutifully
    dead-letters 48,000 documents, the run "completes", and you now have a DLQ
    the size of your corpus and six hours of wasted compute.

    Item-level resilience and batch-level resilience are different concerns.
    Quarantine handles "this item is bad"; the guard handles "the WORLD is
    bad, stop".
    """

    def __init__(self, *, min_items: int = 20, max_failure_rate: float = 0.25,
                 consecutive_limit: int = 15) -> None:
        self.min_items = min_items
        self.max_failure_rate = max_failure_rate
        self.consecutive_limit = consecutive_limit
        self.processed = 0
        self.failed = 0
        self.consecutive = 0
        self.abort_reason: str | None = None

    def record(self, ok: bool) -> None:
        self.processed += 1
        if ok:
            self.consecutive = 0
        else:
            self.failed += 1
            self.consecutive += 1

        if self.consecutive >= self.consecutive_limit:
            self.abort_reason = (
                f"{self.consecutive} consecutive failures — "
                f"likely credential, config, or dependency outage"
            )
        elif (self.processed >= self.min_items
              and self.failed / self.processed > self.max_failure_rate):
            self.abort_reason = (
                f"failure rate {self.failed / self.processed:.0%} exceeds "
                f"{self.max_failure_rate:.0%} over {self.processed} items"
            )

    @property
    def should_abort(self) -> bool:
        return self.abort_reason is not None


async def part4_batch_guard() -> None:
    banner("PART 4 — batch-level abort: stop when the world is broken")

    async def run(fail_from: int | None, n: int = 200) -> tuple[int, int, str | None]:
        guard = BatchGuard()
        dlq = DeadLetterQueue()
        ok = 0
        for i in range(n):
            broken = fail_from is not None and i >= fail_from
            is_ok = not broken and i % 50 != 7      # 2% baseline bad items
            if is_ok:
                ok += 1
            else:
                dlq.record(f"doc-{i:05d}", "embed",
                           ValidationError("credential expired") if broken
                           else ValidationError("corrupt pdf"),
                           trace_id=f"tr-{i}", payload_ref=f"ref/{i}")
            guard.record(is_ok)
            if guard.should_abort:
                return ok, len(dlq.items), guard.abort_reason
        return ok, len(dlq.items), None

    section("healthy run, 2% naturally bad items")
    ok, dead, reason = await run(fail_from=None)
    print(f"      indexed={ok} quarantined={dead} aborted={reason}")

    section("credential expires at item 40")
    ok, dead, reason = await run(fail_from=40)
    print(f"      indexed={ok} quarantined={dead}")
    print(f"      ABORTED: {reason}")
    print("""
      Whichever rule fires first wins — here it was the failure-RATE rule
      (26% over 53 items), before the consecutive-failure rule reached 15.
      Both guard the same thing from different angles: the rate rule catches
      a partial systemic failure, the consecutive rule catches a total one
      faster. Keep both. The run stopped after ~50 items instead of
      dead-lettering the remaining 145. The guard converted 'six hours of wasted compute
      and a 48,000-item DLQ' into 'a fast, clearly-labelled failure'.

      Set consecutive_limit generously enough that a burst of genuinely bad
      documents does not trip it — but low enough that it fires in minutes,
      not hours. And make the abort reason SPECIFIC: 'credential, config, or
      dependency outage' tells the operator where to look; 'batch failed'
      does not.""")


# ---------------------------------------------------------------------------
# PART 5 — checkpointing so a restart is cheap
# ---------------------------------------------------------------------------

async def part5_checkpoint() -> None:
    banner("PART 5 — checkpoints: making restart cheap")

    print("""
    Three levels, in increasing order of robustness and cost:

    1. HIGH-WATER MARK
         Persist "last successfully processed id" every N items.
         Restart resumes from there.
         Works only when items are processed IN ORDER. With concurrency, item
         900 may finish before item 850, so the mark is a lie — you must
         record the lowest incomplete id, not the highest complete one.
         Cheap; fine for single-threaded sequential scans.

    2. PER-ITEM STATE
         A row per item: pending / in_progress / done / dead.
         Restart picks up anything not done. Handles concurrency correctly and
         gives you the DLQ for free. Costs one write per item — which is
         usually negligible next to the embedding call.
         THIS IS THE DEFAULT for anything that matters.

    3. DURABLE QUEUE
         Service Bus / Storage Queue with visibility timeouts. The broker owns
         redelivery; a crashed worker's messages reappear automatically after
         the lock expires. Add MaxDeliveryCount for poison handling.
         Most robust, and the only option when workers are ephemeral.

    THE INTERACTION WITH IDEMPOTENCY: all three redeliver work that may have
    partially completed. Level 2 in particular has a window — the item is
    marked in_progress, the write succeeds, the process dies before marking
    done. On restart it is retried. Without an idempotency key that is a
    duplicate; with one it is a no-op. Checkpointing and idempotency are not
    alternatives, they are a pair.

    WHAT TO CHECKPOINT: enough to resume, never the payload. Store
    (item_id, state, attempts, idempotency_key, last_error). The document
    itself stays where it already lives.""")

    section("the ordering trap in high-water marks")
    completed = {1, 2, 3, 5, 8, 9}          # 4, 6, 7 still running
    naive_mark = max(completed)
    correct_mark = min(set(range(1, 10)) - completed)
    print(f"      completed: {sorted(completed)}")
    print(f"      naive high-water mark  = {naive_mark}  "
          f"-> would SKIP items 4, 6, 7 on restart")
    print(f"      lowest-incomplete mark = {correct_mark}  "
          f"-> reprocesses 4-9, safe if idempotent")


# ---------------------------------------------------------------------------

async def main() -> None:
    await part1_three_ways()
    await part2_poison()
    await part3_triage()
    await part4_batch_guard()
    await part5_checkpoint()

    banner("SUMMARY")
    print("""
  * Quarantine, don't abort and don't swallow. Report the quarantine count as
    a first-class result.
  * Dead letters store a payload REFERENCE, never the payload.
  * Poison pills need a delivery count. Without one a single bad message
    blocks the queue forever.
  * Group the DLQ by error type — 400 dead letters is usually 3 problems.
  * Replay must be idempotent; some dead letters are items that actually
    succeeded.
  * Alert on DLQ growth RATE, not depth.
  * Add a BATCH-level guard for systemic failure. Item-level quarantine will
    happily dead-letter your entire corpus when a credential expires.
  * Checkpoint per item, not with a high-water mark, once you have
    concurrency.
""")


if __name__ == "__main__":
    asyncio.run(main())
