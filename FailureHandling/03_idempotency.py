"""
03 — Idempotency: making retries SAFE, not just useful.

THE PROBLEM IN ONE SENTENCE
---------------------------
When a write times out, you do not know whether it happened.

That is not a rare edge case. Any request that crosses a network has three
outcomes, not two:
    success        you got a 200
    clean failure  you got a 4xx/5xx, nothing was applied
    UNKNOWN        timeout, connection reset, your process died mid-call

The third is the one that matters, and it is indistinguishable from the second
at the client. Retrying a clean failure is free. Retrying an unknown duplicates
the side effect.

For an agent that files invoice disputes, creates purchase orders, or posts
approvals, "we retried and created two disputes" is a correctness bug that
reaches a human's desk. For pure generation it is merely wasted money.

THE CONTRACT
------------
Exactly-once delivery does not exist over an unreliable network. What exists is
at-least-once delivery plus idempotent processing, which together give you
effectively-once semantics. Every durable system you rely on (Service Bus,
Kafka, SQS) makes exactly this trade, and pushes dedup to you.

Run:  python 03_idempotency.py
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum

from failure_lab import (
    AppError,
    FlakyWriteAPI,
    banner,
    section,
)

# ---------------------------------------------------------------------------
# PART 1 — the duplicate, demonstrated
# ---------------------------------------------------------------------------

async def part1_the_duplicate() -> None:
    banner("PART 1 — the write that happened but looked like it failed")

    async def retry_write(api: FlakyWriteAPI, payload: dict,
                          key: str | None) -> dict:
        last = None
        for _ in range(3):
            try:
                return await api.write(payload, idempotency_key=key)
            except AppError as e:
                last = e
        raise last

    section("WITHOUT an idempotency key")
    api = FlakyWriteAPI("erp", fail_first_n=2)
    payload = {"type": "invoice_dispute", "invoice": "INV-88", "amount": 1250.00}
    result = await retry_write(api, payload, key=None)
    print(f"      final result: {result['id']}")
    print(f"      writes actually APPLIED server-side: {len(api.applied)}")
    for r in api.applied:
        print(f"        {r['id']}  {r['type']}  {r['invoice']}  ${r['amount']}")
    print("      -> THREE disputes filed for one invoice. Two of them will")
    print("         reach a human, who now has to work out which is real.")

    section("WITH an idempotency key")
    api2 = FlakyWriteAPI("erp", fail_first_n=2)
    key = "dispute:INV-88:2026-08-20"
    result = await retry_write(api2, payload, key=key)
    print(f"      final result: {result['id']}  "
          f"deduplicated={result.get('deduplicated', False)}")
    print(f"      writes actually APPLIED server-side: {len(api2.applied)}")
    print("      -> ONE dispute. Retries 2 and 3 returned the ORIGINAL result")
    print("         from the server's dedup store instead of writing again.")


# ---------------------------------------------------------------------------
# PART 2 — choosing the key
# ---------------------------------------------------------------------------

def key_random() -> str:
    """WRONG for retries. A fresh uuid per ATTEMPT defeats the whole purpose —
    every retry looks like a new request to the server."""
    return str(uuid.uuid4())


def key_per_request() -> str:
    """Correct if generated ONCE per logical operation and reused across all
    attempts. This is what most SDKs mean by `idempotency_key`."""
    return str(uuid.uuid4())


def key_natural(entity: str, operation: str, period: str) -> str:
    """BEST when a natural key exists. Survives process restarts, retries from
    a different worker, and replays from a queue — none of which an in-memory
    uuid does.

    'dispute:INV-88:2026-08-20' means "at most one dispute per invoice per
    day", which is a business rule you can state and defend, not an
    implementation detail.
    """
    return f"{operation}:{entity}:{period}"


def key_content_hash(payload: dict) -> str:
    """Derive from the payload. Two identical payloads dedup automatically.

    GOTCHA: must be a CANONICAL serialisation — sorted keys, fixed separators,
    stable float formatting. Otherwise `{"a":1,"b":2}` and `{"b":2,"a":1}`
    hash differently and dedup silently stops working.

    SECOND GOTCHA: this dedups legitimate repeats too. If a user genuinely
    wants to file the same dispute twice, a content hash blocks them. Use it
    where repeats are always errors, not where they are merely unusual.
    """
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:32]


async def part2_key_choice() -> None:
    banner("PART 2 — choosing an idempotency key")

    payload = {"invoice": "INV-88", "amount": 1250.00, "type": "dispute"}
    reordered = {"type": "dispute", "amount": 1250.00, "invoice": "INV-88"}

    print(f"    random per attempt : {key_random()[:16]}...  DEFEATS dedup")
    print(f"    per logical request: {key_per_request()[:16]}...  ok if reused")
    print(f"    natural key        : {key_natural('INV-88', 'dispute', '2026-08-20')}")
    print(f"    content hash       : {key_content_hash(payload)[:16]}...")
    print(f"    same payload, keys reordered: "
          f"{key_content_hash(reordered)[:16]}...  <- identical, as required")

    print("""
    DECISION ORDER:
      1. Natural key if one exists. Survives restarts and queue replays.
      2. Content hash if repeats are always errors.
      3. Client-generated uuid, created ONCE per logical operation, persisted
         alongside the work item so a restart reuses it.
      4. Never a uuid generated inside the retry loop.

    SCOPE AND TTL, which people forget:
      * Keys must be scoped per tenant/principal, or one caller's key can
        collide with another's and return them someone else's result.
      * The server's dedup store needs a TTL — typically 24h. Long enough to
        cover any retry window, short enough that the store does not grow
        forever. State the number; it is the window in which a genuine repeat
        will be silently swallowed.""")


# ---------------------------------------------------------------------------
# PART 3 — server-side dedup done properly
# ---------------------------------------------------------------------------

class RecordState(Enum):
    IN_FLIGHT = "in_flight"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class DedupRecord:
    key: str
    state: RecordState
    request_hash: str
    result: dict | None = None
    created_at: float = field(default_factory=time.monotonic)


class IdempotencyStore:
    """A correct server-side dedup store. The subtleties are all in the states.

    NAIVE VERSION (broken):
        if key in store: return store[key]
        result = do_work()
        store[key] = result

    Fails because two concurrent requests with the same key both see an empty
    store and both do the work. The window is small; at scale it is hit daily.

    CORRECT VERSION: reserve the key atomically FIRST, in an IN_FLIGHT state.
    A second request seeing IN_FLIGHT gets a 409 and retries later, rather
    than doing duplicate work.
    """

    def __init__(self, ttl: float = 86400.0) -> None:
        self.records: dict[str, DedupRecord] = {}
        self.ttl = ttl
        self._lock = asyncio.Lock()
        self.conflicts = 0
        self.replays = 0

    async def begin(self, key: str, request_hash: str) -> DedupRecord | None:
        """Reserve the key. Returns an existing record if one exists (caller
        should replay it), or None if this caller owns the work.

        The lock makes reserve-and-check atomic. In a real system this is a
        conditional insert: `INSERT ... ON CONFLICT DO NOTHING`, a Cosmos
        pre-condition on etag, or Redis `SET key val NX`.
        """
        async with self._lock:
            self._evict_expired()
            existing = self.records.get(key)
            if existing is not None:
                # SAFETY CHECK people skip: same key, DIFFERENT payload means
                # the client has a bug (or is malicious). Returning the cached
                # result would silently ignore their new request. Reject it.
                if existing.request_hash != request_hash:
                    raise ValueError(
                        f"idempotency key {key!r} reused with a different payload"
                    )
                if existing.state is RecordState.IN_FLIGHT:
                    self.conflicts += 1
                else:
                    self.replays += 1
                return existing

            self.records[key] = DedupRecord(key, RecordState.IN_FLIGHT, request_hash)
            return None

    async def complete(self, key: str, result: dict) -> None:
        async with self._lock:
            rec = self.records[key]
            rec.state = RecordState.COMPLETED
            rec.result = result

    async def fail(self, key: str) -> None:
        """Release the reservation so a retry can proceed.

        IMPORTANT: only for CLEAN failures where nothing was applied. If the
        side effect may have happened, leaving the record IN_FLIGHT (or moving
        it to a needs-reconciliation state) is safer than releasing it.
        """
        async with self._lock:
            self.records.pop(key, None)

    def _evict_expired(self) -> None:
        now = time.monotonic()
        for k in [k for k, r in self.records.items()
                  if now - r.created_at > self.ttl]:
            del self.records[k]


class IdempotentService:
    """A write endpoint with correct dedup semantics."""

    def __init__(self, store: IdempotencyStore) -> None:
        self.store = store
        self.work_performed = 0

    async def create_dispute(self, payload: dict, *, idempotency_key: str,
                             work_duration: float = 0.05) -> dict:
        request_hash = key_content_hash(payload)
        existing = await self.store.begin(idempotency_key, request_hash)

        if existing is not None:
            if existing.state is RecordState.IN_FLIGHT:
                # 409. The client should back off and retry — the original is
                # still running. Returning a duplicate-created 200 here would
                # be a lie.
                raise AppError("409 conflict: request already in flight",
                               retry_after=0.1)
            return {**existing.result, "replayed": True}

        try:
            self.work_performed += 1
            await asyncio.sleep(work_duration)
            result = {"dispute_id": f"D-{self.work_performed:04d}", **payload}
            await self.store.complete(idempotency_key, result)
            return result
        except Exception:
            await self.store.fail(idempotency_key)
            raise


async def part3_server_dedup() -> None:
    banner("PART 3 — server-side dedup, including the concurrent case")

    store = IdempotencyStore()
    svc = IdempotentService(store)
    payload = {"invoice": "INV-88", "amount": 1250.00}
    key = "dispute:INV-88:2026-08-20"

    section("sequential retries -> replay")
    r1 = await svc.create_dispute(payload, idempotency_key=key)
    r2 = await svc.create_dispute(payload, idempotency_key=key)
    print(f"      first : {r1['dispute_id']} replayed={r1.get('replayed', False)}")
    print(f"      second: {r2['dispute_id']} replayed={r2.get('replayed', False)}")
    print(f"      work performed: {svc.work_performed}")

    section("CONCURRENT duplicates -> 409, not double work")
    store2 = IdempotencyStore()
    svc2 = IdempotentService(store2)

    async def attempt(i: int):
        try:
            r = await svc2.create_dispute(payload, idempotency_key=key,
                                          work_duration=0.05)
            return f"req{i}: {r['dispute_id']}"
        except AppError as e:
            return f"req{i}: {e}"

    results = await asyncio.gather(*(attempt(i) for i in range(5)))
    for r in results:
        print(f"      {r}")
    print(f"      work performed: {svc2.work_performed}  (must be 1)")
    print(f"      conflicts returned: {store2.conflicts}")

    section("...and the NAIVE store, for comparison")

    class NaiveService:
        """check-then-write, no reservation. The bug, made runnable."""

        def __init__(self) -> None:
            self.cache: dict[str, dict] = {}
            self.work_performed = 0

        async def create_dispute(self, payload: dict, *,
                                 idempotency_key: str) -> dict:
            if idempotency_key in self.cache:          # CHECK
                return {**self.cache[idempotency_key], "replayed": True}
            self.work_performed += 1
            await asyncio.sleep(0.05)                  # <-- yield: others run here
            result = {"dispute_id": f"D-{self.work_performed:04d}", **payload}
            self.cache[idempotency_key] = result       # ACT, far too late
            return result

    naive = NaiveService()
    await asyncio.gather(*(
        naive.create_dispute(payload, idempotency_key=key) for _ in range(5)
    ))
    print(f"      work performed: {naive.work_performed}  <- MEASURED, not asserted")
    print("      Five duplicate disputes. Every request saw an empty cache")
    print("      before any of them wrote to it — check-then-act across an")
    print("      await is a race, and dedup is exactly where it hurts most.")
    print("""
      The window here is 50ms wide. In production it is however long your
      write takes, and it is hit whenever a client retries fast or two workers
      pick up the same queue message. Reserving the key atomically first is
      what makes it correct.""")

    section("same key, different payload -> rejected")
    try:
        await svc.create_dispute({"invoice": "INV-88", "amount": 9999.00},
                                 idempotency_key=key)
    except ValueError as e:
        print(f"      {e}")
    print("      Silently returning the cached result would mean the client's")
    print("      corrected amount was ignored with a 200 OK.")


# ---------------------------------------------------------------------------
# PART 4 — client-side: making the whole operation replayable
# ---------------------------------------------------------------------------

@dataclass
class WorkItem:
    """A unit of work carrying its OWN idempotency key, persisted with it.

    This is the piece that makes restart-safety work. If the key lives only in
    a local variable, a process crash between attempt 1 and attempt 2 means
    attempt 2 generates a fresh key and duplicates the write. If the key is
    persisted with the work item, any worker picking it up — after a restart,
    on a different pod, from a queue redelivery — uses the same key.
    """

    item_id: str
    payload: dict
    idempotency_key: str
    attempts: int = 0
    state: str = "pending"

    @classmethod
    def create(cls, item_id: str, payload: dict) -> WorkItem:
        return cls(
            item_id=item_id,
            payload=payload,
            # Generated ONCE, at item creation, never inside the retry loop.
            idempotency_key=f"{item_id}:{key_content_hash(payload)[:12]}",
        )


async def part4_restart_safety() -> None:
    banner("PART 4 — surviving a process restart mid-retry")

    store = IdempotencyStore()
    svc = IdempotentService(store)
    item = WorkItem.create("task-001", {"invoice": "INV-99", "amount": 500.0})

    section("worker A starts the write, then the pod is evicted")
    # Worker A begins the operation but dies before recording success.
    await svc.create_dispute(item.payload, idempotency_key=item.idempotency_key)
    print(f"      write applied server-side. work_performed={svc.work_performed}")
    print("      ...worker A dies before marking the item complete...")

    section("the queue redelivers; worker B picks it up")
    # Worker B reads the SAME persisted item, with the SAME key.
    r = await svc.create_dispute(item.payload, idempotency_key=item.idempotency_key)
    print(f"      worker B result: {r['dispute_id']} replayed={r.get('replayed')}")
    print(f"      work_performed still: {svc.work_performed}  <- no duplicate")

    print("""
    This is why the key is a FIELD on the persisted work item rather than a
    local variable. At-least-once delivery guarantees you WILL see redelivery;
    the key is what makes redelivery harmless.

    The same argument applies to Service Bus / Storage Queue visibility
    timeouts: a message whose lock expires is redelivered, and that is normal
    operation, not an error. Design for it rather than trying to prevent it.""")


# ---------------------------------------------------------------------------
# PART 5 — what is naturally idempotent, and what is not
# ---------------------------------------------------------------------------

async def part5_operation_shapes() -> None:
    banner("PART 5 — designing operations to be idempotent in the first place")

    print("""
    NATURALLY IDEMPOTENT (retry freely):
      GET / read                      no state change
      PUT with a full document        last write wins, converges
      DELETE by id                    second delete is a no-op
      SET status = 'approved'         absolute assignment
      upsert keyed on a natural id    converges

    NOT IDEMPOTENT (need a key, or a redesign):
      POST /disputes                  creates a new entity each time
      counter += 1                    relative mutation
      append to a list                accumulates
      send email / webhook            externally visible, uncancellable
      charge a card                   the expensive one

    THE REDESIGN THAT USUALLY WORKS: turn a relative mutation into an absolute
    one.

        BAD   balance = balance - amount
        GOOD  apply_transaction(txn_id, amount)   -- dedup on txn_id
                                                  -- balance derived from txns

    That is event sourcing in miniature, and it is why ledgers are built that
    way: a duplicate transaction record is detectable and ignorable, whereas a
    duplicate decrement is invisible after the fact.

    FOR AGENT TOOL CALLS SPECIFICALLY:
      * Mark every tool read-only or write. Read-only tools retry freely.
      * Every write tool takes an idempotency key derived from the trace_id
        plus the tool call id — both of which you already have, and both of
        which are stable across retries of the same logical step.
      * If a write tool cannot be made idempotent (sending an email), it must
        require explicit human confirmation rather than being retried
        automatically. "Do not retry" is not a sufficient control, because the
        process can crash after the send and before the record.""")

    section("idempotency key from an agent's own identifiers")
    trace_id = "tr-9f2a41"
    tool_call_id = "call_3"
    print(f"      trace_id={trace_id} tool_call_id={tool_call_id}")
    print(f"      -> key = {trace_id}:{tool_call_id}")
    print("      Stable across retries of that step, unique across steps and")
    print("      across requests. No extra state needed to generate it.")


# ---------------------------------------------------------------------------

async def main() -> None:
    await part1_the_duplicate()
    await part2_key_choice()
    await part3_server_dedup()
    await part4_restart_safety()
    await part5_operation_shapes()

    banner("SUMMARY")
    print("""
  * Every network write has THREE outcomes; the third (unknown) is why
    idempotency exists.
  * Exactly-once does not exist. At-least-once delivery + idempotent
    processing = effectively-once.
  * Generate the key ONCE per logical operation and persist it with the work
    item. A key generated inside the retry loop does nothing.
  * Prefer natural keys, then content hashes, then per-request uuids.
  * Server-side: reserve the key atomically BEFORE doing work, or concurrent
    duplicates both proceed.
  * Same key + different payload = reject, never replay.
  * Scope keys per tenant; give the dedup store a stated TTL.
  * Redesign relative mutations into absolute ones where you can — it removes
    the problem instead of managing it.
  * Non-idempotent, externally-visible actions (email, payment) need human
    confirmation, not automatic retry.
""")


if __name__ == "__main__":
    asyncio.run(main())
