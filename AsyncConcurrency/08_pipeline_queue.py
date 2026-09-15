"""
08 — Producer/consumer pipelines: the shape of every ingestion job.

THE PROBLEM WITH bounded_map FOR INGESTION
------------------------------------------
`bounded_map` (from 04) is right for "I have a list, apply f to all of it".
Ingestion is not that. Ingestion is:

    enumerate SharePoint  ->  download  ->  parse  ->  chunk  ->  embed  ->  index
       (fast, paged)         (I/O)       (CPU)     (CPU)     (rate-limited)  (I/O)

Each stage has a different natural concurrency and a different bottleneck.
Running them as sequential bounded_maps means:
  * you materialise every intermediate result in memory (50k parsed documents),
  * the CPU stages sit idle while the I/O stages run and vice versa,
  * one slow document blocks the entire batch boundary.

A staged pipeline with bounded queues between stages fixes all three. Each
stage runs continuously, each queue provides backpressure, and memory is
bounded by (queue sizes x item size) rather than by corpus size.

WHAT THIS SECTION ALSO COVERS
-----------------------------
Poison-pill shutdown, dead-letter handling, per-stage metrics, and the
graceful-vs-abrupt distinction. These are the parts that turn a demo pipeline
into one you can operate.

Run:  python 08_pipeline_queue.py
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

from fake_llm import FakeLLMClient, Timer, banner

# ---------------------------------------------------------------------------
# Domain objects
# ---------------------------------------------------------------------------

@dataclass
class Doc:
    doc_id: str
    site: str
    bytes_: int = 0
    text: str = ""
    chunks: list[str] = field(default_factory=list)
    vectors: list[list[float]] = field(default_factory=list)
    # Carried end-to-end so a failure anywhere can be traced back.
    trace_id: str = ""


@dataclass
class StageMetrics:
    """Per-stage counters. Emit these; they tell you which stage is the
    bottleneck, which is the only thing worth knowing when tuning."""
    name: str
    processed: int = 0
    failed: int = 0
    total_wait_s: float = 0.0   # time spent BLOCKED on the output queue
    total_work_s: float = 0.0

    def report(self) -> str:
        util = (
            self.total_work_s / (self.total_work_s + self.total_wait_s)
            if (self.total_work_s + self.total_wait_s) > 0
            else 0.0
        )
        return (
            f"{self.name:<10} ok={self.processed:<4} fail={self.failed:<3} "
            f"work={self.total_work_s:5.2f}s blocked={self.total_wait_s:5.2f}s "
            f"util={util:5.1%}"
        )


# ---------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------

class IngestPipeline:
    """A four-stage ingestion pipeline with bounded queues between stages.

    Design notes worth defending in a review:

      * Queue sizes are small (4-8). A large queue hides backpressure and
        converts a throughput problem into a memory problem. If you find
        yourself raising maxsize to "fix" a stall, you are papering over the
        actual bottleneck.
      * Each stage has its own worker count, sized to its own constraint:
        download is I/O (high), parse is CPU (low, = cores), embed is
        rate-limited (set by quota, not by cores).
      * Failures go to a dead-letter list rather than killing the run. A
        50,000-document ingestion that dies on document 31,000 because one PDF
        is corrupt is an operational disaster; quarantining it is correct.
      * Shutdown uses sentinels (one per downstream worker), so every worker
        gets exactly one and exits cleanly.
    """

    def __init__(
        self,
        *,
        download_workers: int = 4,
        parse_workers: int = 2,
        embed_workers: int = 3,
        queue_size: int = 4,
    ) -> None:
        self.download_workers = download_workers
        self.parse_workers = parse_workers
        self.embed_workers = embed_workers

        # BOUNDED queues. This is the backpressure mechanism.
        self.q_download: asyncio.Queue = asyncio.Queue(maxsize=queue_size)
        self.q_parse: asyncio.Queue = asyncio.Queue(maxsize=queue_size)
        self.q_embed: asyncio.Queue = asyncio.Queue(maxsize=queue_size)

        self.metrics = {
            n: StageMetrics(n)
            for n in ("enumerate", "download", "parse", "embed", "index")
        }
        self.dead_letters: list[tuple[str, str]] = []
        self.indexed: list[Doc] = []

        self.embed_client = FakeLLMClient(seed=61, base_latency_s=0.03, jitter_s=0.02)

    # -- helper: put with wait accounting ----------------------------------

    async def _put(self, q: asyncio.Queue, item, stage: str) -> None:
        """Put, measuring how long we were blocked.

        That blocked time is the single most useful diagnostic in a pipeline:
        the stage with high blocked-time is FASTER than its downstream, and
        the stage with ~zero blocked time and high work-time is your
        bottleneck. Without this instrumentation you will tune the wrong stage.
        """
        t0 = time.perf_counter()
        await q.put(item)
        self.metrics[stage].total_wait_s += time.perf_counter() - t0

    # -- stages ------------------------------------------------------------

    async def stage_enumerate(self, n_docs: int) -> None:
        """Lists documents from a source (Graph delta query, blob listing).

        Deliberately a GENERATOR-shaped producer, not a materialised list —
        real enumeration is paged and can be very large. `await q.put(...)`
        naturally throttles the paging.
        """
        for i in range(n_docs):
            doc = Doc(
                doc_id=f"doc-{i:04d}",
                site="contoso.sharepoint.com/sites/isc",
                trace_id=f"tr-{i:04d}",
            )
            await self._put(self.q_download, doc, "enumerate")
            self.metrics["enumerate"].processed += 1

        for _ in range(self.download_workers):
            await self.q_download.put(None)

    async def stage_download(self, wid: int) -> None:
        m = self.metrics["download"]
        while True:
            doc = await self.q_download.get()
            if doc is None:
                self.q_download.task_done()
                return
            t0 = time.perf_counter()
            try:
                await asyncio.sleep(0.02)          # network fetch
                doc.bytes_ = 10_000 + hash(doc.doc_id) % 5_000
                m.processed += 1
                m.total_work_s += time.perf_counter() - t0
                await self._put(self.q_parse, doc, "download")
            except Exception as e:  # noqa: BLE001
                m.failed += 1
                self.dead_letters.append((doc.doc_id, f"download: {e}"))
            finally:
                self.q_download.task_done()

    async def stage_parse(self, wid: int) -> None:
        m = self.metrics["parse"]
        while True:
            doc = await self.q_parse.get()
            if doc is None:
                self.q_parse.task_done()
                return
            t0 = time.perf_counter()
            try:
                # Every 17th doc is corrupt — quarantine, do not abort the run.
                if int(doc.doc_id.split("-")[1]) % 17 == 16:
                    raise ValueError("unparseable PDF: no /Root object")

                # NOTE: real parsing is CPU-bound and belongs in to_thread.
                # Here: await asyncio.to_thread(parse_pdf, doc.bytes_)
                await asyncio.sleep(0.03)
                doc.text = f"content of {doc.doc_id}"
                doc.chunks = [f"{doc.text} chunk {c}" for c in range(3)]
                m.processed += 1
                m.total_work_s += time.perf_counter() - t0
                await self._put(self.q_embed, doc, "parse")
            except Exception as e:  # noqa: BLE001
                m.failed += 1
                # The dead letter carries the trace_id, so this document can be
                # found, fixed, and replayed without re-running the corpus.
                self.dead_letters.append((doc.doc_id, f"parse[{doc.trace_id}]: {e}"))
            finally:
                self.q_parse.task_done()

    async def stage_embed_and_index(self, wid: int) -> None:
        m_embed, m_index = self.metrics["embed"], self.metrics["index"]
        while True:
            doc = await self.q_embed.get()
            if doc is None:
                self.q_embed.task_done()
                return
            t0 = time.perf_counter()
            try:
                # Batch the chunks into ONE embedding request. Batching first,
                # concurrency second — see 04.
                doc.vectors = await self.embed_client.embed(doc.chunks)
                m_embed.processed += 1
                m_embed.total_work_s += time.perf_counter() - t0

                t1 = time.perf_counter()
                await asyncio.sleep(0.01)  # index write
                self.indexed.append(doc)
                m_index.processed += 1
                m_index.total_work_s += time.perf_counter() - t1
            except Exception as e:  # noqa: BLE001
                m_embed.failed += 1
                self.dead_letters.append((doc.doc_id, f"embed[{doc.trace_id}]: {e}"))
            finally:
                self.q_embed.task_done()

    # -- orchestration -----------------------------------------------------

    async def run(self, n_docs: int) -> None:
        """Wire the stages together with correct sentinel propagation.

        The fiddly part: each stage must emit exactly one sentinel PER
        DOWNSTREAM WORKER, and only after all of its own workers have finished.
        Getting this wrong gives you either a hang (too few sentinels) or
        workers exiting early (sentinel consumed by the wrong worker).
        """
        async with asyncio.TaskGroup() as tg:
            tg.create_task(self.stage_enumerate(n_docs), name="enumerate")

            async def download_group() -> None:
                async with asyncio.TaskGroup() as g:
                    for i in range(self.download_workers):
                        g.create_task(self.stage_download(i), name=f"download-{i}")
                # All download workers done => nothing more will enter q_parse.
                for _ in range(self.parse_workers):
                    await self.q_parse.put(None)

            async def parse_group() -> None:
                async with asyncio.TaskGroup() as g:
                    for i in range(self.parse_workers):
                        g.create_task(self.stage_parse(i), name=f"parse-{i}")
                for _ in range(self.embed_workers):
                    await self.q_embed.put(None)

            async def embed_group() -> None:
                async with asyncio.TaskGroup() as g:
                    for i in range(self.embed_workers):
                        g.create_task(self.stage_embed_and_index(i), name=f"embed-{i}")

            tg.create_task(download_group(), name="download-group")
            tg.create_task(parse_group(), name="parse-group")
            tg.create_task(embed_group(), name="embed-group")


# ---------------------------------------------------------------------------
# PART 1 — run it
# ---------------------------------------------------------------------------

async def part1_run_pipeline() -> None:
    banner("PART 1 — a four-stage ingestion pipeline")

    p = IngestPipeline(download_workers=4, parse_workers=2, embed_workers=3,
                       queue_size=4)
    with Timer("ingest 40 documents"):
        await p.run(40)

    print(f"\n    indexed: {len(p.indexed)}   dead-lettered: {len(p.dead_letters)}")
    print("\n    per-stage metrics:")
    for m in p.metrics.values():
        print(f"      {m.report()}")

    print("\n    dead letters (quarantined, run continued):")
    for doc_id, reason in p.dead_letters[:3]:
        print(f"      {doc_id}: {reason}")

    print("""
    READ THE 'blocked' COLUMN. A stage with high blocked time is waiting on
    its downstream — it is not the bottleneck. The stage with high work time
    and low blocked time IS the bottleneck, and is the only one worth adding
    workers to. Adding workers anywhere else just moves the queue.""")


# ---------------------------------------------------------------------------
# PART 2 — tuning: prove where the bottleneck is
# ---------------------------------------------------------------------------

async def part2_tuning() -> None:
    banner("PART 2 — worker counts: measure, don't guess")

    configs = [
        ("balanced        ", {"download_workers": 4, "parse_workers": 2, "embed_workers": 3}),
        ("more downloaders", {"download_workers": 12, "parse_workers": 2, "embed_workers": 3}),
        ("more parsers    ", {"download_workers": 4, "parse_workers": 8, "embed_workers": 3}),
        ("more embedders  ", {"download_workers": 4, "parse_workers": 2, "embed_workers": 12}),
    ]
    for label, cfg in configs:
        p = IngestPipeline(queue_size=4, **cfg)
        t0 = time.perf_counter()
        await p.run(40)
        dt = time.perf_counter() - t0
        print(f"    {label}  {dt:5.2f}s   indexed={len(p.indexed)}")

    print("""
    Only one of these moves the number meaningfully. That is the lesson:
    scaling a non-bottleneck stage does nothing except consume connections
    and memory. Instrument first, then scale the stage the instrumentation
    identifies — the intuition about which stage "feels slow" is usually
    wrong, because the slow-feeling stage is often the one being starved.""")


# ---------------------------------------------------------------------------
# PART 3 — queue sizing and the memory/throughput trade
# ---------------------------------------------------------------------------

async def part3_queue_size() -> None:
    banner("PART 3 — queue depth: backpressure vs buffering")

    for qs in (1, 4, 64):
        p = IngestPipeline(download_workers=4, parse_workers=2,
                           embed_workers=3, queue_size=qs)
        t0 = time.perf_counter()
        await p.run(40)
        dt = time.perf_counter() - t0
        print(f"    queue_size={qs:<3} -> {dt:5.2f}s")

    print("""
    MEASURED RESULT: queue depth barely moves the wall clock here — and that
    is the finding, not a flat demo.

    Why: when one stage is a hard bottleneck (embed, at 100% utilisation from
    PART 1), every other stage is starved regardless of how much buffer sits
    between them. Throughput is set by the bottleneck's service rate. Buffer
    depth cannot exceed it; it can only decide how much work sits idle in RAM
    waiting for it.

    So the real trade-off is NOT time-vs-memory. It is:
      * depth too small  -> stages stall on jitter, and you lose throughput
                            only when arrival times are bursty (not visible
                            with our near-uniform latencies)
      * depth moderate   -> absorbs jitter; the useful range
      * depth large      -> zero throughput gain, linear memory growth at
                            (depth x item size). Parsed PDFs at ~5MB and a
                            depth-1000 queue is 5GB.

    The dangerous pattern this rules out: "the pipeline stalls, so raise the
    queue size". Raising the queue never fixes a bottleneck — it just hides
    the backpressure signal that was correctly telling you where the problem
    was, and converts a throughput problem into an OOM.""")


# ---------------------------------------------------------------------------
# PART 4 — resilient shutdown mid-run
# ---------------------------------------------------------------------------

async def part4_shutdown() -> None:
    banner("PART 4 — cancelling a pipeline mid-run")

    p = IngestPipeline(download_workers=4, parse_workers=2, embed_workers=3)

    task = asyncio.create_task(p.run(200), name="pipeline")
    await asyncio.sleep(0.25)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    print(f"    cancelled mid-run: {len(p.indexed)} documents fully indexed")
    print(f"    enumerate reached: {p.metrics['enumerate'].processed}")
    print("""
    Because TaskGroup owns every worker, cancellation propagates to all of
    them and the whole pipeline unwinds together — no orphaned workers still
    writing to an index after the job "stopped".

    But note what is NOT solved: the documents in flight are simply lost. For
    a restartable job you need, in order of increasing robustness:
      1. a durable checkpoint of the last successfully indexed doc_id,
      2. idempotent index writes keyed on doc_id + content hash, so a replay
         overwrites rather than duplicates,
      3. a real queue (Service Bus, Storage Queue) with visibility timeouts,
         so unacked messages are redelivered automatically.
    Option 3 is the one to reach for if the corpus is large enough that
    re-running from scratch is unacceptable.""")


# ---------------------------------------------------------------------------

async def main() -> None:
    await part1_run_pipeline()
    await part2_tuning()
    await part3_queue_size()
    await part4_shutdown()

    banner("SUMMARY")
    print("""
  * Stage the pipeline; give each stage its own worker count and constraint.
  * Bounded queues between stages = backpressure = bounded memory.
  * Instrument blocked-time per stage; it identifies the bottleneck directly.
  * Quarantine per-item failures to a dead-letter list with the trace_id.
    Never let one bad document abort a large run.
  * Sentinels: one per downstream worker, emitted after the upstream group
    completes. Nested TaskGroups make that ordering explicit.
  * Cancellation is clean but lossy — add checkpoints, idempotent writes, or
    a durable queue if the work must survive a restart.
""")


if __name__ == "__main__":
    asyncio.run(main())
