"""Topic 3: Async I/O and bounded concurrency.

Asyncio is most useful when an application spends time waiting for network-bound
dependencies such as LLM APIs, databases, and storage:

    Task A ---- awaiting network -----------------+
    Task B ---- runs while A is waiting ----------+--> event loop
    Task C ---- awaiting a timer -----------------+

At each ``await``, a coroutine can let the event loop run another ready task.
This improves I/O throughput; it does not make CPU-bound Python work parallel.

This tutorial is deterministic, offline, and uses short sleeps as fake I/O.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass


PROMPTS = (
    "Summarize work order 101",
    "Summarize work order 102",
    "Summarize work order 103",
    "Summarize work order 104",
)
FAKE_IO_DELAY_SECONDS = 0.08


@dataclass
class FakeLLMService:
    """Simulate network waiting and track downstream concurrency."""

    default_delay_seconds: float = FAKE_IO_DELAY_SECONDS
    active_calls: int = 0
    maximum_active_calls: int = 0

    async def generate(
        self,
        prompt: str,
        delay_seconds: float | None = None,
    ) -> str:
        """Wait without blocking the event loop, then return a fake response."""
        delay = self.default_delay_seconds if delay_seconds is None else delay_seconds
        self.active_calls += 1
        self.maximum_active_calls = max(
            self.maximum_active_calls,
            self.active_calls,
        )

        try:
            # A real async client would await network I/O here.
            await asyncio.sleep(delay)
            return f"response for: {prompt}"
        finally:
            # ``finally`` also runs when timeout or cancellation interrupts sleep.
            self.active_calls -= 1


async def demo_sequential() -> float:
    """Await each call before starting the next one."""
    print("\nExample A - sequential I/O")
    service = FakeLLMService()
    started = time.perf_counter()

    responses: list[str] = []
    for prompt in PROMPTS:
        responses.append(await service.generate(prompt))

    elapsed = time.perf_counter() - started
    print(f"Completed {len(responses)} calls in about {elapsed:.2f}s")
    print(f"Maximum simultaneous calls: {service.maximum_active_calls}")
    return elapsed


async def demo_concurrent() -> float:
    """Start independent I/O operations together with structured concurrency."""
    print("\nExample B - concurrent I/O with TaskGroup")
    service = FakeLLMService()
    started = time.perf_counter()

    tasks: list[asyncio.Task[str]] = []
    async with asyncio.TaskGroup() as group:
        for prompt in PROMPTS:
            tasks.append(group.create_task(service.generate(prompt)))

    # Leaving a TaskGroup waits for every task or propagates a failure.
    responses = [task.result() for task in tasks]
    elapsed = time.perf_counter() - started
    print(f"Completed {len(responses)} calls in about {elapsed:.2f}s")
    print(f"Maximum simultaneous calls: {service.maximum_active_calls}")
    return elapsed


async def demo_unbounded_concurrency() -> int:
    """Show that scheduling every pending job can overwhelm a real service."""
    print("\nExample C - unbounded concurrency")
    service = FakeLLMService(default_delay_seconds=0.04)
    many_prompts = tuple(f"Classify request {number}" for number in range(1, 9))

    # gather is common and concise, but this starts all eight calls together.
    await asyncio.gather(*(service.generate(prompt) for prompt in many_prompts))

    print(f"Requests: {len(many_prompts)}")
    print(f"Maximum simultaneous calls: {service.maximum_active_calls}")
    print("Real providers also have connection, quota, memory, and cost limits.")
    return service.maximum_active_calls


async def call_with_limit(
    service: FakeLLMService,
    semaphore: asyncio.Semaphore,
    prompt: str,
) -> str:
    """Wait for capacity before occupying a downstream service connection."""
    async with semaphore:
        return await service.generate(prompt)


async def demo_bounded_concurrency(limit: int = 3) -> int:
    """Let many jobs wait while at most ``limit`` calls use the service."""
    print("\nExample D - bounded concurrency with Semaphore")
    service = FakeLLMService(default_delay_seconds=0.04)
    semaphore = asyncio.Semaphore(limit)
    many_prompts = tuple(f"Classify request {number}" for number in range(1, 9))

    await asyncio.gather(
        *(call_with_limit(service, semaphore, prompt) for prompt in many_prompts)
    )

    print(f"Requests: {len(many_prompts)}")
    print(f"Concurrency limit: {limit}")
    print(f"Maximum observed concurrency: {service.maximum_active_calls}")
    if service.maximum_active_calls > limit:
        raise AssertionError("the downstream concurrency limit was exceeded")
    return service.maximum_active_calls


async def demo_timeout() -> None:
    """Bound how long this caller is willing to wait for a dependency."""
    print("\nExample E - caller timeout")
    service = FakeLLMService()

    try:
        async with asyncio.timeout(0.05):
            await service.generate("Slow inference", delay_seconds=0.20)
    except TimeoutError:
        print("Request timed out as expected (limit=0.05s, service=0.20s).")
    else:
        raise AssertionError("the deliberately slow call should time out")

    if service.active_calls != 0:
        raise AssertionError("timed-out call did not clean up its active state")


async def slow_operation_with_cleanup(service: FakeLLMService) -> str:
    """Use finally for cleanup while allowing cancellation to propagate."""
    try:
        return await service.generate("No longer needed", delay_seconds=0.20)
    finally:
        print("Cancelled coroutine ran its cleanup.")


async def demo_cancellation() -> None:
    """Cancel work whose result is no longer needed."""
    print("\nExample F - task cancellation")
    service = FakeLLMService()
    task = asyncio.create_task(
        slow_operation_with_cleanup(service),
        name="obsolete-llm-request",
    )

    await asyncio.sleep(0.02)  # Let the task enter its simulated I/O wait.
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        print("Caller observed CancelledError as expected.")
    else:
        raise AssertionError("the cancelled task unexpectedly completed")

    if service.active_calls != 0:
        raise AssertionError("cancelled call did not clean up its active state")


def blocking_sdk_call(delay_seconds: float) -> str:
    """Represent a legacy synchronous client that cannot be changed."""
    time.sleep(delay_seconds)
    return "blocking result"


async def bad_blocking_call(delay_seconds: float) -> str:
    """Demonstrate what not to do: block the event-loop thread."""
    return blocking_sdk_call(delay_seconds)


async def isolated_blocking_call(delay_seconds: float) -> str:
    """Move unavoidable blocking work off the event-loop thread."""
    return await asyncio.to_thread(blocking_sdk_call, delay_seconds)


async def progress_marker(started: float) -> float:
    """Record when unrelated asynchronous work gets a chance to progress."""
    await asyncio.sleep(0.01)
    return time.perf_counter() - started


async def demo_blocking_code() -> None:
    """Compare event-loop blocking with isolating a synchronous dependency."""
    print("\nExample G - blocking-code awareness")
    blocking_delay = 0.06

    bad_started = time.perf_counter()
    bad_marker = asyncio.create_task(progress_marker(bad_started))
    await bad_blocking_call(blocking_delay)
    bad_progress_time = await bad_marker

    better_started = time.perf_counter()
    better_marker = asyncio.create_task(progress_marker(better_started))
    await isolated_blocking_call(blocking_delay)
    better_progress_time = await better_marker

    print(f"With time.sleep, unrelated task progressed after ~{bad_progress_time:.2f}s")
    print(f"With to_thread, unrelated task progressed after ~{better_progress_time:.2f}s")
    print("Prefer a native async client; use to_thread for unavoidable blocking I/O.")


async def main() -> None:
    """Run the async examples in one event loop."""
    print("Waiting can overlap, but downstream concurrency must stay bounded.")
    sequential_elapsed = await demo_sequential()
    concurrent_elapsed = await demo_concurrent()
    print(
        "I/O concurrency speedup observed: "
        f"about {sequential_elapsed / concurrent_elapsed:.1f}x"
    )

    unbounded_maximum = await demo_unbounded_concurrency()
    bounded_maximum = await demo_bounded_concurrency(limit=3)
    if unbounded_maximum <= bounded_maximum:
        raise AssertionError("bounded example should expose fewer simultaneous calls")

    await demo_timeout()
    await demo_cancellation()
    await demo_blocking_code()
    print("\nAsyncio overlaps I/O waiting; it does not parallelize CPU-bound Python.")


if __name__ == "__main__":
    asyncio.run(main())
