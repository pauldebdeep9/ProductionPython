"""
06 — Rotation: changing a secret without an outage.

THE PROBLEM
-----------
Every stored secret must be rotatable, and the naive rotation causes an
outage:

    t+0   new secret written to the vault
    t+0   old secret invalidated at the provider
    t+0   every running instance still holds the OLD value
    t+0   100% of requests fail
    t+8m  instances restart and pick up the new value

Eight minutes of total failure, every ninety days, per secret. Teams respond
by rotating less often, which is exactly backwards.

THE FIX has two halves, and you need both:
    1. The PROVIDER must accept two valid secrets during a window. Azure
       OpenAI and Storage give you Key1/Key2 precisely for this.
    2. Your APPLICATION must be able to pick up a new value without a
       restart, and must invalidate its cache on a 401 rather than waiting
       for a TTL.

Run:  python 06_rotation.py
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum

from config_lab import FakeKeyVault, banner, section, show

# ---------------------------------------------------------------------------
# A provider that accepts two keys — the Key1/Key2 pattern
# ---------------------------------------------------------------------------

@dataclass
class DualKeyProvider:
    """Models an Azure resource with a primary and secondary key.

    Both are valid simultaneously. That is the ENTIRE mechanism that makes
    zero-downtime rotation possible, and it is why the rotation procedure is:

        regenerate the SECONDARY  ->  switch clients to it  ->  regenerate
        the PRIMARY

    You never regenerate the key that is currently in use.
    """

    primary: str = "key1-original"
    secondary: str = "key2-original"
    calls_ok: int = 0
    calls_rejected: int = 0

    def call(self, key: str) -> str:
        if key in (self.primary, self.secondary):
            self.calls_ok += 1
            return "200 OK"
        self.calls_rejected += 1
        raise PermissionError("401 invalid key")

    def regenerate_primary(self) -> str:
        self.primary = f"key1-{int(time.time() * 1000) % 100000}"
        return self.primary

    def regenerate_secondary(self) -> str:
        self.secondary = f"key2-{int(time.time() * 1000) % 100000}"
        return self.secondary


# ---------------------------------------------------------------------------
# PART 1 — the naive rotation, and its outage
# ---------------------------------------------------------------------------

def part1_naive() -> None:
    banner("PART 1 — the naive rotation causes a full outage")

    provider = DualKeyProvider()
    # An instance that read its key at startup and holds it forever.
    held_key = provider.primary

    section("before rotation")
    show("requests", f"{sum(1 for _ in range(5) if provider.call(held_key))} OK")

    section("rotate: regenerate the key currently in use")
    provider.regenerate_primary()
    provider.secondary = provider.primary   # both replaced at once
    failures = 0
    for _ in range(5):
        try:
            provider.call(held_key)
        except PermissionError:
            failures += 1
    show("requests after rotation", f"{failures}/5 FAILED")
    print("""
      100% failure until every instance restarts. And note the instance has
      no way to recover on its own: it holds a stale value in memory and
      nothing tells it to re-read.""")


# ---------------------------------------------------------------------------
# PART 2 — the dual-key rotation
# ---------------------------------------------------------------------------

class RotationPhase(Enum):
    STEADY = "steady: all clients on primary"
    STAGE = "staged: secondary regenerated, nobody using it yet"
    MIGRATE = "migrating: clients moving to secondary"
    RETIRE = "retiring: primary regenerated, now unused"


def part2_dual_key() -> None:
    banner("PART 2 — dual-key rotation, zero failed requests")

    provider = DualKeyProvider()

    @dataclass
    class Instance:
        name: str
        key: str

        def request(self, p: DualKeyProvider) -> bool:
            try:
                p.call(self.key)
                return True
            except PermissionError:
                return False

    instances = [Instance(f"pod-{i}", provider.primary) for i in range(4)]

    def traffic() -> tuple[int, int]:
        ok = sum(1 for i in instances for _ in range(3) if i.request(provider))
        total = len(instances) * 3
        return ok, total

    phases: list[tuple[RotationPhase, str]] = []

    # --- STEADY ---
    ok, total = traffic()
    phases.append((RotationPhase.STEADY, f"{ok}/{total} OK"))

    # --- STAGE: regenerate the SECONDARY. Nobody is using it, so this is
    #     a no-op for traffic. This is the step people skip.
    new_secondary = provider.regenerate_secondary()
    ok, total = traffic()
    phases.append((RotationPhase.STAGE, f"{ok}/{total} OK"))

    # --- MIGRATE: move instances one at a time. Both keys are valid, so an
    #     instance mid-migration and one not yet migrated both work.
    for inst in instances:
        inst.key = new_secondary
        ok, total = traffic()
    phases.append((RotationPhase.MIGRATE, f"{ok}/{total} OK"))

    # --- RETIRE: now regenerate the PRIMARY, which nobody holds.
    provider.regenerate_primary()
    ok, total = traffic()
    phases.append((RotationPhase.RETIRE, f"{ok}/{total} OK"))

    for phase, result in phases:
        show(phase.value, result)

    show("total rejected requests across the whole rotation",
         provider.calls_rejected)

    print("""
    Zero failures. The procedure, as a runbook:

      1. Regenerate the SECONDARY key. No client is using it. Zero impact.
      2. Write it to the vault as the new current version.
      3. Roll instances onto it — a normal rolling restart, or a live
         re-read if you have one (PART 3).
      4. VERIFY every instance is on the new key BEFORE step 5. This is the
         step that gets skipped under time pressure, and skipping it turns a
         zero-downtime rotation into PART 1.
      5. Regenerate the PRIMARY. Now nobody holds it, so it is also zero
         impact — and it closes the window in which the old key still works.

    Step 5 matters for security, not availability: until you do it, the
    previously-exposed key is still valid. A rotation that stops at step 4 has
    changed which key is in use without revoking the old one.""")


# ---------------------------------------------------------------------------
# PART 3 — a cache that can be invalidated
# ---------------------------------------------------------------------------

@dataclass
class SecretCache:
    """A TTL cache with EXPLICIT invalidation on auth failure.

    THE TTL ALONE IS NOT ENOUGH. A 15-minute TTL means up to 15 minutes of
    401s after a rotation. The invalidate-on-401 path is what makes recovery
    fast; the TTL is just a backstop for a rotation nobody told you about.

    THE OTHER HALF, easy to miss: the refresh must be SINGLE-FLIGHT. Without
    a lock, N concurrent requests all see the expired entry and all call the
    vault — a stampede against a rate-limited service at exactly the moment
    it is on the critical path. The lock below is not optional.
    """

    vault: FakeKeyVault
    ttl: float = 0.2
    fetch_latency: float = 0.01   # a vault read is a NETWORK call
    _value: str | None = None
    _version: str | None = None
    _fetched_at: float = 0.0
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    fetches: int = 0
    invalidations: int = 0

    async def get(self, name: str) -> str:
        if self._value is not None and time.monotonic() - self._fetched_at < self.ttl:
            return self._value
        async with self._lock:
            # Re-check inside the lock: another task may have refreshed while
            # we waited. Without this the lock serialises the stampede instead
            # of preventing it.
            if self._value is not None and time.monotonic() - self._fetched_at < self.ttl:
                return self._value
            # The await is what makes this a real demonstration: a vault read
            # is a network call, so other coroutines run during it. A
            # synchronous fake would complete before any sibling started and
            # would never show a stampede at all.
            await asyncio.sleep(self.fetch_latency)
            v = self.vault.get_secret(name)
            self.fetches += 1
            self._value, self._version = v.value, v.version
            self._fetched_at = time.monotonic()
            return self._value

    def invalidate(self) -> None:
        """Call this on a 401. Forces the next get() to re-read."""
        self.invalidations += 1
        self._value = None


async def part3_invalidation() -> None:
    banner("PART 3 — invalidate on 401, do not wait for the TTL")

    vault = FakeKeyVault()
    vault.set_secret("aoai-key", "key-v1")
    provider = DualKeyProvider(primary="key-v1", secondary="key-v1")
    cache = SecretCache(vault, ttl=10.0)     # deliberately long TTL

    async def call_provider() -> str:
        """One request, with the standard 401 handling: refresh ONCE."""
        key = await cache.get("aoai-key")
        try:
            return provider.call(key)
        except PermissionError:
            # A 401 means our cached secret may be stale. Invalidate, re-read,
            # retry EXACTLY ONCE. More than once and a genuine permission
            # problem becomes an infinite loop against the vault.
            cache.invalidate()
            key = await cache.get("aoai-key")
            return provider.call(key)

    section("steady state")
    for _ in range(5):
        await call_provider()
    show("vault fetches", cache.fetches)
    show("provider rejections", provider.calls_rejected)

    section("secret rotated out from under us")
    vault.set_secret("aoai-key", "key-v2")
    provider.primary = "key-v2"
    provider.secondary = "key-v2"

    results = [await call_provider() for _ in range(5)]
    show("requests succeeded", f"{results.count('200 OK')}/5")
    show("cache invalidations", cache.invalidations)
    show("vault fetches", cache.fetches)
    show("provider rejections (the one that triggered recovery)",
         provider.calls_rejected)

    print("""
    ONE request saw a 401 and recovered inside the same request. Every
    subsequent request used the new value. With a TTL-only cache and a
    10-second TTL, all five would have failed and recovery would have waited
    for the clock.

    THE COST TO BE HONEST ABOUT: `invalidate()` on every 401 is a
    vault-stampede vector. If your credentials are genuinely revoked, every
    request 401s, every request invalidates, and every request hits the vault
    — turning a permission problem into a vault-throttling incident on top.
    Rate-limit the invalidation itself: at most one refresh per N seconds,
    regardless of how many 401s arrive.""")


# ---------------------------------------------------------------------------
# PART 4 — single-flight, measured
# ---------------------------------------------------------------------------

async def part4_stampede() -> None:
    banner("PART 4 — the refresh stampede")

    # NOTE ON THE FIRST VERSION OF THIS DEMO, because the mistake is
    # instructive: it used ttl=0.0 and reported 50 fetches BOTH with and
    # without the lock — failing to demonstrate its own point.
    #
    # Why: with ttl=0.0 the re-check inside the lock also misses, so each
    # waiter fetches in turn. The lock serialised the stampede instead of
    # collapsing it. A single-flight cache only collapses a burst if the
    # refreshed value is still fresh when the next waiter acquires the lock —
    # i.e. the TTL must be longer than the fetch takes.
    #
    # That is not a quirk of the demo. It is the real tuning constraint: a TTL
    # shorter than your vault latency gives you a permanent stampede.

    async def measure(single_flight: bool, ttl: float) -> int:
        vault = FakeKeyVault()
        vault.set_secret("k", "v1")
        cache = SecretCache(vault, ttl=ttl)
        if not single_flight:
            class NoLock:
                async def __aenter__(self) -> None: ...
                async def __aexit__(self, *a: object) -> None: ...
            cache._lock = NoLock()  # type: ignore[assignment]
        await asyncio.gather(*(cache.get("k") for _ in range(50)))
        return cache.fetches

    section("a realistic TTL (30s), 50 concurrent cold-start requests")
    show("WITHOUT single-flight", await measure(False, ttl=30.0))
    show("WITH single-flight", await measure(True, ttl=30.0))

    section("TTL of zero — the pathological case")
    show("WITHOUT single-flight", await measure(False, ttl=0.0))
    show("WITH single-flight", await measure(True, ttl=0.0))

    print("""
    FIRST PAIR: the lock collapses 50 concurrent cold-start misses into ONE
    vault call. That is the cold-start stampede — every replica in a rolling
    deploy hitting the vault at once — and it is the case that actually
    causes 429s in production.

    SECOND PAIR: with ttl=0.0 the lock buys nothing, because the value is
    stale again before the next waiter acquires it. Serialised, not collapsed.

    THE TUNING RULE this exposes: your TTL must comfortably exceed your vault
    fetch latency, or single-flight degrades to serialisation and you get the
    stampede anyway — just more slowly.""")


# ---------------------------------------------------------------------------
# PART 5 — the rotation checklist
# ---------------------------------------------------------------------------

def part5_checklist() -> None:
    banner("PART 5 — what makes a secret rotatable")

    print("""
    BEFORE you can rotate anything, these must be true:

    [ ] The provider supports TWO valid credentials simultaneously.
        Azure OpenAI, Storage, Cosmos, Service Bus, and Event Hubs all do
        (Key1/Key2). A third-party API with a single key does NOT — for those
        you need a maintenance window, or a proxy that can hold both.

    [ ] Every consumer is INVENTORIED. The rotation fails on the consumer
        nobody remembered: a Logic App, a Power BI dataset, a Function, a
        partner integration, someone's notebook. Tag secrets with their
        consumers and keep it current, because you cannot verify step 4 of the
        runbook against a list you do not have.

    [ ] Consumers can pick up a new value without a deploy — either a live
        re-read (PART 3) or a rolling restart you can trigger on demand.

    [ ] There is a 401 path that invalidates the cache and retries once.

    [ ] Rotation is REHEARSED in a lower environment. A runbook that has
        never been executed is a document, not a procedure — and the first
        execution will be during an incident.

    [ ] Expiry is MONITORED. Alert at 30 days, not on the expiry date.
        Client secrets in Entra expire on a fixed date; the outage arrives
        with no warning and no error that mentions expiry.

    THE BEST ANSWER IS TO NOT HAVE THE SECRET. Everything above is work you do
    not do if the workload uses managed identity — the platform rotates the
    underlying credential and your code never sees it. Reserve stored secrets
    for the things that genuinely cannot use it: third-party APIs, legacy
    systems, and partner integrations.

    BREAK-GLASS, briefly: keep one documented path to rotate a secret when
    your normal automation is the thing that is broken. It must be
    human-executable, tested, and audited — and it must not depend on the system
    it is meant to recover.""")


async def main() -> None:
    part1_naive()
    part2_dual_key()
    await part3_invalidation()
    await part4_stampede()
    part5_checklist()

    banner("SUMMARY")
    print("""
  * Naive rotation = 100% failure until every instance restarts.
  * Dual-key rotation: regenerate SECONDARY, migrate, VERIFY, then regenerate
    PRIMARY. Zero failed requests, and step 5 is what actually revokes the old
    key.
  * A TTL cache alone means up to one TTL of 401s. Invalidate on 401 and
    retry exactly once.
  * Rate-limit the invalidation, or revoked credentials turn into a vault
    stampede.
  * Refresh must be single-flight, with a re-check inside the lock.
  * Inventory every consumer; rehearse the runbook; alert 30 days before
    expiry.
  * Managed identity removes the whole problem. Reserve stored secrets for
    systems that cannot use it.
""")


if __name__ == "__main__":
    asyncio.run(main())
