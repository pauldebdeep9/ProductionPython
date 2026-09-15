"""
03 — Fail fast: validate at startup, not on the first request that hurts.

THE ARGUMENT
------------
A service that starts successfully and then fails on a user's request has
converted a deployment-time problem into a runtime incident. The failure is
now:
    * observed by a user rather than by a pipeline
    * reported through your error-handling path, which is worse than your
      startup path at explaining configuration problems
    * potentially rare — if the broken setting is only read by one code path,
      it may not surface for hours
    * mixed in with real errors in your dashboards

Whereas a service that refuses to start is caught by the deployment, rolls
back automatically, and never takes traffic.

THE DISTINCTION THAT MATTERS
----------------------------
There are two kinds of startup check and they belong in different places:

    SCHEMA / SHAPE      "is the config well-formed?"
                        Pure, fast, no network. ALWAYS at startup, always
                        fatal.

    CONNECTIVITY        "can I actually reach the thing?"
                        Network, slow, can fail transiently. At startup for
                        hard dependencies; as a READINESS probe for the rest.

Conflating them produces one of two bad outcomes: a service that will not
start because a non-critical dependency is briefly down, or a service that
starts happily with a malformed endpoint.

Run:  python 03_fail_fast.py
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum

from config_lab import banner, section, show

# ---------------------------------------------------------------------------
# PART 1 — the cost of lazy validation
# ---------------------------------------------------------------------------

def part1_lazy_cost() -> None:
    banner("PART 1 — what lazy validation actually costs")

    print("""
    A malformed `ISC_AI__LIMITS__PER_REQUEST_TOKEN_CAP` in a config used only
    by the summarisation path:

      LAZY                                    FAIL-FAST
      ----                                    ---------
      t+0    deploy succeeds                  t+0    deploy fails
      t+0    health check green               t+0    rollback triggered
      t+0    takes production traffic         t+0    previous version serving
      t+3h   first summarisation request      t+2m   engineer sees the error
      t+3h   500 to a user                           with the variable name
      t+3h   alert fires
      t+3h20 someone reads the traceback
      t+3h35 correlate to this morning's deploy
      t+3h50 rollback
      -----------------------------------------------
      3h50m and one user-visible outage       2 minutes, zero impact

    The deploy pipeline is the cheapest place to fail. Every layer past it
    multiplies the cost of the same mistake.

    THE COROLLARY people miss: this only works if the platform ACTS on a
    failed start. An App Service that restarts a crashing container forever,
    or a Kubernetes deployment with no `maxUnavailable` and no readiness gate,
    will happily replace a working version with a crash-looping one. Fail-fast
    is a contract between your process and your orchestrator; check both
    halves.""")


# ---------------------------------------------------------------------------
# PART 2 — the check taxonomy
# ---------------------------------------------------------------------------

class CheckKind(Enum):
    SCHEMA = "schema"            # pure, fast, deterministic
    CONNECTIVITY = "connectivity"  # network, slow, can flake
    PERMISSION = "permission"    # network + authorization
    CAPABILITY = "capability"    # "does this deployment actually exist?"


class Severity(Enum):
    FATAL = "fatal"          # do not start
    DEGRADED = "degraded"    # start, but report not-ready
    WARNING = "warning"      # start, log it


@dataclass
class CheckResult:
    name: str
    kind: CheckKind
    severity: Severity
    ok: bool
    detail: str = ""
    elapsed_ms: float = 0.0

    def line(self) -> str:
        mark = "ok  " if self.ok else ("FAIL" if self.severity is Severity.FATAL
                                       else "warn")
        return (f"{mark}  {self.name:<32} {self.kind.value:<13} "
                f"{self.elapsed_ms:>6.1f}ms  {self.detail}")


@dataclass
class StartupValidator:
    """Runs every check, collects ALL results, then decides.

    THE KEY DESIGN CHOICE: do not abort on the first failure. An operator
    fixing config wants the complete list, exactly as in script 02's
    all-errors-at-once. Aborting on the first failure recreates the
    one-problem-per-deploy loop that fail-fast was supposed to eliminate.
    """

    results: list[CheckResult] = field(default_factory=list)

    async def run(self, name: str, kind: CheckKind, severity: Severity,
                  fn, timeout_s: float = 5.0) -> CheckResult:
        t0 = time.perf_counter()
        try:
            async with asyncio.timeout(timeout_s):
                detail = await fn()
            r = CheckResult(name, kind, severity, True, detail or "",
                            (time.perf_counter() - t0) * 1000)
        except TimeoutError:
            r = CheckResult(name, kind, severity, False,
                            f"timed out after {timeout_s}s",
                            (time.perf_counter() - t0) * 1000)
        except Exception as e:
            r = CheckResult(name, kind, severity, False,
                            f"{type(e).__name__}: {e}",
                            (time.perf_counter() - t0) * 1000)
        self.results.append(r)
        return r

    @property
    def fatal_failures(self) -> list[CheckResult]:
        return [r for r in self.results
                if not r.ok and r.severity is Severity.FATAL]

    @property
    def degraded(self) -> list[CheckResult]:
        return [r for r in self.results
                if not r.ok and r.severity is Severity.DEGRADED]

    def report(self) -> str:
        return "\n".join(f"      {r.line()}" for r in self.results)


# ---------------------------------------------------------------------------
# PART 3 — a realistic startup sequence
# ---------------------------------------------------------------------------

@dataclass
class FakeWorld:
    """Controls which dependencies are healthy, so scenarios are scriptable."""

    openai_reachable: bool = True
    deployment_exists: bool = True
    search_reachable: bool = True
    search_index_exists: bool = True
    keyvault_reachable: bool = True
    telemetry_reachable: bool = True
    has_search_read_role: bool = True


async def build_checks(v: StartupValidator, w: FakeWorld) -> None:
    """The checks a RAG service should run before accepting traffic.

    Note the severity assignments — each one is an argument.
    """

    # --- SCHEMA: pure, instant, always fatal -------------------------------
    async def schema() -> str:
        # In the real service this is just `AppSettings()` — pydantic-settings
        # has already done the work (script 02). The check exists so that a
        # config problem appears in the SAME report as everything else, rather
        # than as a separate traceback before the report is even built.
        return "settings parsed and validated"

    await v.run("config.schema", CheckKind.SCHEMA, Severity.FATAL, schema)

    # --- CONNECTIVITY to the model endpoint: FATAL ------------------------
    # Fatal because the service does literally nothing without it. There is no
    # degraded mode worth serving.
    async def openai_conn() -> str:
        await asyncio.sleep(0.01)
        if not w.openai_reachable:
            raise ConnectionError("could not reach isc-aoai-prod.openai.azure.com")
        return "reachable"

    await v.run("azure_openai.connectivity", CheckKind.CONNECTIVITY,
                Severity.FATAL, openai_conn)

    # --- CAPABILITY: does the DEPLOYMENT exist? FATAL ---------------------
    # THIS IS THE CHECK PEOPLE SKIP, and it catches the single most common
    # Azure OpenAI misconfiguration: a correct endpoint with a deployment name
    # that does not exist in that resource. Connectivity passes; every request
    # 404s. One HEAD request at startup turns a mystery into a startup error.
    async def deployment_exists() -> str:
        await asyncio.sleep(0.01)
        if not w.deployment_exists:
            raise LookupError(
                "deployment 'gpt-4o-mini-prod' not found on this resource "
                "(a deployment name is not a model name)"
            )
        return "deployment 'gpt-4o-mini-prod' found"

    await v.run("azure_openai.deployment", CheckKind.CAPABILITY,
                Severity.FATAL, deployment_exists)

    # --- CONNECTIVITY to search: FATAL ------------------------------------
    async def search_conn() -> str:
        await asyncio.sleep(0.01)
        if not w.search_reachable:
            raise ConnectionError("could not reach isc-search.search.windows.net")
        return "reachable"

    await v.run("search.connectivity", CheckKind.CONNECTIVITY,
                Severity.FATAL, search_conn)

    # --- CAPABILITY: does the INDEX exist? FATAL --------------------------
    async def index_exists() -> str:
        await asyncio.sleep(0.01)
        if not w.search_index_exists:
            raise LookupError("index 'isc-docs-v3' does not exist")
        return "index 'isc-docs-v3' found, 48,201 documents"

    await v.run("search.index", CheckKind.CAPABILITY, Severity.FATAL,
                index_exists)

    # --- PERMISSION: can we actually READ? FATAL --------------------------
    # Reachability is not authorization. A managed identity with no role
    # assignment reaches the endpoint fine and 403s on every query. Checking
    # the permission at startup catches the missing RBAC assignment — which is
    # the most common failure when moving from a key to managed identity.
    async def search_permission() -> str:
        await asyncio.sleep(0.01)
        if not w.has_search_read_role:
            raise PermissionError(
                "403 on a probe query: the managed identity is missing the "
                "'Search Index Data Reader' role on isc-docs-v3"
            )
        return "probe query returned 200"

    await v.run("search.permission", CheckKind.PERMISSION, Severity.FATAL,
                search_permission)

    # --- Key Vault: FATAL only if we actually need it ---------------------
    async def keyvault() -> str:
        await asyncio.sleep(0.01)
        if not w.keyvault_reachable:
            raise ConnectionError("kv-isc-prod.vault.azure.net unreachable")
        return "reachable, 4 secrets resolved"

    await v.run("keyvault.connectivity", CheckKind.CONNECTIVITY,
                Severity.FATAL, keyvault)

    # --- Telemetry: WARNING, not fatal ------------------------------------
    # THE JUDGEMENT: a service that refuses to start because its metrics sink
    # is down has made observability a single point of failure for
    # availability. Log loudly, start anyway. This is the fail-OPEN side of
    # the fail-open/fail-closed decision from the failure-handling tutorial:
    # telemetry only makes the result better, so its absence degrades rather
    # than invalidates.
    async def telemetry() -> str:
        await asyncio.sleep(0.01)
        if not w.telemetry_reachable:
            raise ConnectionError("Application Insights ingestion unreachable")
        return "connected"

    await v.run("telemetry.connectivity", CheckKind.CONNECTIVITY,
                Severity.WARNING, telemetry)


async def part3_startup_sequence() -> None:
    banner("PART 3 — a startup sequence, three scenarios")

    scenarios = [
        ("everything healthy", FakeWorld()),
        ("telemetry down (start anyway)", FakeWorld(telemetry_reachable=False)),
        ("deployment name wrong", FakeWorld(deployment_exists=False)),
        ("managed identity missing a role",
         FakeWorld(has_search_read_role=False)),
    ]

    for label, world in scenarios:
        section(label)
        v = StartupValidator()
        await build_checks(v, world)
        print(v.report())
        if v.fatal_failures:
            print(f"\n      STARTUP ABORTED — {len(v.fatal_failures)} fatal "
                  f"check(s) failed:")
            for r in v.fatal_failures:
                print(f"        {r.name}: {r.detail}")
        else:
            warnings = [r for r in v.results
                        if not r.ok and r.severity is Severity.WARNING]
            print(f"\n      STARTED ({len(warnings)} warning(s))")


# ---------------------------------------------------------------------------
# PART 4 — liveness vs readiness vs startup
# ---------------------------------------------------------------------------

def part4_probes() -> None:
    banner("PART 4 — three probes, three questions")

    print("""
    Kubernetes names three probes and they answer genuinely different
    questions. Getting them backwards is a common and expensive mistake.

    STARTUP PROBE   "has initialisation finished?"
      Runs first; disables the other two until it passes. Use it when startup
      is slow — loading a local model, warming a cache, running migrations.
      Give it a generous failureThreshold. Without it, a slow-starting
      container is killed by the liveness probe and never starts at all,
      producing a crash loop that looks like a code bug.

    LIVENESS PROBE  "is this process wedged?"
      Failing it RESTARTS the container. So it must check ONLY things a
      restart can fix: a deadlocked event loop, an exhausted thread pool.
      THE CLASSIC MISTAKE is checking dependencies here. If your liveness
      probe calls Azure OpenAI and Azure OpenAI has a bad ten minutes, every
      replica restarts simultaneously — you have converted a partial
      dependency outage into a total self-inflicted one.
      Keep it trivial: return 200 if the event loop can schedule a callback.

    READINESS PROBE "should I receive traffic right now?"
      Failing it removes the pod from the load balancer WITHOUT restarting.
      This is where dependency checks belong. A pod that cannot reach search
      should stop receiving requests but keep running, so it can rejoin when
      search recovers.

    THE RULE, in one line: liveness checks the PROCESS, readiness checks its
    DEPENDENCIES, startup checks INITIALISATION.

    AND THE ONE-WAY DOOR: startup validation is fatal and permanent; readiness
    is continuous and reversible. A dependency that might recover belongs in
    readiness. A configuration value that is simply wrong belongs in startup —
    it will never become right on its own.""")

    section("what each probe should touch")
    table = [
        ("event loop responsive", "liveness", "a restart fixes a wedged loop"),
        ("config parsed", "startup", "cannot self-heal; fail permanently"),
        ("deployment exists", "startup", "cannot self-heal"),
        ("search reachable", "readiness", "may recover; do not restart"),
        ("token acquisition works", "readiness", "may recover"),
        ("Key Vault reachable", "startup + readiness", "needed to boot AND ongoing"),
        ("telemetry reachable", "neither", "log it; never gate traffic on it"),
        ("warm cache populated", "startup", "one-time initialisation"),
    ]
    print(f"      {'check':<28} {'probe':<20} why")
    print(f"      {'-' * 28} {'-' * 20} {'-' * 30}")
    for check, probe, why in table:
        print(f"      {check:<28} {probe:<20} {why}")


# ---------------------------------------------------------------------------
# PART 5 — what NOT to do at startup
# ---------------------------------------------------------------------------

async def part5_antipatterns() -> None:
    banner("PART 5 — startup checks that make things worse")

    print("""
    1. NO TIMEOUT ON A STARTUP CHECK
       An unreachable endpoint that black-holes packets hangs the check
       forever. The container never becomes ready, the orchestrator waits,
       and the deployment stalls with no error. EVERY check needs a timeout —
       the StartupValidator above enforces one.

    2. A CHECK THAT COSTS MONEY
       "Verify the model works" implemented as a real completion, on every
       replica, on every restart, forever. At 50 replicas cycling during a
       rollout that is 50 billable calls that tell you nothing a metadata
       request would not. Probe with the cheapest call that proves the thing
       you care about.

    3. A CHECK THAT WRITES
       Verifying the index by writing a test document leaves debris, and does
       so under whatever identity the service runs as. Probe with reads.

    4. SERIAL CHECKS
       Eight checks at 2s each is 16 seconds added to every pod start, which
       during a rolling deploy is minutes of reduced capacity. Run them
       CONCURRENTLY — they are independent.

    5. CHECKING EVERYTHING
       Every optional integration verified at startup means any one of them
       being down blocks your deploy. Be deliberate about which failures are
       genuinely fatal; the rest belong in readiness.

    6. NO CACHE OF THE RESULT
       A readiness probe that runs a full dependency sweep every 5 seconds
       generates real load against your dependencies — and at 50 replicas
       that is 600 requests/minute of pure probe traffic. Cache the result
       for a few seconds.""")

    section("serial vs concurrent, measured")

    async def slow_check(name: str) -> str:
        await asyncio.sleep(0.08)
        return "ok"

    names = [f"dep-{i}" for i in range(8)]

    t0 = time.perf_counter()
    for n in names:
        await slow_check(n)
    serial = time.perf_counter() - t0

    t0 = time.perf_counter()
    await asyncio.gather(*(slow_check(n) for n in names))
    concurrent = time.perf_counter() - t0

    show("8 checks, serial", f"{serial * 1000:.0f}ms")
    show("8 checks, concurrent", f"{concurrent * 1000:.0f}ms")
    show("added to every pod start (serial)", f"{serial * 1000:.0f}ms")
    print(f"      Across a 50-pod rolling deploy that is "
          f"{serial * 50:.1f}s vs {concurrent * 50:.1f}s of extra startup time.")


# ---------------------------------------------------------------------------
# PART 6 — config drift and the config endpoint
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ConfigFingerprint:
    """A hash of the effective configuration, computed at startup.

    WHY: "staging and production behave differently" is usually a config
    difference nobody can find. A fingerprint plus a redacted dump makes the
    comparison mechanical rather than archaeological.

    THE RULE FOR THE DUMP: it must be REDACTED, and the redaction must be
    structural (SecretStr, script 04) rather than a denylist of key names.
    A denylist misses `AZURE_STORAGE_CONNECTION_STRING` the day someone adds
    it under a name your regex did not anticipate.
    """

    fingerprint: str
    environment: str
    values: dict[str, object]

    @classmethod
    def compute(cls, values: dict[str, object], environment: str) -> ConfigFingerprint:
        import hashlib
        import json as _json
        canonical = _json.dumps(values, sort_keys=True, default=str)
        digest = hashlib.sha256(canonical.encode()).hexdigest()[:12]
        return cls(digest, environment, values)


def part6_drift() -> None:
    banner("PART 6 — config fingerprints and the diagnostic endpoint")

    staging = {"top_k": 8, "rerank": True, "deployment": "gpt-4o-mini",
               "max_tokens": 512, "streaming": True, "temperature": 0.0}
    production = {"top_k": 8, "rerank": False, "deployment": "gpt-4o-mini",
                  "max_tokens": 512, "streaming": True, "temperature": 0.0}

    f1 = ConfigFingerprint.compute(staging, "staging")
    f2 = ConfigFingerprint.compute(production, "production")
    show("staging fingerprint", f1.fingerprint)
    show("production fingerprint", f2.fingerprint)

    diff = {k: (staging.get(k), production.get(k))
            for k in staging.keys() | production.keys()
            if staging.get(k) != production.get(k)}
    show("differences", diff)

    print("""
    Log the fingerprint at startup, one line, at INFO. Then:
      * "did this deploy change config?" is a log comparison
      * "do staging and production match?" is comparing two strings
      * a config change with no code change is visible in your logs

    SHIP A DIAGNOSTIC ENDPOINT — `GET /diagnostics/config` returning the
    redacted effective config plus the provenance from script 01. Guard it
    behind admin authz. It replaces the "add a print statement and redeploy"
    loop entirely.

    AND FINGERPRINT THE PROMPT TOO. In an LLM system the prompt template is
    configuration in every meaningful sense — it changes behaviour, it is
    deployed, and it needs to be diffable. A prompt hash in the same startup
    line means "the answers changed and nobody deployed code" becomes an
    answerable question. See script 07.""")


async def main() -> None:
    part1_lazy_cost()
    banner("PART 2 — the check taxonomy (see source)")
    print("""    SCHEMA        pure, fast, no network      -> always startup, fatal
    CONNECTIVITY  network, can flake          -> startup if hard dependency
    PERMISSION    network + authorization     -> startup; catches missing RBAC
    CAPABILITY    "does the thing exist?"     -> startup; catches wrong names""")
    await part3_startup_sequence()
    part4_probes()
    await part5_antipatterns()
    part6_drift()

    banner("SUMMARY")
    print("""
  * Failing in the pipeline costs minutes; failing at runtime costs hours and
    a user-visible outage.
  * Collect ALL check results before deciding — one report, not one problem
    per deploy.
  * Separate SCHEMA (pure, always fatal) from CONNECTIVITY (network, may
    recover, often readiness).
  * Check CAPABILITY and PERMISSION, not just reachability. A valid endpoint
    with a wrong deployment name, or a managed identity with no role
    assignment, both pass a connectivity check and fail every request.
  * Liveness checks the PROCESS; readiness checks DEPENDENCIES; startup checks
    INITIALISATION. Never call a dependency from a liveness probe.
  * Every check needs a timeout. Run them concurrently. Never probe with a
    call that costs money or writes data.
  * Fingerprint the effective config (and the prompt) at startup, and ship a
    redacted diagnostics endpoint.
""")


if __name__ == "__main__":
    asyncio.run(main())
