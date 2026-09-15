# Configuration and Secrets — Production Python, GenAI Focus

Eleven files, ~4,900 lines. Runnable offline; needs `pydantic`,
`pydantic-settings`, `pytest`.

**Verified state:** 9/9 scripts run clean · `ruff` passes with the **bandit
(`S`) security rules enabled** · 36/36 tests pass · the suite is
**mutation-tested** (removing `SecretStr` and the production guard breaks 4
tests) · the capstone runs a leak scanner over 6 emitted surfaces and asserts
zero.

---

## Files

| File | Topic | The thing worth taking away |
|---|---|---|
| `config_lab.py` | Harness | A **leak detector** that makes "we don't log secrets" testable |
| `01_config_layers.py` | Layering and precedence | Track **provenance**; ship an `explain` command |
| `02_pydantic_settings.py` | One settings class | All errors at once, with the variable name |
| `03_fail_fast.py` | Startup validation | Check **capability** and **permission**, not just reachability |
| `04_secrets.py` | Seven leak surfaces | Defend **structurally**, not with discipline |
| `05_credentials.py` | Managed identity | One credential object — measured **100x** amplification otherwise |
| `06_rotation.py` | Zero-downtime rotation | Dual-key window + invalidate on 401 |
| `07_llm_config.py` | GenAI-specific | Prompts are config; cost arithmetic at startup |
| `08_capstone.py` | Everything composed | Ends by auditing itself for leaks |
| `09_test_config.py` | Testing config | Env isolation, and a negative control for the detector |

Run any file: `python 04_secrets.py`. Test: `pytest 09_test_config.py -v`.

---

## Suggested path

**Two hours** — 02, 04, 03. The settings class, the leak surfaces, and
fail-fast. That's most of the practical value.

**Reviewing someone's config PR** — the checklist below, then 01 part 5 (traps)
and 04.

**Moving from API keys to managed identity** — 05, then 06.

---

## The premise

"Don't hardcode secrets" is solved. The leaks that reach production logs are all
the same shape: **some code path formatted an object**.

| Surface | How it happens |
|---|---|
| `repr` / `str` | Debugger, pytest assertion, `print()`, traceback rendering |
| Exception messages | Connection string interpolated into the error |
| Logs | `log.info("config: %s", cfg)` at startup |
| Serialisation | `json.dumps(asdict(config))` to a diagnostics endpoint |
| URLs | Access logs, proxies, `Referer`, APM `http.url`, screenshots |
| `subprocess` argv | `/proc/<pid>/cmdline`, `ps aux` |
| Telemetry | Span attributes shipped to a workspace with a larger reader set |

You cannot audit hundreds of format calls. So the defence is structural: give
the secret a type that refuses to render itself. `SecretStr` makes the careless
path safe, and `.get_secret_value()` is the one hole — which is greppable.

---

## Measured results

**Denylist redaction misses 3 of 4** (`04`). Redacting by key name catches
`api_key` and misses `sas_url`, `conn`, and `bootstrap_servers` — none of whose
names contains a suspicious word. Redact by **type**, at declaration.

**100x credential amplification** (`05`). One credential object: 100
`get_token()` calls → **1** call to the identity endpoint (99 cache hits). A new
credential per request: **100** calls. IMDS is rate-limited, so this surfaces as
authentication failures that look like an identity problem.

**Single-flight collapses a 50-way stampede to 1** (`06`). 50 concurrent
cold-start requests without the lock: 50 vault fetches. With it: 1.

**Zero failed requests through a dual-key rotation** (`06`), versus 100% failure
for the naive version.

**Cost arithmetic catches a 17x bill increase** (`07`). Switching
`ISC__OPENAI__MODEL` from `gpt-4o-mini` to `gpt-4o` — one environment variable,
no code change, no error — moves worst-case daily spend from **$15.07 to
$251.20**. The startup validator rejects it with the arithmetic in the message.

**Seven config errors reported at once** (`02`), each naming its environment
variable. Each deploy round trip avoided is 5–20 minutes.

**The capstone audits itself**: 6 emitted surfaces, 0 leaks. All 8 planted
misconfigurations rejected at startup.

---

## Two things that cost me a debugging round

**`NoDecode` (`02`).** pydantic-settings treats any `list`/`dict` field as
complex and runs `json.loads()` on the raw env value **inside the source** —
before any `field_validator(mode="before")` runs. A comma-separated value fails
with a `JSONDecodeError` that points at JSON parsing and gives no hint your
validator exists. `mode="before"` is not early enough. The fix is
`Annotated[list[str], NoDecode]`.

**Counting the wrong link (`05`).** My first version of the caching demo counted
`links[2]` (ManagedIdentity) and reported 0 calls in both cases, because
WorkloadIdentity sits ahead of it and answers first. The demo appeared to prove
the opposite of what it proves. Same class of mistake as attributing loop-lag to
the wrong phase in the async tutorial: measure what actually happened.

A third, smaller one: the stampede demo in `06` initially used `ttl=0.0` and
reported 50 fetches both with and without the lock. With a zero TTL the re-check
inside the lock also misses, so the lock serialises the stampede instead of
collapsing it. That turned out to be a real tuning constraint worth keeping —
**your TTL must exceed your vault fetch latency** — so the script now shows both
cases.

---

## Decision tables

### Precedence (lowest to highest)

```
code defaults  <  config file  <  .env  <  environment  <  CLI args
```

CLI wins because an operator under pressure must be able to override anything
without a redeploy. Defaults sit lowest and **must be safe** — they are what
runs when every other layer fails to load.

### What goes where

| Layer | Contents |
|---|---|
| Code defaults | Timeouts, retries, chunk sizes, flags **off**, safe fallbacks |
| Config file (checked in) | Endpoints, deployment names, index names, quotas |
| Environment | Per-instance values, anything from Key Vault, platform vars |
| Secret store | Keys, connection strings, client secrets |
| **Not config at all** | Anything whose change should require a PR |

That last row: if a value changed by 20% and nobody noticed for a week, what
happens? *Slightly slower* → config. *Different business outcomes* → code, with
a test, behind a PR.

### Probes

| Probe | Question | Checks |
|---|---|---|
| Startup | Has init finished? | Config schema, deployment exists, warm cache |
| Liveness | Is the process wedged? | **Only** things a restart fixes |
| Readiness | Should I get traffic? | Dependencies that may recover |

Never call a dependency from a liveness probe — a bad ten minutes at Azure
OpenAI then restarts every replica simultaneously.

---

## Review checklist

**Layering**
- [ ] Every variable prefixed; `__` for nesting
- [ ] Never `bool()` on a string — `bool("false")` is `True`
- [ ] Empty string is **set**; `.get(k, default)` won't save you — use `min_length=1`
- [ ] Base + overlay, **deep** merged (a shallow merge silently deletes spend caps)
- [ ] Provenance tracked; an `explain` command exists

**Settings class**
- [ ] One class is the whole surface; no scattered `os.environ.get()`
- [ ] Not read at import time
- [ ] `extra="forbid"` on your namespace, `"ignore"` where you read the platform's
- [ ] Cross-field rules: no static key in prod, no DEBUG in prod, no empty `allowed_groups`
- [ ] Values with implied relationships checked against each other
- [ ] `.env.example` generated from the class and diffed in CI

**Startup**
- [ ] Checks run concurrently, each with a timeout
- [ ] **All** results collected before deciding
- [ ] Capability and permission checked, not just reachability
- [ ] Key Vault references validated as **resolved** (empty *and* literal forms)
- [ ] Config fingerprint logged

**Secrets**
- [ ] `SecretStr` at declaration; redaction by **type**, never by key name
- [ ] `.get_secret_value()` calls adjacent to the consuming client
- [ ] Never in a URL, never in argv
- [ ] Telemetry carries identifiers and counts only

**Credentials**
- [ ] One credential object per process
- [ ] Refresh margin ~5 min; 401 → forced refresh, retry **once**
- [ ] Explicit credential in production, not the fallback chain
- [ ] Least privilege by role: User not Contributor, Reader not Owner

**Rotation**
- [ ] Provider supports two valid credentials
- [ ] Consumers inventoried; runbook rehearsed
- [ ] Cache invalidates on 401, rate-limited; refresh single-flight
- [ ] Expiry alerted 30 days ahead

---

## Azure-specific notes

- **A deployment name is not a model name.** Check the deployment *exists* at
  startup — one HEAD request turns a mystery 404 into a startup error.
- **`model_version` can change under you** on auto-update deployments. Record it
  per response and alert on change; it explains "the answers got worse last
  Tuesday and nobody deployed."
- **Key Vault references fail silently** to `""` or the literal
  `@Microsoft.KeyVault(...)` string. Validate both.
- **Key Vault has two incompatible permission models** — access policies and
  RBAC. A vault in access-policy mode ignores your role assignments, and the 403
  says nothing about which is in use.
- **RBAC propagation is not instant.** A deploy that assigns a role then
  immediately validates permissions at startup will fail once and succeed on
  retry — which looks like flakiness and is a race in your IaC.
- **The chain falls back to the developer.** On a laptop with `az login` active,
  broken managed-identity config silently uses *your* credentials, which usually
  have more access than the workload. It works locally and fails only when
  deployed.

---

## Connections to your existing work

**Your permission-trimming invariant has a config half.** `allowed_groups` empty
means trimming is off, and it's one blank environment variable away. The
capstone makes that a startup-fatal rule in production, alongside "no DEBUG
logging in prod" — which exists because DEBUG moves retrieved chunk text into
Application Insights, a workspace whose readers are a different set from the
document's readers.

**The P2 disposition prompt should be fingerprinted.** You report results as
k/12 named scenarios; adding `prompt.version` and `prompt.fingerprint` to each
run means a change in results can be attributed to a prompt edit rather than to
model nondeterminism. Right now those two causes are indistinguishable in your
records — and since you've already established that temperature 0 isn't
deterministic, a fingerprint is what separates "the prompt changed" from "the
sample moved."

**The `{context}` guard is worth stealing directly.** `PromptTemplate.render()`
raising on a missing variable prevents the highest-severity silent failure in a
RAG system: a literal `{context}` reaching the model, which then answers
fluently from parametric knowledge with no grounding and no error. Six lines.

**Retrofit order:** the settings class first (it subsumes scattered
`os.environ.get()` and gives you all-errors-at-once), then `SecretStr` on any
credential field, then the leak-detector test. All three are additive. The
credential work only matters once you have a non-production tenant to test
managed identity against — which is still blocked on tenant access, so building
the settings class against synthetic config is the parallel-track move.
