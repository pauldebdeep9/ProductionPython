"""
05 — Credentials: managed identity, the credential chain, and Key Vault.

THE GOAL
--------
No secret in your configuration at all. Not in a file, not in an environment
variable, not in a vault your code reads. The workload proves its identity to
the platform and receives short-lived tokens.

That is what managed identity buys, and the reasons are not abstract:
    * nothing to rotate, because nothing is stored
    * nothing to leak, because there is no long-lived value
    * every access is attributable to a named identity in sign-in logs
    * access is revoked by removing a role assignment, taking effect in
      minutes, without a deploy

THE COST, stated honestly: it does not work on a laptop the same way it works
in Azure, so you need a credential CHAIN, and the chain is where the confusing
failures live. Most of this script is about the chain.

Run:  python 05_credentials.py
"""

from __future__ import annotations

import time

from config_lab import (
    CredentialFailed,
    FakeCredentialLink,
    FakeDefaultAzureCredential,
    FakeKeyVault,
    banner,
    section,
    show,
    verdict,
)

# Scopes. Note the `.default` suffix — it means "all the static permissions
# already granted to this identity for this resource". Omitting it is one of
# the most common causes of an AADSTS error that reads like a permissions
# problem when it is actually a malformed scope.
AOAI_SCOPE = "https://cognitiveservices.azure.com/.default"
SEARCH_SCOPE = "https://search.azure.com/.default"
VAULT_SCOPE = "https://vault.azure.net/.default"
STORAGE_SCOPE = "https://storage.azure.com/.default"


# ---------------------------------------------------------------------------
# PART 1 — the chain, and why order matters
# ---------------------------------------------------------------------------

def make_chain(*, in_azure: bool, env_creds: bool = False,
               cli_logged_in: bool = True) -> FakeDefaultAzureCredential:
    """Build a chain matching DefaultAzureCredential's real order."""
    return FakeDefaultAzureCredential([
        FakeCredentialLink("EnvironmentCredential", available=env_creds),
        FakeCredentialLink("WorkloadIdentityCredential", available=in_azure),
        FakeCredentialLink("ManagedIdentityCredential", available=in_azure,
                           latency=0.002),
        FakeCredentialLink("SharedTokenCacheCredential", available=False),
        FakeCredentialLink("AzureCliCredential",
                           available=cli_logged_in and not in_azure),
    ])


def part1_chain() -> None:
    banner("PART 1 — the same code, three environments")

    scenarios = [
        ("developer laptop (az login)", make_chain(in_azure=False)),
        ("CI with a service principal",
         make_chain(in_azure=False, env_creds=True, cli_logged_in=False)),
        ("production (managed identity)", make_chain(in_azure=True)),
    ]

    for label, cred in scenarios:
        token = cred.get_token(AOAI_SCOPE)
        section(label)
        for attempt in cred.chain_attempts:
            print(f"      {attempt}")
        show("token", token)

    print("""
    THE PROPERTY THAT MATTERS: the application code is identical in all three.
    `DefaultAzureCredential()` and `get_token(scope)`. No `if environment ==
    "local"` branch, which means the production credential path is exercised —
    in shape — on every developer's machine.

    Compare with the alternative: `if local: use API key, else: use managed
    identity`. That produces a production-only code path that is first
    exercised in production.""")


# ---------------------------------------------------------------------------
# PART 2 — token caching
# ---------------------------------------------------------------------------

def part2_caching() -> None:
    banner("PART 2 — caching, and the cost of getting it wrong")

    def endpoint_calls(c: FakeDefaultAzureCredential) -> int:
        """Total calls made to any identity-endpoint link.

        NOTE: count across ALL links, not `links[2]`. In this chain
        WorkloadIdentityCredential sits ahead of ManagedIdentityCredential and
        answers first, so counting one hardcoded index reports zero and makes
        the demo look like it proves the opposite of what it proves. Same
        class of bug as attributing loop-lag to the wrong phase — measure the
        thing that actually happened, not the thing you assumed would.
        """
        return sum(link.calls for link in c.links if link.available)

    section("ONE credential object, reused (correct)")
    cred = make_chain(in_azure=True)
    for _ in range(100):
        cred.get_token(AOAI_SCOPE)
    show("get_token() calls made", 100)
    show("actual identity-endpoint calls", endpoint_calls(cred))
    show("cache hits / misses", f"{cred.cache_hits} / {cred.cache_misses}")

    section("a NEW credential per request (the bug)")
    total = 0
    for _ in range(100):
        c = make_chain(in_azure=True)
        c.get_token(AOAI_SCOPE)
        total += endpoint_calls(c)
    show("get_token() calls made", 100)
    show("actual identity-endpoint calls", total)
    show("amplification", f"{total}x")

    print("""
    THE PRODUCTION SYMPTOM: the IMDS endpoint (169.254.169.254) is RATE
    LIMITED. Construct a credential per request under load and you get 429s
    from IMDS, which surface as authentication failures — an incident that
    looks like an identity problem and is actually a lifecycle bug.

    Same rule as the HTTP client in the async tutorial: construct it ONCE, in
    your application lifespan, and share it. `DefaultAzureCredential` is
    thread-safe and async-safe for this purpose.""")

    section("per-scope caching")
    cred2 = make_chain(in_azure=True)
    for scope in (AOAI_SCOPE, SEARCH_SCOPE, VAULT_SCOPE,
                  AOAI_SCOPE, SEARCH_SCOPE):
        cred2.get_token(scope)
    show("5 get_token calls across 3 scopes", "")
    show("cache hits / misses", f"{cred2.cache_hits} / {cred2.cache_misses}")
    print("      Tokens are cached PER SCOPE — a token for Azure OpenAI is")
    show("", "not valid for Search. Three misses, two hits.")


# ---------------------------------------------------------------------------
# PART 3 — refresh margin
# ---------------------------------------------------------------------------

def part3_refresh() -> None:
    banner("PART 3 — refresh BEFORE expiry, not at it")

    cred = FakeDefaultAzureCredential([
        FakeCredentialLink("ManagedIdentityCredential", lifetime=1.0),
    ])
    link = cred.links[0]

    section("with a refresh margin (correct)")
    cred.get_token(AOAI_SCOPE, refresh_margin=0.5)
    time.sleep(0.6)                       # 0.4s of life left, inside the margin
    cred.get_token(AOAI_SCOPE, refresh_margin=0.5)
    show("token acquisitions", link.calls)
    show("refreshed while still valid", "yes — no request ever saw a 401")

    section("without a margin (the bug)")
    cred2 = FakeDefaultAzureCredential([
        FakeCredentialLink("ManagedIdentityCredential", lifetime=1.0),
    ])
    cred2.get_token(AOAI_SCOPE, refresh_margin=0.0)
    time.sleep(0.6)
    cred2.get_token(AOAI_SCOPE, refresh_margin=0.0)
    show("token acquisitions", cred2.links[0].calls)
    print("""
      The second call reused a token with 0.4s of life left. By the time it
      reaches Azure OpenAI — after DNS, TLS, and any queueing — it may have
      expired, producing a 401 for a token that was valid when you checked.

      A 5-minute margin on a 1-hour token is the conventional choice: long
      enough to cover any plausible request latency and clock skew, short
      enough that you are not refreshing constantly.

    THE OTHER HALF: a 401 must still be handled. Treat it as retryable EXACTLY
    ONCE, with a forced token refresh before the retry — and never more than
    once, or a genuine permission problem becomes an infinite loop. This is
    the `ESCALATE` disposition from the failure-handling tutorial: a 401 that
    survives a refresh needs a human, not a backoff.""")


# ---------------------------------------------------------------------------
# PART 4 — the chain's failure modes
# ---------------------------------------------------------------------------

def part4_failures() -> None:
    banner("PART 4 — why chain failures are confusing")

    section("nothing available")
    cred = FakeDefaultAzureCredential([
        FakeCredentialLink("EnvironmentCredential", available=False),
        FakeCredentialLink("ManagedIdentityCredential", available=False),
        FakeCredentialLink("AzureCliCredential", available=False),
    ])
    try:
        cred.get_token(AOAI_SCOPE)
    except CredentialFailed as e:
        print(f"      {str(e)[:150]}")

    print("""
    THE REAL ERROR is much worse than this — a wall of text listing every link
    and its individual failure, in which the actual cause is one line. People
    read the first link's message, conclude the problem is EnvironmentCredential,
    and go looking for missing env vars when the real issue is a missing role
    assignment.

    HOW TO DEBUG IT, in order:
      1. `AZURE_LOG_LEVEL=debug` — logs which credential was tried and why
         each was skipped.
      2. `exclude_*_credential=True` to narrow the chain to the one you
         expect. In production, be EXPLICIT: use ManagedIdentityCredential
         directly rather than DefaultAzureCredential. A chain that can fall
         back is a chain that can run as the wrong principal.
      3. Check the token's claims. `oid`/`appid` tells you WHICH identity you
         actually got — the single most useful diagnostic, because "it
         authenticated but 403s" almost always means the wrong identity.

    THE SILENT-FALLBACK HAZARD, and this is the one worth designing against:
    on a developer machine with `az login` active, a broken managed-identity
    setup falls through to the developer's OWN credentials — which usually
    have MORE permission than the workload. The code works locally, is
    tested locally, and fails only when deployed. Worse, if it happens in a
    shared environment, the service is running as a person.""")

    section("user-assigned vs system-assigned")
    print("""
      SYSTEM-ASSIGNED: lifecycle tied to the resource. Deleted with it.
        Simple; good for a single service that owns its own identity.
        Problem: recreating the resource creates a NEW identity, so every
        role assignment must be redone. Painful in IaC teardown/rebuild.

      USER-ASSIGNED: a standalone resource, attachable to many services.
        Survives resource recreation, so role assignments persist. Shareable
        across a scale set or several apps that need the same access.
        REQUIRED CONFIG: when more than one identity is attached, you MUST
        specify which:

            ManagedIdentityCredential(client_id=settings.managed_identity_client_id)

        Omitting it with multiple identities attached produces an ambiguous
        request and a confusing failure. That client_id is a legitimate piece
        of NON-SECRET configuration — it is a GUID identifying an identity, not
        a credential, and it belongs in your settings class like any endpoint.""")


# ---------------------------------------------------------------------------
# PART 5 — Key Vault: two ways, and when each applies
# ---------------------------------------------------------------------------

def part5_keyvault() -> None:
    banner("PART 5 — Key Vault references vs SDK access")

    kv = FakeKeyVault("kv-isc-prod")
    kv.set_secret("sql-password", "Pw-PROD-REDACTED")
    kv.set_secret("partner-api-key", "pk-PARTNER-REDACTED")

    section("A) Key Vault REFERENCE — the platform resolves it")
    print(f"""
      In App Service / Container Apps / Functions, set an app setting to:

          SQL_PASSWORD=@Microsoft.KeyVault(SecretUri={kv.uri('sql-password')})

      The PLATFORM resolves it using the app's managed identity and injects
      the plain value as an ordinary environment variable. Your code reads
      `os.environ["SQL_PASSWORD"]` — or better, a settings field — and has no
      Key Vault dependency at all.

      ADVANTAGES:
        * no SDK, no extra latency, no vault throttling
        * no vault code path to test
        * works identically with a plain env var locally

      LIMITATIONS, and they are the reason B exists:
        * resolved at APP START (and on setting change). A rotated secret
          needs a restart — see script 06.
        * a resolution failure yields an EMPTY or literal-reference value
          rather than an error. This is the empty-string trap from script 01
          in its most dangerous form, and it is why `min_length=1` on the
          field matters: without it you get a client configured with "".""")

    section("B) SDK access — when you need versioning or rotation")
    print("""
      SecretClient(vault_url, credential=DefaultAzureCredential())

      Use it when you need to re-read without a restart, pin a version, or
      list secrets. The rules:
        * ONE client for the process, like every other client
        * CACHE with a TTL. Vault reads are rate limited; fetching per
          request WILL throttle under load.
        * handle 429 and 403 explicitly — 403 here usually means a missing
          RBAC role or an access-policy/RBAC mode mismatch.""")

    section("validating that a reference actually resolved")
    for label, value in [
        ("resolved correctly", "Pw-PROD-REDACTED"),
        ("resolution failed -> empty", ""),
        ("resolution failed -> literal", "@Microsoft.KeyVault(SecretUri=...)"),
    ]:
        looks_unresolved = (not value) or value.startswith("@Microsoft.KeyVault")
        verdict(not looks_unresolved,
                f"{label:<34} value={value[:30]!r}")

    print("""
      Both failure modes are detectable in one validator, and both should be
      FATAL at startup:

          @field_validator("sql_password")
          def must_be_resolved(cls, v: SecretStr) -> SecretStr:
              raw = v.get_secret_value()
              if not raw or raw.startswith("@Microsoft.KeyVault"):
                  raise ValueError(
                      "Key Vault reference did not resolve — check the app's "
                      "managed identity has 'Key Vault Secrets User' on the vault"
                  )
              return v

      Note the error message names the LIKELY CAUSE and the role. A message
      that says "invalid value" sends someone to the wrong place.""")


# ---------------------------------------------------------------------------
# PART 6 — RBAC and the least-privilege table
# ---------------------------------------------------------------------------

def part6_rbac() -> None:
    banner("PART 6 — role assignments are configuration too")

    print("""
    The role assignments a RAG service needs, and the ones people
    over-provision:

      Azure OpenAI          Cognitive Services OpenAI User
                            NOT Contributor. User can call the model;
                            Contributor can also DELETE the deployment.

      Azure AI Search       Search Index Data Reader   (query path)
                            Search Index Data Contributor (ingestion only)
                            Split these across two identities if the query
                            service and the ingestion job are separate — and
                            they should be.

      Key Vault             Key Vault Secrets User     (get/list)
                            NOT Key Vault Administrator, which can also
                            write and delete secrets.

      Storage               Storage Blob Data Reader
                            NOT Storage Blob Data Owner.

      App Insights          Monitoring Metrics Publisher

    TWO THINGS THAT CATCH PEOPLE:

    1. RBAC PROPAGATION IS NOT INSTANT. A new role assignment can take
       several minutes to take effect. A deployment that assigns a role and
       immediately starts a container that validates permissions at startup
       will fail on the first attempt and succeed on the retry — which looks
       like flakiness and is actually a race in your IaC. Either add a wait,
       or make the permission check a READINESS check rather than a startup
       one so the pod retries into readiness instead of crash-looping.

    2. KEY VAULT HAS TWO PERMISSION MODELS. Access policies (legacy) and RBAC.
       A vault in access-policy mode ignores your role assignments entirely,
       and vice versa. The resulting 403 says nothing about which model is
       in use. Check the vault's `enableRbacAuthorization` flag first — it
       will save an hour.

    AND THE CONFIGURATION ANGLE: role assignments belong in IaC, in the same
    repo, reviewed in the same PR as the code that needs them. A permission
    granted by hand in the portal is invisible, unreviewed, and gone the next
    time the environment is rebuilt.""")


def main() -> None:
    part1_chain()
    part2_caching()
    part3_refresh()
    part4_failures()
    part5_keyvault()
    part6_rbac()

    banner("SUMMARY")
    print("""
  * Managed identity means no stored secret: nothing to rotate, nothing to
    leak, every access attributable, revocation without a deploy.
  * DefaultAzureCredential keeps ONE code path across laptop, CI, and
    production. Construct it ONCE — per-request construction throttles IMDS.
  * Tokens cache PER SCOPE. Refresh with a ~5 minute margin, and handle a 401
    with exactly one forced-refresh retry.
  * In production prefer an EXPLICIT credential over the chain. A chain that
    can fall back can run as the wrong principal — and on a laptop it falls
    back to the developer, who has more access than the workload.
  * `client_id` for a user-assigned identity is non-secret config and belongs
    in your settings class.
  * Key Vault references are simpler (no SDK, no throttling) but resolve at
    start and fail SILENTLY to empty. Validate that they resolved.
  * Least privilege by role: User not Contributor, Reader not Owner, Secrets
    User not Administrator.
  * RBAC propagation is not instant; Key Vault has two incompatible
    permission models. Both cost an hour if you do not know them.
""")


if __name__ == "__main__":
    main()
