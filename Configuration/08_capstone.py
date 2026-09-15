"""
08 — Capstone: a fully configured, fail-fast, leak-free RAG service.

WHAT THIS COMPOSES
------------------
  01  layering + provenance    -> defaults < file < env, with `explain`
  02  pydantic-settings        -> one class, nested, cross-field rules
  03  fail-fast startup        -> schema + capability + permission checks
  04  SecretStr                -> structural redaction, verified by a scanner
  05  credential chain         -> one code path across laptop and production
  06  cache invalidation       -> rotation without a restart
  07  LLM config               -> prompt fingerprints, cost arithmetic

THE TEST THIS SCRIPT APPLIES TO ITSELF
--------------------------------------
At the end it runs the LeakDetector over everything the service emitted —
startup logs, the diagnostics endpoint, an exception traceback, and a span —
and asserts zero leaks. That assertion is the deliverable; the rest is setup.

Run:  python 08_capstone.py
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from decimal import Decimal
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    Field,
    HttpUrl,
    SecretStr,
    ValidationError,
    computed_field,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from config_lab import (
    FAKE_API_KEY,
    FakeCredentialLink,
    FakeDefaultAzureCredential,
    FakeKeyVault,
    LeakDetector,
    banner,
    clean_env,
    env,
    section,
    show,
    verdict,
)

logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                    format="%(message)s", force=True)
log = logging.getLogger("isc.capstone")

DETECTOR = LeakDetector()
DETECTOR.register("AOAI_KEY", FAKE_API_KEY)
DETECTOR.register("VAULT_SECRET", "Pw-VAULTED-a91f3c-DO-NOT-SHIP")


# ===========================================================================
# 1. SETTINGS
# ===========================================================================

class Environment(StrEnum):
    LOCAL = "local"
    STAGING = "staging"
    PRODUCTION = "production"


PRICING: dict[str, tuple[Decimal, Decimal]] = {
    "gpt-4o-mini": (Decimal("0.00015"), Decimal("0.00060")),
    "gpt-4o": (Decimal("0.00250"), Decimal("0.01000")),
}


class OpenAISettings(BaseSettings):
    endpoint: HttpUrl
    deployment: str = Field(min_length=1)
    model: Literal["gpt-4o-mini", "gpt-4o"] = "gpt-4o-mini"
    api_version: str = Field(default="2024-10-21",
                             pattern=r"^\d{4}-\d{2}-\d{2}(-preview)?$")
    api_key: SecretStr | None = None
    max_input_tokens: int = Field(default=8000, ge=100)
    max_output_tokens: int = Field(default=512, ge=1)

    @field_validator("endpoint")
    @classmethod
    def azure_host(cls, v: HttpUrl) -> HttpUrl:
        if not (v.host or "").endswith(".openai.azure.com"):
            raise ValueError(f"{v.host!r} is not an Azure OpenAI endpoint")
        return v

    @field_validator("api_key")
    @classmethod
    def keyvault_reference_must_have_resolved(
        cls, v: SecretStr | None
    ) -> SecretStr | None:
        """Catch an unresolved Key Vault reference (script 05).

        Both failure shapes are handled: empty, and the literal reference
        string left in place. Both are silent otherwise.
        """
        if v is None:
            return None
        raw = v.get_secret_value()
        if not raw or raw.startswith("@Microsoft.KeyVault"):
            raise ValueError(
                "Key Vault reference did not resolve — check the app's "
                "managed identity has 'Key Vault Secrets User' on the vault"
            )
        return v


class SearchSettings(BaseSettings):
    endpoint: HttpUrl
    index_name: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9-]{1,127}$")
    top_k: int = Field(default=5, ge=1, le=50)
    avg_chunk_tokens: int = Field(default=800, ge=50)


class LimitSettings(BaseSettings):
    max_concurrency: int = Field(default=8, ge=1, le=1000)
    daily_request_cap: int = Field(default=10_000, ge=1)
    daily_spend_cap_usd: Decimal = Field(default=Decimal("50"), gt=0)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ISC__", env_nested_delimiter="__",
        case_sensitive=False, extra="forbid",
    )

    environment: Environment = Environment.LOCAL
    service_version: str = "0.0.0-dev"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    managed_identity_client_id: str | None = None

    openai: OpenAISettings
    search: SearchSettings
    limits: LimitSettings = Field(default_factory=LimitSettings)  # type: ignore[arg-type]

    allowed_groups: Annotated[list[str], NoDecode] = Field(default_factory=list)

    @field_validator("allowed_groups", mode="before")
    @classmethod
    def parse_groups(cls, v: Any) -> Any:
        if isinstance(v, str):
            v = v.strip()
            if not v:
                return []
            return (json.loads(v) if v.startswith("[")
                    else [p.strip() for p in v.split(",") if p.strip()])
        return v

    # -- cross-field rules ------------------------------------------------

    @model_validator(mode="after")
    def production_hardening(self) -> Settings:
        if self.environment is Environment.PRODUCTION:
            if self.openai.api_key is not None:
                raise ValueError("api_key must not be set in production; "
                                 "use managed identity")
            if self.log_level == "DEBUG":
                raise ValueError("DEBUG logging in production risks logging "
                                 "prompts and retrieved document content")
            if not self.allowed_groups:
                raise ValueError("allowed_groups must be non-empty in "
                                 "production or permission trimming is off")
        return self

    @model_validator(mode="after")
    def cost_arithmetic(self) -> Settings:
        """Script 07's check: are the caps mutually consistent?"""
        if self.worst_case_daily_usd > self.limits.daily_spend_cap_usd:
            raise ValueError(
                f"{self.limits.daily_request_cap} requests at a worst case of "
                f"${self.worst_case_request_usd:.4f} is "
                f"${self.worst_case_daily_usd:.2f}/day, exceeding the cap of "
                f"${self.limits.daily_spend_cap_usd}"
            )
        return self

    @model_validator(mode="after")
    def context_budget(self) -> Settings:
        """top_k x chunk size must leave room for the prompt itself."""
        needed = self.search.top_k * self.search.avg_chunk_tokens
        budget = int(self.openai.max_input_tokens * 0.8)
        if needed > budget:
            raise ValueError(
                f"top_k={self.search.top_k} x {self.search.avg_chunk_tokens} "
                f"tokens = {needed}, exceeding 80% of max_input_tokens "
                f"({budget}). Retrieval will be truncated."
            )
        return self

    # -- derived ----------------------------------------------------------

    @computed_field  # type: ignore[prop-decorator]
    @property
    def uses_managed_identity(self) -> bool:
        return self.openai.api_key is None

    @property
    def worst_case_request_usd(self) -> Decimal:
        inp, out = PRICING[self.openai.model]
        return (Decimal(self.openai.max_input_tokens) / 1000 * inp
                + Decimal(self.openai.max_output_tokens) / 1000 * out)

    @property
    def worst_case_daily_usd(self) -> Decimal:
        return self.worst_case_request_usd * self.limits.daily_request_cap

    # -- diagnostics ------------------------------------------------------

    def redacted(self) -> dict[str, Any]:
        """Type-based redaction (script 04), recursive."""

        def walk(m: BaseModel) -> dict[str, Any]:
            out: dict[str, Any] = {}
            for name, value in m:
                if isinstance(value, SecretStr):
                    out[name] = "***REDACTED***"
                elif isinstance(value, BaseModel):
                    out[name] = walk(value)
                else:
                    out[name] = str(value) if not isinstance(
                        value, (int, float, bool, list, type(None))) else value
            return out

        return walk(self)

    @property
    def fingerprint(self) -> str:
        canonical = json.dumps(self.redacted(), sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode()).hexdigest()[:12]


# ===========================================================================
# 2. PROMPTS
# ===========================================================================

@dataclass(frozen=True)
class Prompt:
    name: str
    version: str
    template: str
    required: frozenset[str]

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(self.template.encode()).hexdigest()[:12]

    def render(self, **kw: object) -> str:
        missing = self.required - kw.keys()
        if missing:
            raise ValueError(f"{self.name}@{self.version} missing {sorted(missing)}")
        return self.template.format(**kw)


DISPOSITION_PROMPT = Prompt(
    "disposition", "v3",
    "ISC invoice exception analyst.\nContext:\n{context}\nQ: {question}\n"
    "Respond with JSON.",
    frozenset({"context", "question"}),
)


# ===========================================================================
# 3. STARTUP VALIDATION
# ===========================================================================

@dataclass
class Check:
    name: str
    ok: bool
    fatal: bool
    detail: str = ""


@dataclass
class World:
    deployment_exists: bool = True
    index_exists: bool = True
    has_search_role: bool = True
    telemetry_up: bool = True


async def run_startup_checks(s: Settings, w: World) -> list[Check]:
    """Concurrent, timeout-bounded, all results collected (script 03)."""

    async def deployment() -> Check:
        await asyncio.sleep(0.005)
        return Check("openai.deployment", w.deployment_exists, True,
                     "" if w.deployment_exists
                     else f"deployment {s.openai.deployment!r} not found "
                          f"(a deployment name is not a model name)")

    async def index() -> Check:
        await asyncio.sleep(0.005)
        return Check("search.index", w.index_exists, True,
                     "" if w.index_exists
                     else f"index {s.search.index_name!r} does not exist")

    async def permission() -> Check:
        await asyncio.sleep(0.005)
        return Check("search.permission", w.has_search_role, True,
                     "" if w.has_search_role
                     else "403 — managed identity lacks "
                          "'Search Index Data Reader'")

    async def telemetry() -> Check:
        await asyncio.sleep(0.005)
        return Check("telemetry", w.telemetry_up, False,
                     "" if w.telemetry_up else "App Insights unreachable")

    async with asyncio.timeout(5.0):
        return list(await asyncio.gather(deployment(), index(),
                                         permission(), telemetry()))


# ===========================================================================
# 4. THE SERVICE
# ===========================================================================

@dataclass
class SecretCache:
    vault: FakeKeyVault
    ttl: float = 30.0
    _value: str | None = None
    _at: float = 0.0
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    fetches: int = 0
    invalidations: int = 0

    async def get(self, name: str) -> str:
        if self._value and time.monotonic() - self._at < self.ttl:
            return self._value
        async with self._lock:
            if self._value and time.monotonic() - self._at < self.ttl:
                return self._value
            await asyncio.sleep(0.002)
            self._value = self.vault.get_secret(name).value
            self._at = time.monotonic()
            self.fetches += 1
            return self._value

    def invalidate(self) -> None:
        self.invalidations += 1
        self._value = None


@dataclass
class RagService:
    settings: Settings
    credential: FakeDefaultAzureCredential
    secrets: SecretCache
    spans: list[dict[str, Any]] = field(default_factory=list)

    def startup_log(self) -> dict[str, Any]:
        """The one line from script 07 part 6."""
        record = {
            "event": "service.started",
            "environment": self.settings.environment.value,
            "version": self.settings.service_version,
            "config_fingerprint": self.settings.fingerprint,
            "model": {
                "deployment": self.settings.openai.deployment,
                "model": self.settings.openai.model,
                "api_version": self.settings.openai.api_version,
            },
            "prompts": {
                DISPOSITION_PROMPT.name:
                    f"{DISPOSITION_PROMPT.version}@{DISPOSITION_PROMPT.fingerprint}",
            },
            "identity": {
                "mode": "managed_identity"
                        if self.settings.uses_managed_identity else "api_key",
                "client_id": self.settings.managed_identity_client_id,
            },
            "limits": {
                "max_concurrency": self.settings.limits.max_concurrency,
                "worst_case_daily_usd": float(self.settings.worst_case_daily_usd),
            },
        }
        log.info(json.dumps(record))
        return record

    async def answer(self, question: str, groups: frozenset[str]) -> dict[str, Any]:
        token = self.credential.get_token(
            "https://cognitiveservices.azure.com/.default")
        chunks = [f"[c{i}] retrieved passage {i}" for i in range(self.settings.search.top_k)]
        prompt = DISPOSITION_PROMPT.render(
            context="\n".join(chunks), question=question)

        span = {
            "name": "rag.answer",
            "attributes": {
                # Identifiers and shapes only. No prompt text, no chunk text,
                # no token value.
                "deployment": self.settings.openai.deployment,
                "model": self.settings.openai.model,
                "prompt.version": DISPOSITION_PROMPT.version,
                "prompt.fingerprint": DISPOSITION_PROMPT.fingerprint,
                "retrieval.k": self.settings.search.top_k,
                "prompt.chars": len(prompt),
                "token.expires_in_s": round(token.seconds_remaining),
                "principal.groups": sorted(groups),
            },
        }
        self.spans.append(span)
        return {"answer": "…", "span": span}


# ===========================================================================
# DEMONSTRATIONS
# ===========================================================================

GOOD_ENV = {
    "ISC__ENVIRONMENT": "production",
    "ISC__LOG_LEVEL": "INFO",
    "ISC__SERVICE_VERSION": "2026.8.3+a1b2c3d",
    "ISC__MANAGED_IDENTITY_CLIENT_ID": "8f2a41c7-0000-0000-0000-000000000001",
    "ISC__ALLOWED_GROUPS": "isc-all, isc-readers ,",
    "ISC__OPENAI__ENDPOINT": "https://isc-aoai-prod.openai.azure.com/",
    "ISC__OPENAI__DEPLOYMENT": "gpt-4o-mini-prod",
    "ISC__SEARCH__ENDPOINT": "https://isc-search.search.windows.net/",
    "ISC__SEARCH__INDEX_NAME": "isc-docs-v3",
}


async def demo1_startup() -> Settings:
    banner("DEMO 1 — startup: parse, validate, check, log")

    with clean_env("ISC__"), env(**GOOD_ENV):
        s = Settings()  # type: ignore[call-arg]

    section("settings")
    show("environment", s.environment.value)
    show("allowed_groups (cleaned)", s.allowed_groups)
    show("uses_managed_identity", s.uses_managed_identity)
    show("worst case daily spend", f"${s.worst_case_daily_usd:.2f}")
    show("config fingerprint", s.fingerprint)

    section("startup checks")
    checks = await run_startup_checks(s, World())
    for c in checks:
        print(f"      {'ok  ' if c.ok else 'FAIL'}  {c.name:<22} {c.detail}")

    section("startup log line")
    cred = FakeDefaultAzureCredential([
        FakeCredentialLink("ManagedIdentityCredential", available=True)])
    vault = FakeKeyVault()
    vault.set_secret("partner-key", "Pw-VAULTED-a91f3c-DO-NOT-SHIP")
    svc = RagService(s, cred, SecretCache(vault))
    with DETECTOR.capture_logs("isc.capstone"):
        svc.startup_log()
    return s


async def demo2_bad_configs() -> None:
    banner("DEMO 2 — misconfigurations rejected at startup")

    cases = [
        ("api key in production", {"ISC__OPENAI__API_KEY": FAKE_API_KEY}),
        ("DEBUG logging in production", {"ISC__LOG_LEVEL": "DEBUG"}),
        ("no allowed_groups", {"ISC__ALLOWED_GROUPS": ""}),
        ("unresolved Key Vault reference",
         {"ISC__OPENAI__API_KEY": "@Microsoft.KeyVault(SecretUri=...)",
          "ISC__ENVIRONMENT": "staging"}),
        ("expensive model, unchanged caps",
         {"ISC__OPENAI__MODEL": "gpt-4o"}),
        ("top_k too large for the context window",
         {"ISC__SEARCH__TOP_K": "20"}),
        ("typo in a variable name", {"ISC__SEARCH__TOPK": "8"}),
        ("wrong endpoint pasted",
         {"ISC__OPENAI__ENDPOINT": "https://api.openai.com/v1"}),
    ]
    for label, overrides in cases:
        with clean_env("ISC__"), env(**{**GOOD_ENV, **overrides}):
            try:
                Settings()  # type: ignore[call-arg]
                show(label, "ACCEPTED (unexpected)")
            except ValidationError as e:
                msg = e.errors()[0]["msg"].replace("Value error, ", "")
                show(label, f"REJECTED: {msg[:44]}")


async def demo3_runtime(s: Settings) -> RagService:
    banner("DEMO 3 — a request, with telemetry that carries no secrets")

    cred = FakeDefaultAzureCredential([
        FakeCredentialLink("ManagedIdentityCredential", available=True)])
    vault = FakeKeyVault()
    vault.set_secret("partner-key", "Pw-VAULTED-a91f3c-DO-NOT-SHIP")
    svc = RagService(s, cred, SecretCache(vault))

    result = await svc.answer("why is invoice 88 over-billed?",
                              frozenset({"isc-all"}))
    print(json.dumps(result["span"], indent=2))

    section("credential caching under load")
    for _ in range(200):
        cred.get_token("https://cognitiveservices.azure.com/.default")
    show("get_token calls", 201)
    show("identity endpoint calls", cred.links[0].calls)
    show("cache hits / misses", f"{cred.cache_hits} / {cred.cache_misses}")
    return svc


async def demo4_leak_audit(svc: RagService) -> None:
    banner("DEMO 4 — the audit: does anything leak?")

    # Force a secret into memory the way a real request would.
    await svc.secrets.get("partner-key")

    surfaces: list[tuple[str, str]] = [
        ("settings repr", repr(svc.settings)),
        ("settings model_dump_json", svc.settings.model_dump_json()),
        ("redacted diagnostics", json.dumps(svc.settings.redacted())),
        ("startup log record", json.dumps(svc.startup_log())),
        ("all spans", json.dumps(svc.spans)),
    ]

    try:
        raise ConnectionError(
            f"failed to reach {svc.settings.openai.endpoint} as "
            f"{svc.settings.managed_identity_client_id}")
    except ConnectionError as e:
        DETECTOR.scan_exception(e)
        surfaces.append(("exception message", str(e)))

    before = len(DETECTOR.leaks)
    for label, text in surfaces:
        found = DETECTOR.scan(text, label)
        verdict(not found, f"{label:<28} {len(found)} leak(s)")

    section("audit result")
    total_new = len(DETECTOR.leaks) - before
    print(f"      {DETECTOR.report()}")
    verdict(total_new == 0,
            f"{total_new} leak(s) across {len(surfaces)} emitted surfaces")

    print("""
    THIS IS THE ASSERTION THAT MATTERS. Everything else in this file is
    setup for it: a scanner that knows the real secret values, run over
    everything the service emitted, asserting zero.

    Put exactly this in your test suite (script 09). "We are careful about
    logging secrets" is not a control; a red test is.""")


async def main() -> None:
    s = await demo1_startup()
    await demo2_bad_configs()
    svc = await demo3_runtime(s)
    await demo4_leak_audit(svc)

    banner("WHAT TO DEFEND IN A DESIGN REVIEW")
    print("""
  1. One settings class is the entire configuration surface. `extra="forbid"`
     turns a typo into a startup failure rather than a silent default.
  2. Cross-field validators encode production rules that per-field
     constraints cannot: no static key in prod, no DEBUG logging in prod, no
     empty allowed_groups, caps that are mutually consistent, retrieval that
     fits the context window.
  3. Key Vault references are validated as RESOLVED. Both failure shapes —
     empty and the literal reference — are caught, because both are silent.
  4. Secrets are SecretStr, so redaction follows from the type rather than
     from a denylist of field names someone has to maintain.
  5. Telemetry carries identifiers, counts, and fingerprints — never prompt
     text, chunk text, or token values. Telemetry is a separate trust
     boundary with a larger reader set.
  6. One credential object for the process. 201 get_token calls, one call to
     the identity endpoint.
  7. Startup checks run concurrently, are timeout-bounded, and collect ALL
     results before deciding — one report, not one problem per deploy.
  8. Prompt fingerprints are logged, so any answer traces back to the exact
     text that produced it.
  9. The config fingerprint makes "did config change?" and "does production
     match staging?" string comparisons.
 10. A leak scanner runs over every emitted surface and asserts zero.
""")


if __name__ == "__main__":
    asyncio.run(main())
