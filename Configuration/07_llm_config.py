"""
07 — Configuration in LLM systems: what is different here.

WHAT MAKES GENAI CONFIG UNUSUAL
-------------------------------
Three things that do not come up in a normal service:

  1. THE PROMPT IS CONFIGURATION — it changes behaviour, it is deployed, it
     needs versioning and diffing — but unlike a timeout, changing it changes
     what the system SAYS. That puts it in an awkward middle ground between
     "operational knob" and "code that needs review".

  2. CONFIG HAS A DIRECT DOLLAR COST. `max_tokens`, `top_k`, and which
     deployment you route to are cost decisions. A config change can multiply
     your bill without any code change and without any error.

  3. QUALITY IS NOT OBSERVABLE FROM CONFIG. A wrong timeout produces errors.
     A wrong temperature or a truncated context produces plausible, fluent,
     WRONG answers, at a normal latency, with a 200 status code.

That third point is the one that should change how you treat these settings.
Run:  python 07_llm_config.py
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, Field, ValidationError, model_validator

from config_lab import banner, section, show

# ---------------------------------------------------------------------------
# PART 1 — deployment vs model, again, because it keeps costing people time
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelBinding:
    """The four things you need to call a model, and they are all different.

    People collapse these into one string and then cannot answer basic
    questions: which physical model served this request? what does it cost?
    is this deployment provisioned or pay-as-you-go?
    """

    deployment: str        # what YOU named it. Goes in the API call.
    model: str             # the underlying model family.
    model_version: str     # the dated snapshot actually serving.
    api_version: str       # the REST contract version.

    def cost_key(self) -> str:
        """Pricing follows the MODEL, not your deployment name. If you record
        only the deployment, you cannot attribute cost without a lookup table
        that will drift."""
        return f"{self.model}:{self.model_version}"


def part1_bindings() -> None:
    banner("PART 1 — deployment name is not a model name")

    bindings = [
        ModelBinding("gpt-4o-mini-prod", "gpt-4o-mini", "2024-07-18", "2024-10-21"),
        ModelBinding("isc-summariser", "gpt-4o-mini", "2024-07-18", "2024-10-21"),
        ModelBinding("isc-reasoner", "gpt-4o", "2024-11-20", "2024-10-21"),
    ]
    print(f"    {'deployment':<22} {'model':<14} {'version':<12} cost key")
    print(f"    {'-' * 22} {'-' * 14} {'-' * 12} {'-' * 24}")
    for b in bindings:
        print(f"    {b.deployment:<22} {b.model:<14} {b.model_version:<12} "
              f"{b.cost_key()}")

    print("""
    Two of these deployments run the SAME model. A dashboard grouped by
    deployment name shows them as separate things; grouped by cost_key it
    shows one. You want both views, which means recording both.

    AND THE ONE THAT BITES: `model_version` can change UNDER YOU. An Azure
    OpenAI deployment set to auto-update moves to a new model snapshot on
    Microsoft's schedule. Your config did not change, your code did not
    change, and your outputs did.

    SO: record the model_version returned in each response, and alert when it
    changes. That single alert explains a whole category of "the answers got
    worse last Tuesday and nobody deployed anything".""")


# ---------------------------------------------------------------------------
# PART 2 — prompts are configuration, with a caveat
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PromptTemplate:
    """A versioned, fingerprinted prompt.

    THE FINGERPRINT IS THE POINT. Without it you cannot answer "which prompt
    produced this answer?" for a response you are investigating six weeks
    later, and you cannot correlate a quality change with a prompt change.
    """

    name: str
    version: str
    template: str
    required_vars: frozenset[str] = frozenset()

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(self.template.encode()).hexdigest()[:12]

    def render(self, **kwargs: object) -> str:
        missing = self.required_vars - kwargs.keys()
        if missing:
            # FAIL LOUDLY. A prompt rendered with a missing variable produces
            # a literal "{context}" in the text sent to the model, which the
            # model will cheerfully answer around. No error, plausible output,
            # entirely ungrounded.
            raise ValueError(
                f"prompt {self.name}@{self.version} missing variables: "
                f"{sorted(missing)}"
            )
        return self.template.format(**kwargs)


PROMPTS = {
    "disposition": PromptTemplate(
        name="disposition", version="v3",
        template=("You are an ISC invoice exception analyst.\n"
                  "Context:\n{context}\n\nQuestion: {question}\n"
                  "Respond with JSON matching the schema."),
        required_vars=frozenset({"context", "question"}),
    ),
}


def part2_prompts() -> None:
    banner("PART 2 — prompts as versioned, fingerprinted config")

    p = PROMPTS["disposition"]
    show("name@version", f"{p.name}@{p.version}")
    show("fingerprint", p.fingerprint)

    section("a missing variable fails loudly")
    try:
        p.render(question="why is invoice 88 over-billed?")
    except ValueError as e:
        show("render() with no context", f"ValueError: {e}")
    print("""
      Without that check, `{context}` reaches the model as a literal string.
      The model answers from parametric knowledge, fluently, with no
      retrieved grounding — and nothing in the response or the metrics says
      so. This is the highest-severity silent failure in a RAG system and it
      is prevented by six lines.""")

    print("""
    WHERE THE PROMPT SHOULD LIVE — and this is a genuine trade-off, not a
    settled question:

    IN THE REPO (recommended default)
      + reviewed, diffed, versioned with the code that depends on it
      + a prompt change is a deploy, so it appears in your change log
      + testable in CI against an eval set
      - iteration requires a deploy, which slows prompt engineering

    IN A CONFIG STORE (App Configuration, a database)
      + change without deploying; A/B testing is easy
      - a change that alters what the system SAYS, with no review, no test,
        and no deploy record
      - the version that produced a given answer is not reconstructible
        unless you log the fingerprint with every response

    MY POSITION: repo by default. A prompt determines the system's output;
    that is the definition of something that should require review. If you do
    externalise it, treat it like a schema migration — versioned, immutable
    once published, with the fingerprint logged on every response so any
    answer can be traced back to the exact text that produced it.

    THE MINIMUM EITHER WAY: log `prompt.name`, `prompt.version`, and
    `prompt.fingerprint` on every request. Then "did the prompt change?" is a
    query, not an archaeology project.""")


# ---------------------------------------------------------------------------
# PART 3 — cost caps as configuration
# ---------------------------------------------------------------------------

PRICING: dict[str, tuple[Decimal, Decimal]] = {
    # (input per 1k tokens, output per 1k tokens) — illustrative
    "gpt-4o-mini": (Decimal("0.00015"), Decimal("0.00060")),
    "gpt-4o": (Decimal("0.00250"), Decimal("0.01000")),
}


class CostSettings(BaseModel):
    """Cost limits, with the arithmetic done at STARTUP.

    THE INSIGHT: a spend cap you cannot reach is not a cap. Validating that
    your per-request limits are consistent with your daily cap turns a
    surprise invoice into a startup error.
    """

    model: Literal["gpt-4o-mini", "gpt-4o"] = "gpt-4o-mini"
    max_input_tokens: int = Field(default=8000, ge=100)
    max_output_tokens: int = Field(default=512, ge=1)
    daily_request_cap: int = Field(default=10_000, ge=1)
    daily_spend_cap_usd: Decimal = Field(default=Decimal("50.00"), gt=0)

    @property
    def worst_case_request_usd(self) -> Decimal:
        inp, out = PRICING[self.model]
        return (Decimal(self.max_input_tokens) / 1000 * inp
                + Decimal(self.max_output_tokens) / 1000 * out)

    @property
    def worst_case_daily_usd(self) -> Decimal:
        return self.worst_case_request_usd * self.daily_request_cap

    @model_validator(mode="after")
    def caps_must_be_consistent(self) -> CostSettings:
        if self.worst_case_daily_usd > self.daily_spend_cap_usd:
            raise ValueError(
                f"inconsistent caps: {self.daily_request_cap} requests at a "
                f"worst case of ${self.worst_case_request_usd:.4f} each is "
                f"${self.worst_case_daily_usd:.2f}/day, which exceeds the "
                f"daily_spend_cap_usd of ${self.daily_spend_cap_usd}. "
                f"Lower the request cap, lower max tokens, or raise the "
                f"spend cap deliberately."
            )
        return self


def part3_cost() -> None:
    banner("PART 3 — cost caps that are actually consistent")

    section("a sane configuration")
    c = CostSettings()
    show("model", c.model)
    show("worst case per request", f"${c.worst_case_request_usd:.5f}")
    show("worst case per day", f"${c.worst_case_daily_usd:.2f}")
    show("daily cap", f"${c.daily_spend_cap_usd}")

    section("switching the model without revisiting the caps")
    try:
        CostSettings(model="gpt-4o")
    except ValidationError as e:
        msg = str(e).split("Value error, ")[-1].split(" [type")[0]
        for line in msg.split(". "):
            if line.strip():
                print(f"      {line.strip()}.")

    print("""
    THE SCENARIO THIS PREVENTS: someone changes ISC_AI__COST__MODEL from
    gpt-4o-mini to gpt-4o to improve answer quality. It is one environment
    variable. Nothing errors. Quality does improve. The bill goes up roughly
    17x and nobody finds out until the monthly invoice.

    With the validator, that change fails at startup with the arithmetic
    spelled out, and the person making it has to raise the spend cap
    deliberately — which is a decision someone can review.

    THE BROADER PATTERN: any two config values with an implied relationship
    should be checked against each other at startup. Other examples in a RAG
    system:
      * top_k x average_chunk_tokens must fit inside max_input_tokens
      * max_concurrency x per_request_tokens must fit inside your TPM quota
      * request timeout must exceed max_output_tokens x per-token latency
    Each is two lines in a model_validator and each prevents a class of
    incident that is otherwise diagnosed from the outside.""")

    section("top_k vs context window, checked")
    for top_k, chunk_tokens, limit in [(5, 800, 8000), (20, 800, 8000)]:
        needed = top_k * chunk_tokens
        ok = needed < limit * 0.8      # leave headroom for the prompt itself
        show(f"top_k={top_k}, {chunk_tokens} tok/chunk",
             f"{needed} tokens vs {limit} limit — "
             f"{'ok' if ok else 'WILL TRUNCATE OR 400'}")


# ---------------------------------------------------------------------------
# PART 4 — routing and feature flags
# ---------------------------------------------------------------------------

class RoutingRule(BaseModel):
    """Route a request class to a deployment.

    Model routing is legitimately operational: you want to move traffic to a
    cheaper model, or to a different region during an incident, without a
    deploy. That makes it config.
    """

    name: str
    match_intent: Literal["simple_lookup", "analysis", "generation", "default"]
    deployment: str
    max_output_tokens: int = 512
    enabled: bool = True


DEFAULT_ROUTES = [
    RoutingRule(name="cheap-lookups", match_intent="simple_lookup",
                deployment="gpt-4o-mini-prod", max_output_tokens=256),
    RoutingRule(name="reasoning", match_intent="analysis",
                deployment="isc-reasoner", max_output_tokens=1024),
    RoutingRule(name="fallback", match_intent="default",
                deployment="gpt-4o-mini-prod", max_output_tokens=512),
]


def part4_routing() -> None:
    banner("PART 4 — routing rules and flag safety")

    def route(intent: str, rules: list[RoutingRule]) -> RoutingRule:
        for r in rules:
            if r.enabled and r.match_intent == intent:
                return r
        # THE IMPORTANT PART: there must ALWAYS be a default, and it must be
        # the cheap, safe one. A routing table with no match should not raise
        # and should not pick the expensive model.
        for r in rules:
            if r.enabled and r.match_intent == "default":
                return r
        raise RuntimeError("no enabled default route — misconfiguration")

    for intent in ("simple_lookup", "analysis", "unknown_new_intent"):
        r = route(intent, DEFAULT_ROUTES)
        show(intent, f"-> {r.deployment} (max_tokens={r.max_output_tokens})")

    section("the reasoning route disabled during an incident")
    degraded = [r.model_copy(update={"enabled": r.name != "reasoning"})
                for r in DEFAULT_ROUTES]
    r = route("analysis", degraded)
    show("analysis (reasoning route off)", f"-> {r.deployment}")
    print("      Falls back to the cheap model rather than failing. Whether")
    print("      that is right is a product decision — but it must be a")
    print("      DECISION, and the degraded answer must be labelled.")

    print("""
    FEATURE FLAG RULES for LLM systems:

      DEFAULT OFF, always. A flag that defaults on is enabled everywhere it
      was not explicitly configured, including in the environment you forgot.

      FLAGS THAT COST MONEY NEED A SEPARATE CAP. `enable_reasoning_model` is
      not a boolean, it is a spending decision. Pair it with a request cap.

      REMOVE FLAGS. A flag older than one release cycle is not a flag, it is
      an untested code path. Every flag doubles the configuration space you
      are nominally supporting.

      NEVER FLAG A SAFETY CONTROL. `enable_permission_filtering` must not
      exist. Anything that can turn off permission trimming, output
      filtering, or audit logging is one portal click from an incident, and
      it will be clicked during a demo.""")


# ---------------------------------------------------------------------------
# PART 5 — per-tenant configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TenantConfig:
    tenant_id: str
    allowed_groups: frozenset[str]
    daily_spend_cap_usd: Decimal
    deployment: str
    data_residency: Literal["global", "eu", "us", "apac"]


def part5_tenants() -> None:
    banner("PART 5 — per-tenant config, and where it must not live")

    tenants = {
        "isc-sg": TenantConfig("isc-sg", frozenset({"isc-all", "isc-sg"}),
                               Decimal("50"), "gpt-4o-mini-prod", "apac"),
        "isc-eu": TenantConfig("isc-eu", frozenset({"isc-all", "isc-eu"}),
                               Decimal("200"), "gpt-4o-mini-eu", "eu"),
    }
    for t in tenants.values():
        show(t.tenant_id, f"groups={sorted(t.allowed_groups)} "
                          f"cap=${t.daily_spend_cap_usd} residency={t.data_residency}")

    print("""
    THE HARD RULE: per-tenant config is DATA, not deployment configuration.
    It lives in a store, keyed by tenant, and is loaded per request.

    The failure mode of putting it in environment variables is that adding a
    tenant becomes a deploy, so it gets batched, so someone eventually adds a
    tenant by editing a shared variable under time pressure and truncates
    another tenant's group list.

    AND THE SECURITY BOUNDARY: `allowed_groups` here is per-tenant DATA that
    gates retrieval. It must be:
      * loaded from an authoritative source, not cached indefinitely
      * keyed on the authenticated tenant, never on a request parameter
      * FAIL CLOSED — a tenant whose config cannot be loaded gets no results,
        not default results

    That last one connects to the failure-handling tutorial: falling back to
    a default tenant config on a lookup failure is the same shape of bug as
    falling back from a permission-trimmed index to an untrimmed one.

    DATA RESIDENCY IS CONFIG WITH LEGAL CONSEQUENCES. `deployment` and
    `data_residency` together determine which region processes the prompt.
    Getting it wrong is not a bug, it is a compliance incident — so validate
    at startup that every configured deployment is in a region consistent
    with its tenant's residency requirement, and make the check fatal.""")


# ---------------------------------------------------------------------------
# PART 6 — the startup log line
# ---------------------------------------------------------------------------

def part6_startup_line() -> None:
    banner("PART 6 — one startup line that answers most questions")

    record = {
        "event": "service.started",
        "service": "isc-rag",
        "version": "2026.8.3+a1b2c3d",
        "environment": "production",
        "config_fingerprint": "7f3a91c40be2",
        "model": {
            "deployment": "gpt-4o-mini-prod",
            "model": "gpt-4o-mini",
            "model_version": "2024-07-18",
            "api_version": "2024-10-21",
        },
        "prompts": {"disposition": "v3@a41f9c02be71"},
        "retrieval": {"index": "isc-docs-v3", "top_k": 5, "rerank": True},
        "limits": {"max_concurrency": 16, "rpm": 300,
                   "daily_spend_cap_usd": 50.0},
        "identity": {"mode": "managed_identity",
                     "client_id": "8f2a-...-41c7"},
        "features": {"streaming": True, "tool_calling": False},
    }
    print(json.dumps(record, indent=2)[:1100])

    print("""
    NOTE WHAT IS ABSENT: no keys, no connection strings, no endpoints with
    embedded credentials. Every value here is either an identifier, a
    non-secret setting, or a fingerprint.

    WHAT IT LETS YOU ANSWER, without redeploying anything:
      * "did config change between these two deploys?"        fingerprint diff
      * "which prompt version produced last week's answers?"  prompts
      * "did the model version change under us?"              model_version
      * "is production running what staging ran?"             fingerprint
      * "are we on managed identity or a key?"                identity.mode

    One line, at INFO, on every start. It is the cheapest observability in
    the whole system.""")


def main() -> None:
    part1_bindings()
    part2_prompts()
    part3_cost()
    part4_routing()
    part5_tenants()
    part6_startup_line()

    banner("SUMMARY")
    print("""
  * deployment / model / model_version / api_version are four different
    things. Record all four; alert when model_version changes under you.
  * Prompts are config: version them, fingerprint them, log the fingerprint
    with every response. Default to keeping them in the repo, because a
    prompt change changes what the system SAYS.
  * A missing prompt variable must fail loudly — a literal "{context}" in the
    prompt produces a fluent, ungrounded answer with no error.
  * Validate cost arithmetic at startup. A one-variable model switch can be a
    17x bill increase with no error anywhere.
  * Check config values that have implied relationships against each other:
    top_k x chunk_tokens vs context window, concurrency x tokens vs TPM quota.
  * Flags default OFF, get removed after a release, and never gate a safety
    control.
  * Per-tenant config is DATA, loaded per request, failing closed.
  * One startup log line with a config fingerprint, prompt fingerprints, and
    the model binding answers most incident questions for free.
""")


if __name__ == "__main__":
    main()
