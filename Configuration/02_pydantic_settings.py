"""
02 — pydantic-settings: one class that is the whole configuration surface.

WHY THIS LIBRARY RATHER THAN os.environ
---------------------------------------
The scattered-`os.environ.get()` approach has no single place that answers
"what does this service need to run?". A settings class answers it in one
screen, and gets four things for free:

    PARSING       "8" -> 8, "true" -> True, "a,b" -> ["a","b"], with a
                  boolean vocabulary that actually works
    VALIDATION    ranges, patterns, required-ness, cross-field rules
    FAIL-FAST     every problem reported at once, at startup, with the
                  variable name and what was wrong
    DOCUMENTATION the class IS the reference, and can generate a
                  .env.example

The fourth matters more than it sounds. A settings class with descriptions is
the artefact you hand someone onboarding, and it cannot drift from reality
because the service reads the same declaration.

Run:  python 02_pydantic_settings.py
"""

from __future__ import annotations

import json
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import (
    AliasChoices,
    Field,
    HttpUrl,
    SecretStr,
    ValidationError,
    computed_field,
    field_validator,
    model_validator,
)
from pydantic_settings import (
    BaseSettings,
    NoDecode,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

from config_lab import (
    FAKE_API_KEY,
    banner,
    clean_env,
    env,
    show,
)

# ---------------------------------------------------------------------------
# PART 1 — nested settings
# ---------------------------------------------------------------------------

class Environment(StrEnum):
    """`StrEnum` (3.11+) so it compares equal to a plain string AND
    serialises cleanly. A bare `Enum` requires `.value` at every boundary.

    On <3.11 the equivalent is `class Environment(str, Enum)`, which ruff's
    UP042 will tell you to replace once you are on 3.11 — the two are not
    quite identical (`str(Environment.LOCAL)` differs) so check any code that
    formats one before switching."""

    LOCAL = "local"
    DEV = "dev"
    STAGING = "staging"
    PRODUCTION = "production"


class AzureOpenAISettings(BaseSettings):
    """A nested settings group.

    Env vars for this group, given the parent's prefix and `__` delimiter:
        ISC_AI__OPENAI__ENDPOINT
        ISC_AI__OPENAI__DEPLOYMENT
        ISC_AI__OPENAI__API_VERSION
    """

    endpoint: HttpUrl = Field(
        description="Azure OpenAI resource endpoint, e.g. "
                    "https://isc-aoai-prod.openai.azure.com/",
    )

    deployment: str = Field(
        min_length=1,
        description="DEPLOYMENT name, not a model name. This is an arbitrary "
                    "string chosen when the deployment was created.",
    )
    """The distinction is load-bearing. Code that hardcodes 'gpt-4o-mini' as a
    deployment name works in the tenant where someone happened to name it that
    and breaks everywhere else, with a 404 that reads like the model does not
    exist."""

    api_version: str = Field(
        default="2024-10-21",
        pattern=r"^\d{4}-\d{2}-\d{2}(-preview)?$",
        description="Azure OpenAI API version (dated).",
    )

    api_key: SecretStr | None = Field(
        default=None,
        description="Only for local development. Production uses managed "
                    "identity — leave unset.",
    )

    max_tokens: int = Field(default=512, ge=1, le=16384)
    timeout_s: float = Field(default=60.0, gt=0, le=600)

    @field_validator("endpoint")
    @classmethod
    def must_be_azure_openai(cls, v: HttpUrl) -> HttpUrl:
        """Catch the wrong-endpoint-shape mistake at startup rather than as a
        404 on the first request. People paste the portal URL, the resource id,
        or an api.openai.com URL here constantly."""
        host = v.host or ""
        if not (host.endswith(".openai.azure.com")
                or host.endswith(".cognitiveservices.azure.com")):
            raise ValueError(
                f"host {host!r} does not look like an Azure OpenAI endpoint; "
                f"expected *.openai.azure.com"
            )
        return v


class SearchSettings(BaseSettings):
    endpoint: HttpUrl = Field(description="Azure AI Search endpoint.")
    index_name: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9-]{1,127}$")
    top_k: int = Field(default=5, ge=1, le=50)
    semantic_ranker: bool = False


class LimitSettings(BaseSettings):
    max_concurrency: int = Field(default=8, ge=1, le=1000)
    requests_per_minute: int = Field(default=60, ge=1)
    daily_spend_cap_usd: float = Field(default=10.0, gt=0)
    per_request_token_cap: int = Field(default=8000, ge=100)


class AppSettings(BaseSettings):
    """THE configuration surface. One class, one import, one source of truth."""

    model_config = SettingsConfigDict(
        env_prefix="ISC_AI__",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="forbid",
        # `extra="forbid"` on a settings class is a judgement call. It catches
        # typos — ISC_AI__TOPK instead of ISC_AI__TOP_K would otherwise be
        # silently ignored, leaving the default in place. The cost is that any
        # unrecognised ISC_AI__* variable becomes a startup failure, which is
        # loud if someone sets a variable for a future release. Worth it: a
        # silently-ignored config variable is much worse than a loud one.
        validate_default=True,
    )

    environment: Environment = Environment.LOCAL
    service_name: str = "isc-rag"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    openai: AzureOpenAISettings
    search: SearchSettings
    limits: LimitSettings = Field(default_factory=LimitSettings)  # type: ignore[arg-type]

    allowed_groups: Annotated[list[str], NoDecode] = Field(
        default_factory=list,
        description="Comma-separated or JSON array of Entra group ids.",
    )
    """NOTE THE `NoDecode`. This cost me a debugging round, and it is a real
    gotcha rather than a detail.

    pydantic-settings treats any list/dict/set field as COMPLEX and runs
    `json.loads()` on the raw env value INSIDE THE SOURCE — before any
    `field_validator(mode="before")` gets a chance to run. So a
    comma-separated value raises, from the source layer, with:

        SettingsError: error parsing value for field "allowed_groups"
        from source "EnvSettingsSource"
        ... json.decoder.JSONDecodeError: Expecting value

    which points at JSON parsing and gives no hint that your validator exists.
    `mode="before"` is not early enough; the decode happens before "before".

    `Annotated[list[str], NoDecode]` disables that source-level decode and
    hands the raw string to your validator, which is what you wanted. Its
    counterpart `ForceDecode` does the opposite, for a field you want JSON-
    decoded that pydantic-settings would not treat as complex."""

    @field_validator("allowed_groups", mode="before")
    @classmethod
    def parse_groups(cls, v: Any) -> Any:
        """Accept both a JSON array and a comma-separated string.

        WHY BOTH: a JSON array is unambiguous and is what you want in a
        Kubernetes ConfigMap. A comma-separated string is what someone types
        into the App Service portal. Supporting both costs six lines and
        removes an entire category of "why is my config not applying".

        NOTE the trailing-empty-element strip. From script 01: a trailing
        comma produces "" as a group id, which in a permission filter means
        an empty group — either matching nothing or, in a badly written
        filter, matching everything.
        """
        if isinstance(v, str):
            v = v.strip()
            if not v:
                return []
            if v.startswith("["):
                return json.loads(v)
            return [part.strip() for part in v.split(",") if part.strip()]
        return v

    @model_validator(mode="after")
    def production_rules(self) -> AppSettings:
        """CROSS-FIELD RULES. This is where a settings class earns its keep —
        none of these can be expressed as a per-field constraint.

        Each rule below is a real production incident in miniature.
        """
        if self.environment is Environment.PRODUCTION:
            if self.openai.api_key is not None:
                raise ValueError(
                    "openai.api_key must not be set in production — "
                    "use managed identity. A static key in prod is a "
                    "rotation and audit problem."
                )
            if self.log_level == "DEBUG":
                raise ValueError(
                    "log_level=DEBUG in production risks logging prompts and "
                    "retrieved document content"
                )
            if not self.allowed_groups:
                raise ValueError(
                    "allowed_groups must be non-empty in production — "
                    "an empty list disables permission trimming"
                )
        if self.environment is Environment.LOCAL and self.limits.daily_spend_cap_usd > 25:
            raise ValueError(
                f"daily_spend_cap_usd={self.limits.daily_spend_cap_usd} is too "
                f"high for a local environment; cap local spend at 25"
            )
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def uses_managed_identity(self) -> bool:
        """A DERIVED value, computed once, not a config field.

        Making this a settings field would let someone set
        `use_managed_identity=true` while also supplying an api_key, creating
        two sources of truth that can disagree. Derive it.
        """
        return self.openai.api_key is None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def openai_scope(self) -> str:
        return "https://cognitiveservices.azure.com/.default"


BASE_ENV = {
    "ISC_AI__OPENAI__ENDPOINT": "https://isc-aoai-prod.openai.azure.com/",
    "ISC_AI__OPENAI__DEPLOYMENT": "gpt-4o-mini-prod",
    "ISC_AI__SEARCH__ENDPOINT": "https://isc-search.search.windows.net/",
    "ISC_AI__SEARCH__INDEX_NAME": "isc-docs-v3",
}


def part1_nested() -> None:
    banner("PART 1 — nested settings from flat environment variables")

    with clean_env("ISC_AI__"), env(
        **BASE_ENV,
        ISC_AI__ENVIRONMENT="staging",
        ISC_AI__LOG_LEVEL="WARNING",
        ISC_AI__SEARCH__TOP_K="8",
        ISC_AI__LIMITS__MAX_CONCURRENCY="16",
        ISC_AI__ALLOWED_GROUPS="isc-all, isc-readers ,",
    ):
        s = AppSettings()  # type: ignore[call-arg]
        show("environment", s.environment)
        show("openai.deployment", s.openai.deployment)
        show("openai.api_version (default)", s.openai.api_version)
        show("search.top_k (parsed from '8')", f"{s.search.top_k!r}")
        show("limits.max_concurrency", s.limits.max_concurrency)
        show("limits.daily_spend_cap_usd (default)", s.limits.daily_spend_cap_usd)
        show("allowed_groups (stripped)", s.allowed_groups)
        show("uses_managed_identity (derived)", s.uses_managed_identity)

    print("""
    Note `allowed_groups`: the input was "isc-all, isc-readers ," — with
    whitespace and a trailing comma, exactly as a human would paste it into a
    portal field. The validator produced a clean two-element list rather than
    a three-element one containing "".""")


# ---------------------------------------------------------------------------
# PART 2 — errors reported all at once
# ---------------------------------------------------------------------------

def part2_error_quality() -> None:
    banner("PART 2 — every problem at once, with the variable name")

    with clean_env("ISC_AI__"), env(
        ISC_AI__OPENAI__ENDPOINT="https://api.openai.com/v1",
        ISC_AI__OPENAI__DEPLOYMENT="",
        ISC_AI__OPENAI__API_VERSION="v1",
        ISC_AI__SEARCH__ENDPOINT="not-a-url",
        ISC_AI__SEARCH__INDEX_NAME="Invalid_Index_Name",
        ISC_AI__SEARCH__TOP_K="500",
        ISC_AI__LOG_LEVEL="TRACE",
    ):
        try:
            AppSettings()  # type: ignore[call-arg]
        except ValidationError as e:
            show("problems found", e.error_count())
            print()
            for err in e.errors():
                loc = "__".join(str(x) for x in err["loc"]).upper()
                print(f"      ISC_AI__{loc:<26} {err['msg'][:52]}")

    print("""
    SEVEN problems, one startup, one error. Compare with the
    `os.environ.get()` approach, which surfaces them one deployment at a time:
    fix the endpoint, redeploy, discover the deployment name is empty,
    redeploy, discover the API version is malformed...

    Each round trip through a deployment pipeline is 5-20 minutes. Reporting
    all of them at once is the single biggest practical win of a settings
    class, and it is why validation belongs at STARTUP rather than at first
    use — script 03.""")


# ---------------------------------------------------------------------------
# PART 3 — cross-field production rules
# ---------------------------------------------------------------------------

def part3_cross_field() -> None:
    banner("PART 3 — rules that only make sense across fields")

    cases = [
        ("api key set in production",
         {"ISC_AI__ENVIRONMENT": "production",
          "ISC_AI__OPENAI__API_KEY": FAKE_API_KEY,
          "ISC_AI__ALLOWED_GROUPS": "isc-all",
          "ISC_AI__LOG_LEVEL": "WARNING"}),
        ("DEBUG logging in production",
         {"ISC_AI__ENVIRONMENT": "production", "ISC_AI__LOG_LEVEL": "DEBUG",
          "ISC_AI__ALLOWED_GROUPS": "isc-all"}),
        ("no allowed_groups in production",
         {"ISC_AI__ENVIRONMENT": "production", "ISC_AI__LOG_LEVEL": "WARNING"}),
        ("high spend cap locally",
         {"ISC_AI__ENVIRONMENT": "local",
          "ISC_AI__LIMITS__DAILY_SPEND_CAP_USD": "500"}),
        ("valid production config",
         {"ISC_AI__ENVIRONMENT": "production", "ISC_AI__LOG_LEVEL": "WARNING",
          "ISC_AI__ALLOWED_GROUPS": "isc-all,isc-readers"}),
    ]

    for label, overrides in cases:
        with clean_env("ISC_AI__"), env(**BASE_ENV, **overrides):
            try:
                AppSettings()  # type: ignore[call-arg]
                show(label, "ACCEPTED")
            except ValidationError as e:
                msg = e.errors()[0]["msg"].replace("Value error, ", "")
                show(label, f"REJECTED: {msg[:46]}")

    print("""
    Read the first three rows as three separate incidents that cannot now
    happen:

      * A static API key in production means a credential that does not
        rotate, does not appear in Entra sign-in logs, and cannot be
        attributed to a workload identity.
      * DEBUG logging in production means prompts and retrieved chunks in
        Application Insights — document content crossing into a telemetry
        workspace whose readers are a different set of people.
      * Empty allowed_groups means permission trimming is disabled and every
        query searches the whole index.

    None is expressible as a per-field constraint. All three are three lines
    in a model_validator, checked before the process accepts traffic.""")


# ---------------------------------------------------------------------------
# PART 4 — aliases and platform variables
# ---------------------------------------------------------------------------

class PlatformAwareSettings(BaseSettings):
    """Reads variables you do not control alongside your own.

    Platforms inject their own names — PORT, WEBSITE_HOSTNAME, K_SERVICE,
    WEBSITES_PORT — and you cannot prefix them. `AliasChoices` lets one field
    accept several, in priority order.
    """

    model_config = SettingsConfigDict(env_prefix="ISC_AI__", extra="ignore")

    port: int = Field(
        default=8000,
        validation_alias=AliasChoices(
            "ISC_AI__PORT",     # ours wins
            "WEBSITES_PORT",    # Azure App Service
            "PORT",             # Container Apps, Heroku, generic
        ),
    )

    instance_id: str = Field(
        default="local",
        validation_alias=AliasChoices(
            "ISC_AI__INSTANCE_ID",
            "WEBSITE_INSTANCE_ID",     # App Service
            "CONTAINER_APP_REPLICA_NAME",
            "HOSTNAME",                # Kubernetes
        ),
    )


def part4_aliases() -> None:
    banner("PART 4 — AliasChoices for platform-injected variables")

    scenarios = [
        ("local", {}),
        ("App Service", {"WEBSITES_PORT": "8080",
                         "WEBSITE_INSTANCE_ID": "abc123def"}),
        ("Container Apps", {"PORT": "3000",
                            "CONTAINER_APP_REPLICA_NAME": "isc-rag--x7k2"}),
        ("explicit override wins", {"PORT": "3000", "ISC_AI__PORT": "9999"}),
    ]
    for label, overrides in scenarios:
        # Clear all platform vars first, THEN apply the scenario's. Passing
        # both as kwargs to one call collides when a scenario sets one of the
        # names being cleared.
        cleared = dict.fromkeys(
            ("WEBSITES_PORT", "PORT", "WEBSITE_INSTANCE_ID",
             "CONTAINER_APP_REPLICA_NAME"))
        with clean_env("ISC_AI__"), env(**{**cleared, **overrides}):
            s = PlatformAwareSettings()
            show(label, f"port={s.port} instance={s.instance_id}")

    print("""
    The same code runs unchanged on a laptop, App Service, and Container Apps.
    Order in AliasChoices is priority order, so an explicit ISC_AI__PORT beats
    whatever the platform injected — which is what you want for debugging.

    NOTE `extra="ignore"` here rather than "forbid". A settings class that
    reads platform variables cannot forbid extras, because the platform sets
    dozens of them. Use "forbid" on your own prefixed class and "ignore" on
    anything reading the platform's namespace.""")


# ---------------------------------------------------------------------------
# PART 5 — custom sources
# ---------------------------------------------------------------------------

class KeyVaultSource(PydanticBaseSettingsSource):
    """A custom settings source.

    In production this would call Key Vault via DefaultAzureCredential. Here
    it is a dict, so the mechanism is visible without network access.

    THE IMPORTANT PART IS ORDERING, not this class. See
    `settings_customise_sources` below — that classmethod is where you declare
    precedence, and it is the pydantic-settings equivalent of script 01's
    layer ordering.
    """

    def __init__(self, settings_cls: type[BaseSettings],
                 vault_values: dict[str, Any]) -> None:
        super().__init__(settings_cls)
        self.vault_values = vault_values
        self.fetch_count = 0

    def get_field_value(self, field: Any, field_name: str) -> tuple[Any, str, bool]:
        return self.vault_values.get(field_name), field_name, False

    def __call__(self) -> dict[str, Any]:
        self.fetch_count += 1
        return dict(self.vault_values)


class VaultBackedSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ISC_AI__", extra="ignore")

    deployment: str = "gpt-4o-mini"
    api_key: SecretStr | None = None
    index_name: str = "default-index"

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Declare source precedence. FIRST IN THE TUPLE WINS.

        The default order is:
            init > env > dotenv > file_secrets

        Here we insert a Key Vault source between env and dotenv, so:
            explicit kwargs  >  env vars  >  KEY VAULT  >  .env  >  secrets dir

        WHY KEY VAULT BELOW ENV: an operator setting an environment variable
        during an incident must be able to override the vault without a vault
        write. Same reasoning as CLI-args-win in script 01.

        `file_secret_settings` reads /run/secrets/<field> — the Docker and
        Kubernetes secrets-as-files convention, and a genuinely good one
        because a file-mounted secret never appears in `env` output or in a
        process listing.
        """
        vault = KeyVaultSource(settings_cls, {"api_key": "kv-injected-secret"})
        return (init_settings, env_settings, vault, dotenv_settings,
                file_secret_settings)


def part5_custom_sources() -> None:
    banner("PART 5 — custom source ordering")

    with clean_env("ISC_AI__"):
        s = VaultBackedSettings()
        show("api_key from the vault source", s.api_key)

    with clean_env("ISC_AI__"), env(ISC_AI__API_KEY="env-wins"):
        s2 = VaultBackedSettings()
        show("env var overrides the vault", s2.api_key)

    print("""
    THE PRACTICAL WARNING about a Key Vault settings source: it runs ONCE, at
    construction. That is correct behaviour and it has two consequences.

      1. A rotated secret is not picked up until the process restarts. See
         script 06 for how to handle rotation without one.
      2. If the vault is unreachable at startup, construction fails and the
         process does not start — which is the RIGHT failure. A service that
         starts without its credentials only fails later, on a user's request,
         in a code path with worse error handling.

    Also: do not fetch secrets from Key Vault on every request. Reads are
    network calls and are rate limited; a busy service will get 429s from the
    vault, which is a confusing incident because nothing about it looks like
    a config problem.""")


# ---------------------------------------------------------------------------
# PART 6 — generating .env.example from the class
# ---------------------------------------------------------------------------

def generate_env_example(model: type[BaseSettings], prefix: str = "ISC_AI__",
                         path: str = "") -> str:
    """Emit a .env.example from the settings class itself.

    Because it is GENERATED, it cannot drift from what the service actually
    reads — which is the failure mode of every hand-maintained
    .env.example on earth.

    Wire this into CI: regenerate, diff against the committed file, fail if
    they differ. Then a new required setting cannot merge without its
    documentation.
    """
    lines: list[str] = []
    for name, f in model.model_fields.items():
        key = f"{prefix}{path}{name.upper()}"
        ann = f.annotation
        if isinstance(ann, type) and issubclass(ann, BaseSettings):
            lines.append(f"\n# --- {name} ---")
            lines.append(generate_env_example(ann, prefix, f"{path}{name.upper()}__"))
            continue
        if f.description:
            for chunk in f.description.split(". "):
                if chunk.strip():
                    lines.append(f"# {chunk.strip().rstrip('.')}.")
        required = f.is_required()
        default = "" if required else f.default
        if hasattr(default, "get_secret_value"):
            default = "<secret>"
        lines.append(f"{key}={default if default is not None else ''}"
                     + ("   # REQUIRED" if required else ""))
    return "\n".join(lines)


def part6_generate_docs() -> None:
    banner("PART 6 — generate .env.example from the class")

    text = generate_env_example(AppSettings)
    for line in text.splitlines()[:30]:
        print(f"    {line}")
    print("    ...")

    print("""
    Regenerate this in CI and diff it against the committed file. A new
    required setting then cannot merge without updating the example, and
    onboarding never hits "why does it not start" for a variable nobody
    documented.""")


def main() -> None:
    part1_nested()
    part2_error_quality()
    part3_cross_field()
    part4_aliases()
    part5_custom_sources()
    part6_generate_docs()

    banner("SUMMARY")
    print("""
  * One settings class is the whole configuration surface — parsing,
    validation, fail-fast, and documentation from one declaration.
  * Nest with `env_nested_delimiter="__"`; prefix everything.
  * `extra="forbid"` on your own namespace catches typos that would otherwise
    silently leave a default in place. `extra="ignore"` where you read the
    platform's namespace.
  * All errors reported at once, with the variable name. Each round trip
    through a deploy pipeline you avoid is 5-20 minutes.
  * `model_validator` for cross-field production rules: no static key in prod,
    no DEBUG logging in prod, no empty allowed_groups.
  * `computed_field` for derived values, so two settings cannot disagree.
  * `AliasChoices` for platform-injected variables you do not control.
  * `settings_customise_sources` declares precedence. Env above Key Vault, so
    an operator can override during an incident.
  * Generate .env.example from the class and diff it in CI.
""")


if __name__ == "__main__":
    main()
