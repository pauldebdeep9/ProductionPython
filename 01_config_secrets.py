"""Topic 1: Configuration and secrets.

Configuration enters an application from the outside, is converted to typed
values, and is validated before the application starts doing useful work:

    environment variables or a local .env file
                         |
                         v
                 Settings validation
                         |
              +----------+-----------+
              |                      |
              v                      v
       application starts     startup fails clearly

Run this tutorial with ``python 01_config_secrets.py``. It does not contact an
external service and does not require a real API key.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Literal

from pydantic import Field, SecretStr, ValidationError, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Validated configuration for a small AI service.

    Field names map to environment variables without regard to case, so
    ``max_concurrency`` reads ``MAX_CONCURRENCY``. Pydantic also converts string
    environment values such as ``"5"`` into the declared Python types.

    A local .env file is convenient for development and is gitignored. In
    production, secrets should normally be injected by the deployment platform
    or its secret manager, never committed to source control.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: Literal["dev", "prod"] = "dev"
    llm_model: str = "demo-model"
    llm_api_key: SecretStr | None = None
    max_concurrency: int = Field(default=5, gt=0)
    request_timeout_seconds: float = Field(default=30.0, gt=0)

    @model_validator(mode="after")
    def require_production_api_key(self) -> Settings:
        """Allow fake services in dev, but require credentials in prod."""
        if self.app_env == "prod" and self.llm_api_key is None:
            raise ValueError("LLM_API_KEY is required when APP_ENV=prod")
        return self


class EnvironmentOnlySettings(Settings):
    """Settings variant that keeps deterministic demos isolated from ``.env``."""

    model_config = SettingsConfigDict(env_file=None, extra="ignore")


CONFIG_ENV_NAMES = {
    "APP_ENV",
    "LLM_MODEL",
    "LLM_API_KEY",
    "MAX_CONCURRENCY",
    "REQUEST_TIMEOUT_SECONDS",
}


def load_settings() -> Settings:
    """Load real startup configuration from the environment and optional .env."""
    return Settings()


@contextmanager
def controlled_environment(values: Mapping[str, str]) -> Iterator[None]:
    """Temporarily isolate tutorial configuration from the user's shell.

    Applications normally call ``load_settings`` directly. The demonstrations
    use this helper so an existing APP_ENV or LLM_API_KEY cannot change their
    outcome. Every previous value is restored, even if validation raises.
    """
    relevant_keys = {
        key for key in os.environ if key.upper() in CONFIG_ENV_NAMES
    } | CONFIG_ENV_NAMES
    previous_values = {
        key: os.environ[key] for key in relevant_keys if key in os.environ
    }

    try:
        for key in relevant_keys:
            os.environ.pop(key, None)
        os.environ.update(values)
        yield
    finally:
        for key in relevant_keys:
            os.environ.pop(key, None)
        os.environ.update(previous_values)


def show_safe_settings(settings: Settings) -> None:
    """Display operationally useful configuration without revealing secrets."""
    print(f"Environment: {settings.app_env}")
    print(f"Model: {settings.llm_model}")
    print(f"Max concurrency: {settings.max_concurrency}")
    print(f"Request timeout: {settings.request_timeout_seconds:.1f}s")
    print(f"API key configured: {'yes' if settings.llm_api_key else 'no'}")


def show_validation_failure(error: ValidationError) -> None:
    """Print concise errors without echoing input values that may be sensitive."""
    for detail in error.errors(include_input=False, include_url=False):
        location = ".".join(str(part) for part in detail["loc"]) or "settings"
        print(f"Rejected {location}: {detail['msg']}")


def demo_valid_configuration() -> Settings:
    """Load typed development settings from controlled environment variables."""
    print("\nExample A - valid development configuration")
    environment = {
        "APP_ENV": "dev",
        "LLM_MODEL": "example-model",
        "LLM_API_KEY": "tutorial-placeholder-not-a-real-key",
        "MAX_CONCURRENCY": "5",
        "REQUEST_TIMEOUT_SECONDS": "30",
    }
    with controlled_environment(environment):
        settings = EnvironmentOnlySettings()

    show_safe_settings(settings)
    print(f"Typed concurrency value: {type(settings.max_concurrency).__name__}")
    return settings


def demo_secret_is_hidden(settings: Settings) -> None:
    """Show SecretStr's safe representation without extracting its value."""
    print("\nExample B - secret-safe output")
    print(f"Secret field: {settings.llm_api_key!r}")
    print("Use get_secret_value() only at the narrow external SDK boundary.")


def demo_invalid_configuration() -> None:
    """Reject malformed configuration before request processing begins."""
    print("\nExample C - invalid configuration fails immediately")
    environment = {
        "APP_ENV": "dev",
        "LLM_MODEL": "example-model",
        "MAX_CONCURRENCY": "0",
        "REQUEST_TIMEOUT_SECONDS": "30",
    }
    with controlled_environment(environment):
        try:
            EnvironmentOnlySettings()
        except ValidationError as error:
            show_validation_failure(error)
        else:  # This branch would indicate that the demonstration is broken.
            raise AssertionError("MAX_CONCURRENCY=0 should be rejected")


def demo_production_validation() -> None:
    """Require stronger configuration when the application runs in production."""
    print("\nExample D - production requires an API key")
    environment = {
        "APP_ENV": "prod",
        "LLM_MODEL": "production-model",
        "MAX_CONCURRENCY": "10",
        "REQUEST_TIMEOUT_SECONDS": "15",
    }
    with controlled_environment(environment):
        try:
            EnvironmentOnlySettings()
        except ValidationError as error:
            show_validation_failure(error)
        else:  # This branch would indicate that the production guard failed.
            raise AssertionError("production without LLM_API_KEY should be rejected")


def main() -> None:
    """Run four deterministic, offline configuration demonstrations."""
    print("External configuration -> validation -> start or fail fast")
    settings = demo_valid_configuration()
    demo_secret_is_hidden(settings)
    demo_invalid_configuration()
    demo_production_validation()


if __name__ == "__main__":
    main()
