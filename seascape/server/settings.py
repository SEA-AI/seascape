"""The server's settings."""

import re
from pathlib import Path
from typing import Annotated

from pydantic import (
    AnyUrl,
    PositiveFloat,
    PositiveInt,
    SecretStr,
    UrlConstraints,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

LOCAL_HOSTS = ("localhost:*", "127.0.0.1:*", "[::1]:*")


class Settings(BaseSettings):
    """The scenarios served, who may reach them, and what a job may ask of this host."""

    # Compose passes a variable set empty in .env as "". nan would pass every
    # comparison and turn a limit off.
    model_config = SettingsConfigDict(
        env_prefix="SEASCAPE_",
        env_ignore_empty=True,
        allow_inf_nan=False,
        frozen=True,
        # An error otherwise echoes the rejected input, a secret among them.
        hide_input_in_errors=True,
    )

    scenarios: Path
    allowed_hosts: Annotated[tuple[str, ...], NoDecode] = LOCAL_HOSTS
    max_duration_s: PositiveFloat = 60.0
    max_pixels: PositiveInt = 3840 * 2160
    max_images: PositiveInt = 10_000
    min_free_gb: PositiveFloat = 10.0
    max_run_s: PositiveFloat = 12 * 3600.0
    max_queued: PositiveInt = 20
    # Judgement: more seeds than a client reviews.
    max_variants: PositiveInt = 100
    # Judgement: bounds a frame's render time.
    max_samples: PositiveInt = 1024
    # The path is the credential: an MCP client may send no header.
    mcp_secret: SecretStr | None = None
    # The secret travels in the path, in the clear over http.
    public_url: Annotated[AnyUrl, UrlConstraints(allowed_schemes=["https"])] | None = (
        None
    )

    @field_validator("allowed_hosts", mode="before")
    @classmethod
    def _listed(cls, value: object) -> object:
        if isinstance(value, str):
            return (
                tuple(h for part in value.split(",") if (h := part.strip()))
                or LOCAL_HOSTS
            )
        return value

    @model_validator(mode="after")
    def _secret_guards_the_path(self) -> "Settings":
        if self.public_url and not self.mcp_secret:
            raise ValueError("SEASCAPE_PUBLIC_URL needs SEASCAPE_MCP_SECRET")
        # The only credential on a public URL, and a route path, where `/` and `{}`
        # mean something else. Judgement: enough entropy to resist guessing.
        if self.mcp_secret and not re.fullmatch(
            r"[A-Za-z0-9_-]{32,}", self.mcp_secret.get_secret_value()
        ):
            raise ValueError(
                "SEASCAPE_MCP_SECRET needs 32 or more of A-Z, a-z, 0-9, _ and -"
            )
        return self
