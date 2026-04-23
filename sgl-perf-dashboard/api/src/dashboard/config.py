"""Runtime configuration from environment variables."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Environment-driven settings. See .env.example for all keys."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # MinIO / S3
    minio_endpoint: str = Field(..., description="Full URL to the MinIO S3 API")
    minio_access_key: str = Field(..., description="Read-only access key")
    minio_secret_key: str = Field(..., description="Read-only secret key")
    minio_bucket: str = Field(default="sglang-ci-logs")
    minio_region: str = Field(default="us-east-1")

    # GitHub (optional; enrichment degrades if missing)
    github_token: str = Field(default="")
    github_repo: str = Field(default="sgl-project/sglang")

    # Anthropic (M3+; unused in M1)
    anthropic_api_key: str = Field(default="")

    # Storage
    db_path: str = Field(default="/data/dashboard.db")
    retention_days: int = Field(default=90, ge=1, le=365)

    # Ingester
    ingester_interval_seconds: int = Field(default=300, ge=30, le=3600)

    # Reconciler — mount scripts/ci/slurm/nightly-configs.yaml into the container
    # at this path so the reconciler knows the expected matrix per workflow run.
    nightly_configs_path: str = Field(default="/app/nightly-configs.yaml")
    nightly_runner: str = Field(default="gb200")
    reconcile_window_days: int = Field(default=30, ge=1, le=365)

    # API server
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000, ge=1, le=65535)

    @property
    def github_enrichment_enabled(self) -> bool:
        return bool(self.github_token)


# Singleton access — import this from other modules.
settings = Settings()  # type: ignore[call-arg]
