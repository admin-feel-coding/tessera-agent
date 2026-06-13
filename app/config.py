from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    anthropic_api_key: str = "placeholder-replace-me"
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"
    tessera_data_url: str = "http://localhost:8002"
    internal_api_key: str
    port: int = 8001
    log_level: str = "INFO"
    analyze_timeout_seconds: float = 30.0
    max_analyze_retries: int = 2


settings = Settings()
