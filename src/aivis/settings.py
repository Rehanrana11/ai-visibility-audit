from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    ANTHROPIC_API_KEY: str | None = None
    OPENAI_API_KEY: str | None = None
    GOOGLE_API_KEY: str | None = None
    XAI_API_KEY: str | None = None

    def has_key(self, provider: str) -> bool:
        key_map = {
            "anthropic": self.ANTHROPIC_API_KEY,
            "openai": self.OPENAI_API_KEY,
            "google": self.GOOGLE_API_KEY,
            "grok": self.XAI_API_KEY,
        }
        val = key_map.get(provider)
        return val is not None and len(val) > 0

    def get_key(self, provider: str) -> str:
        key_map = {
            "anthropic": self.ANTHROPIC_API_KEY,
            "openai": self.OPENAI_API_KEY,
            "google": self.GOOGLE_API_KEY,
            "grok": self.XAI_API_KEY,
        }
        val = key_map.get(provider)
        if not val:
            raise RuntimeError(
                f"{provider.upper()} API key not set. "
                f"Add {provider.upper()}_API_KEY to .env or environment."
            )
        return val


settings = Settings()