from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    OPENAI_API_KEY: str = Field(..., description="The API key for OpenAI")
    MODEL_NAME: str = Field("gpt-4o", description="The model to use")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


# We do not instantiate 'settings' globally here to avoid
# crashing the app on import if variables are missing.
# Users should instantiate Settings() where needed or use a dependency injection pattern.
