from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    OPENAI_API_KEY: str = Field(..., description="The API key for OpenAI")
    OPENAI_BASE_URL: str = Field(..., description="The Base URL for the OpenAI API")
    MODEL_NAME: str = Field("gpt-4o", description="The model to use")
    CHROMA_DB_DIR: str = Field(
        "data/chroma_db", description="Path to ChromaDB persistence directory"
    )
    CHROMA_HOST: str | None = Field(
        None, description="ChromaDB server host for HTTP client mode"
    )
    CHROMA_PORT: int = Field(
        8000, description="ChromaDB server port for HTTP client mode"
    )
    CORPUS_DIR: str = Field("data/corpus", description="Path to document corpus")
    CORPUS_LARGE_DIR: str = Field(
        "data/corpus_large", description="Path to large document corpus"
    )

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


# We do not instantiate 'settings' globally here to avoid
# crashing the app on import if variables are missing.
# Users should instantiate Settings() where needed or use a dependency injection pattern.
