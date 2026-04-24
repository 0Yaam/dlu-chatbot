from functools import lru_cache
from pathlib import Path

from pydantic import AnyHttpUrl, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables or `.env`."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    token: str = Field(..., validation_alias="TOKEN")
    webhook_url: AnyHttpUrl = Field(..., validation_alias="WEBHOOK_URL")
    groq_api_key: str = Field(..., validation_alias="GROQ_API_KEY")
    groq_model: str = Field("llama-3.1-8b-instant", validation_alias="GROQ_MODEL")
    chroma_collection: str = Field("dlu_documents", validation_alias="CHROMA_COLLECTION")
    chroma_persist_dir: Path = Field(Path("vector_store"), validation_alias="CHROMA_PERSIST_DIR")
    embedding_model: str = Field(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        validation_alias="EMBEDDING_MODEL",
    )
    embedding_device: str = Field("cpu", validation_alias="EMBEDDING_DEVICE")

    @property
    def telegram_webhook_url(self) -> str:
        """Expose the public Telegram webhook endpoint."""
        return f"{str(self.webhook_url).rstrip('/')}/webhook"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
