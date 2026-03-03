from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="REFINERY_", extra="ignore")

    workspace_root: Path = Field(default=Path.cwd())
    artifacts_dir: Path = Field(default=Path(".refinery"))
    max_cost_per_doc: float = Field(default=2.5)
    enable_vision: bool = Field(default=False)
    openrouter_api_key: str | None = Field(default=None)
    vector_top_k: int = Field(default=5)
    chunk_max_tokens: int = Field(default=350)
    deterministic_seed: int = Field(default=7)
    ocr_enabled: bool = Field(default=True)
    ocr_engine: str = Field(default="tesseract")
    ocr_lang_default: str = Field(default="eng")
    ocr_lang_fallback: str = Field(default="eng+amh")
    ocr_amharic_enabled: bool = Field(default=True)
    embedding_model: str = Field(default="multilingual-lexical")
    multilingual_embeddings: bool = Field(default=True)
    language_detection_mode: Literal["script", "langdetect"] = Field(default="script")

    @property
    def resolved_artifacts_dir(self) -> Path:
        if self.artifacts_dir.is_absolute():
            return self.artifacts_dir
        return (self.workspace_root / self.artifacts_dir).resolve()
