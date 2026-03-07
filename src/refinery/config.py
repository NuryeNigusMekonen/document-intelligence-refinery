from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="REFINERY_", extra="ignore")

    workspace_root: Path = Field(default=Path.cwd())
    artifacts_dir: Path = Field(default=Path(".refinery"))
    runtime_rules_file: Path = Field(default=Path("rubric/extraction_rules.yaml"))
    max_cost_per_doc: float = Field(default=2.5)
    enable_vision: bool = Field(default=False)
    layout_engine: str = Field(default="default")
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
    use_ollama_summaries: bool = Field(default=True)
    ollama_model: str = Field(default="llama3.1:8b")
    ollama_host: str = Field(default="http://127.0.0.1:11434")
    use_ollama_answers: bool = Field(default=True)
    ollama_answer_model: str = Field(default="")
    ollama_max_context_chars: int = Field(default=12000)
    query_use_langgraph: bool = Field(default=True)
    query_semantic_top_k: int = Field(default=5)

    triage_low_char_page_threshold: int = Field(default=80)
    triage_form_fillable_min_fields: int = Field(default=1)
    triage_scanned_image_min_image_ratio: float = Field(default=0.75)
    triage_scanned_image_max_char_density: float = Field(default=0.00008)
    triage_mixed_min_image_ratio: float = Field(default=0.20)
    triage_mixed_max_char_density: float = Field(default=0.0002)
    triage_mixed_mode_pages_fraction_threshold: float = Field(default=0.25)
    triage_table_heavy_ratio: float = Field(default=0.25)
    triage_figure_heavy_image_ratio: float = Field(default=0.50)
    triage_multi_column_min_columns: int = Field(default=2)
    triage_multi_column_fraction: float = Field(default=0.45)
    triage_layout_mixed_signal_count: int = Field(default=2)
    triage_vision_max_char_density: float = Field(default=0.0001)
    triage_vision_min_image_ratio: float = Field(default=0.60)
    triage_layout_low_char_pages_fraction: float = Field(default=0.30)
    router_budget_min_vision_headroom: float = Field(default=0.08)
    router_domain_risk_boost_high: float = Field(default=0.06)
    router_domain_risk_boost_medium: float = Field(default=0.03)
    router_domain_risk_confidence_gate: float = Field(default=0.65)

    def model_post_init(self, __context) -> None:
        if not self.ollama_answer_model:
            self.ollama_answer_model = self.ollama_model

    @property
    def resolved_artifacts_dir(self) -> Path:
        if self.artifacts_dir.is_absolute():
            return self.artifacts_dir
        return (self.workspace_root / self.artifacts_dir).resolve()
