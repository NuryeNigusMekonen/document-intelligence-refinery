from __future__ import annotations

from datetime import datetime, UTC
from typing import Any, Literal

from pydantic import BaseModel, Field


BBox = tuple[float, float, float, float]


class LanguageHint(BaseModel):
    language: str = "unknown"
    confidence: float = 0.0


class DocumentProfile(BaseModel):
    doc_id: str
    doc_name: str
    sha256: str
    origin_type: Literal["native_digital", "scanned_image", "mixed", "form_fillable"]
    layout_complexity: Literal["single_column", "multi_column", "table_heavy", "figure_heavy", "mixed"]
    language_hint: LanguageHint
    domain_hint: Literal["financial", "legal", "technical", "medical", "general"]
    estimated_extraction_cost: Literal["fast_text_sufficient", "needs_layout_model", "needs_vision_model"]
    page_count: int
    avg_char_density: float
    image_area_ratio: float
    whitespace_ratio: float


class ProvenanceRef(BaseModel):
    doc_name: str
    ref_type: Literal["pdf_bbox", "word_section", "markdown_lines", "excel_cells", "image_bbox"]
    page_number: int | None = None
    bbox: BBox | None = None
    section_path: list[str] | None = None
    line_range: tuple[int, int] | None = None
    sheet_name: str | None = None
    cell_range: str | None = None
    content_hash: str


class TextBlock(BaseModel):
    text: str
    bbox: BBox
    reading_order: int
    confidence: float = 1.0
    provenance: ProvenanceRef | None = None


class TableObject(BaseModel):
    bbox: BBox
    headers: list[str]
    rows: list[list[str]]
    reading_order: int
    confidence: float = 0.0
    provenance: ProvenanceRef | None = None


class FigureObject(BaseModel):
    bbox: BBox
    caption: str | None = None
    reading_order: int
    confidence: float = 0.0
    provenance: ProvenanceRef | None = None


class ExtractedPage(BaseModel):
    page_number: int
    width: float
    height: float
    blocks: list[TextBlock] = Field(default_factory=list)
    tables: list[TableObject] = Field(default_factory=list)
    figures: list[FigureObject] = Field(default_factory=list)
    quality: dict[str, float] | None = None
    unresolved_needs_vision: bool = False


class ExtractedDocument(BaseModel):
    doc_id: str
    doc_name: str
    pages: list[ExtractedPage]


class LogicalDocumentUnit(BaseModel):
    ldu_id: str
    chunk_type: Literal["paragraph", "table", "figure", "list", "section_summary", "fact"]
    content: str | None = None
    structured_payload: dict[str, Any] | None = None
    token_count: int
    bounding_box: BBox | None = None
    parent_section: str | None = None
    parent_section_path: list[str] = Field(default_factory=list)
    page_refs: list[ProvenanceRef]
    content_hash: str
    relationships: list[str] = Field(default_factory=list)


class SectionNode(BaseModel):
    title: str
    page_start: int
    page_end: int
    child_sections: list["SectionNode"] = Field(default_factory=list)
    key_entities: list[str] = Field(default_factory=list)
    summary: str
    data_types_present: list[Literal["tables", "figures", "equations", "lists"]] = Field(default_factory=list)


class PageIndex(BaseModel):
    doc_id: str
    doc_name: str
    root_sections: list[SectionNode]


class LedgerEntry(BaseModel):
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    doc_id: str
    origin_type: str | None = None
    layout_complexity: str | None = None
    strategy_used: str
    confidence_score: float
    cost_estimate: float
    processing_time_ms: int
    pages_processed: int | None = None
    blocks_extracted: int | None = None
    tables_extracted: int | None = None
    table_pages_count: int | None = None
    ocr_word_conf_mean: float | None = None
    ethiopic_ratio: float | None = None
    garbage_ratio: float | None = None
    avg_word_len: float | None = None
    alpha_ratio: float | None = None
    token_diversity: float | None = None
    escalations: list[str] = Field(default_factory=list)
    notes: str = ""
    detected_language: str | None = None
    ocr_lang_used: str | None = None


class QueryAnswer(BaseModel):
    answer: str
    provenance_chain: list[ProvenanceRef]
    confidence: float
