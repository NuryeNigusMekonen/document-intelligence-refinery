from __future__ import annotations

from datetime import datetime, UTC
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_serializer, model_validator


class BBox(BaseModel):
    x0: float = Field(ge=0)
    y0: float = Field(ge=0)
    x1: float = Field(ge=0)
    y1: float = Field(ge=0)

    @model_validator(mode="before")
    @classmethod
    def _coerce_sequence(cls, value: Any) -> Any:
        if isinstance(value, (tuple, list)) and len(value) == 4:
            return {"x0": float(value[0]), "y0": float(value[1]), "x1": float(value[2]), "y1": float(value[3])}
        return value

    @model_validator(mode="after")
    def _validate_geometry(self) -> "BBox":
        if self.x1 <= self.x0:
            raise ValueError("bbox x1 must be greater than x0")
        if self.y1 <= self.y0:
            raise ValueError("bbox y1 must be greater than y0")
        return self

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.x0, self.y0, self.x1, self.y1)

    def __getitem__(self, index: int) -> float:
        return self.as_tuple()[index]

    def __len__(self) -> int:
        return 4

    def __eq__(self, other: object) -> bool:
        if isinstance(other, BBox):
            return self.as_tuple() == other.as_tuple()
        if isinstance(other, (tuple, list)) and len(other) == 4:
            try:
                return self.as_tuple() == tuple(float(v) for v in other)
            except (TypeError, ValueError):
                return False
        return False

    @model_serializer(mode="plain")
    def _serialize(self) -> list[float]:
        return [self.x0, self.y0, self.x1, self.y1]


class LanguageHint(BaseModel):
    language: str = "unknown"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class DocumentProfile(BaseModel):
    doc_id: str
    doc_name: str
    sha256: str
    origin_type: Literal["native_digital", "scanned_image", "mixed", "form_fillable"]
    layout_complexity: Literal["single_column", "multi_column", "table_heavy", "figure_heavy", "mixed"]
    language_hint: LanguageHint
    domain_hint: Literal["financial", "legal", "technical", "medical", "general"]
    estimated_extraction_cost: Literal["fast_text_sufficient", "needs_layout_model", "needs_vision_model"]
    origin_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    layout_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    domain_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    extraction_cost_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    page_count: int = Field(ge=1)
    avg_char_density: float = Field(ge=0.0)
    image_area_ratio: float = Field(ge=0.0, le=1.0)
    whitespace_ratio: float = Field(ge=0.0, le=1.0)


class ProvenanceRef(BaseModel):
    doc_name: str = Field(min_length=1)
    ref_type: Literal["pdf_bbox", "word_section", "markdown_lines", "excel_cells", "image_bbox"]
    page_number: int | None = Field(default=None, ge=1)
    bbox: BBox | tuple[float, float, float, float] | None = None
    section_path: list[str] | None = None
    line_range: tuple[int, int] | None = None
    sheet_name: str | None = None
    cell_range: str | None = None
    content_hash: str

    @field_validator("line_range")
    @classmethod
    def _validate_line_range(cls, value: tuple[int, int] | None) -> tuple[int, int] | None:
        if value is None:
            return value
        start, end = value
        if start < 1 or end < 1:
            raise ValueError("line_range values must be >= 1")
        if end < start:
            raise ValueError("line_range end must be >= start")
        return value

    @field_validator("bbox", mode="before")
    @classmethod
    def _coerce_bbox(cls, value: BBox | tuple[float, float, float, float] | None) -> BBox | None:
        if value is None or isinstance(value, BBox):
            return value
        return BBox.model_validate(value)

    @field_validator("section_path")
    @classmethod
    def _validate_section_path(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return value
        cleaned = [s.strip() for s in value if s and s.strip()]
        return cleaned or None

    @model_validator(mode="after")
    def _validate_ref_type_payload(self) -> "ProvenanceRef":
        if self.ref_type in {"pdf_bbox", "image_bbox"} and self.bbox is None:
            raise ValueError(f"{self.ref_type} provenance requires bbox")
        if self.ref_type == "word_section" and not self.section_path:
            raise ValueError("word_section provenance requires section_path")
        if self.ref_type == "markdown_lines" and self.line_range is None:
            raise ValueError("markdown_lines provenance requires line_range")
        if self.ref_type == "excel_cells" and (not self.sheet_name or not self.cell_range):
            raise ValueError("excel_cells provenance requires sheet_name and cell_range")
        return self


class ProvenanceChain(BaseModel):
    steps: list[ProvenanceRef] = Field(min_length=1)
    chain_type: Literal["single_source", "aggregated", "multi_hop"] = "single_source"

    @classmethod
    def from_refs(cls, refs: list[ProvenanceRef]) -> "ProvenanceChain":
        kind: Literal["single_source", "aggregated", "multi_hop"] = "single_source" if len(refs) == 1 else "aggregated"
        return cls(steps=refs, chain_type=kind)

    @model_validator(mode="after")
    def _validate_chain_shape(self) -> "ProvenanceChain":
        if self.chain_type == "single_source" and len(self.steps) != 1:
            raise ValueError("single_source provenance_chain must contain exactly one step")
        if self.chain_type in {"aggregated", "multi_hop"} and len(self.steps) < 2:
            raise ValueError(f"{self.chain_type} provenance_chain must contain at least two steps")
        return self


class TextBlock(BaseModel):
    text: str
    bbox: BBox | tuple[float, float, float, float]
    reading_order: int = Field(ge=0)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    confidence_signals: list["ConfidenceSignal"] = Field(default_factory=list)
    provenance: ProvenanceRef | None = None

    @field_validator("bbox", mode="before")
    @classmethod
    def _coerce_bbox(cls, value: BBox | tuple[float, float, float, float]) -> BBox:
        if isinstance(value, BBox):
            return value
        return BBox.model_validate(value)


class TableObject(BaseModel):
    bbox: BBox | tuple[float, float, float, float]
    headers: list[str]
    rows: list[list[str]]
    reading_order: int = Field(ge=0)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    confidence_signals: list["ConfidenceSignal"] = Field(default_factory=list)
    provenance: ProvenanceRef | None = None

    @field_validator("bbox", mode="before")
    @classmethod
    def _coerce_bbox(cls, value: BBox | tuple[float, float, float, float]) -> BBox:
        if isinstance(value, BBox):
            return value
        return BBox.model_validate(value)


class FigureObject(BaseModel):
    bbox: BBox | tuple[float, float, float, float]
    caption: str | None = None
    reading_order: int = Field(ge=0)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    confidence_signals: list["ConfidenceSignal"] = Field(default_factory=list)
    provenance: ProvenanceRef | None = None

    @field_validator("bbox", mode="before")
    @classmethod
    def _coerce_bbox(cls, value: BBox | tuple[float, float, float, float]) -> BBox:
        if isinstance(value, BBox):
            return value
        return BBox.model_validate(value)


class ConfidenceSignal(BaseModel):
    signal: str = Field(min_length=1)
    value: float
    normalized_value: float | None = Field(default=None, ge=0.0, le=1.0)
    weight: float | None = Field(default=None, ge=0.0)
    threshold: float | None = None
    passed: bool | None = None


class ExtractedPage(BaseModel):
    page_number: int = Field(ge=1)
    width: float = Field(gt=0)
    height: float = Field(gt=0)
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
    token_count: int = Field(ge=0)
    bounding_box: BBox | tuple[float, float, float, float] | None = None
    parent_section: str | None = None
    parent_section_path: list[str] = Field(default_factory=list)
    page_refs: list[ProvenanceRef] = Field(min_length=1)
    provenance_chain: ProvenanceChain | None = None
    content_hash: str
    relationships: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _ensure_provenance_chain(self) -> "LogicalDocumentUnit":
        if self.provenance_chain is None and self.page_refs:
            self.provenance_chain = ProvenanceChain.from_refs(self.page_refs)
        return self

    @field_validator("bounding_box", mode="before")
    @classmethod
    def _coerce_bounding_box(cls, value: BBox | tuple[float, float, float, float] | None) -> BBox | None:
        if value is None or isinstance(value, BBox):
            return value
        return BBox.model_validate(value)


LDU = LogicalDocumentUnit


class SectionNode(BaseModel):
    title: str
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    child_sections: list["SectionNode"] = Field(default_factory=list)
    key_entities: list[str] = Field(default_factory=list)
    summary: str
    data_types_present: list[Literal["tables", "figures", "equations", "lists"]] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_page_range(self) -> "SectionNode":
        if self.page_end < self.page_start:
            raise ValueError("page_end must be >= page_start")
        return self


class PageIndex(BaseModel):
    doc_id: str
    doc_name: str
    root_sections: list[SectionNode]


class RoutingPolicyContext(BaseModel):
    domain_hint: str
    domain_confidence: float = Field(ge=0.0, le=1.0)
    max_cost_per_doc: float = Field(ge=0.0)
    budget_spent: float = Field(ge=0.0)
    budget_headroom: float
    min_vision_headroom: float = Field(ge=0.0)
    layout_confidence_floor: float = Field(ge=0.0, le=1.0)
    vision_confidence_floor: float = Field(ge=0.0, le=1.0)
    domain_risk_boost: float = Field(ge=0.0, le=1.0)
    vision_allowed: bool = True
    vision_policy_reason: str = ""


class RoutingAttempt(BaseModel):
    stage: str = Field(min_length=1)
    strategy: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    cost_delta: float = Field(ge=0.0)
    cumulative_cost: float = Field(ge=0.0)
    notes: str = ""


class PageStrategyHistory(BaseModel):
    page_number: int = Field(ge=1)
    attempted_strategies: list[str] = Field(default_factory=list)
    final_strategy: str = Field(min_length=1)
    unresolved_needs_vision: bool = False
    block_count: int = Field(ge=0)
    table_count: int = Field(ge=0)
    figure_count: int = Field(ge=0)
    ocr_word_conf_mean: float | None = None


class LedgerEntry(BaseModel):
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    doc_id: str
    origin_type: str | None = None
    layout_complexity: str | None = None
    strategy_used: str
    confidence_score: float = Field(ge=0.0, le=1.0)
    cost_estimate: float = Field(ge=0.0)
    processing_time_ms: int = Field(ge=0)
    pages_processed: int | None = Field(default=None, ge=0)
    blocks_extracted: int | None = Field(default=None, ge=0)
    tables_extracted: int | None = Field(default=None, ge=0)
    table_pages_count: int | None = Field(default=None, ge=0)
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
    policy_context: RoutingPolicyContext | None = None
    routing_attempts: list[RoutingAttempt] = Field(default_factory=list)
    page_strategy_history: list[PageStrategyHistory] = Field(default_factory=list)


class QueryAnswer(BaseModel):
    answer: str
    provenance_chain: ProvenanceChain
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("provenance_chain", mode="before")
    @classmethod
    def _coerce_legacy_provenance_chain(cls, value: Any) -> Any:
        if isinstance(value, list):
            refs = [ProvenanceRef.model_validate(v) for v in value]
            return ProvenanceChain.from_refs(refs)
        return value
