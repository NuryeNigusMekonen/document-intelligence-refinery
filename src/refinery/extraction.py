from __future__ import annotations

import base64
import json
import logging
import re
import time
from abc import ABC, abstractmethod
from pathlib import Path
from statistics import median
from typing import Callable
from urllib import request

import fitz
import pdfplumber
from PIL import Image

from .adapters import DoclingAdapter
from .config import Settings
from .lang_detect import detect_language, select_ocr_lang
from .models import (
    ConfidenceSignal,
    DocumentProfile,
    ExtractedDocument,
    ExtractedPage,
    FigureObject,
    LedgerEntry,
    PageStrategyHistory,
    ProvenanceRef,
    RoutingAttempt,
    RoutingPolicyContext,
    TableObject,
    TextBlock,
)
from .runtime_rules import load_runtime_rules
from .storage import ArtifactStore

logger = logging.getLogger(__name__)


ExtractionResult = tuple[ExtractedDocument, float, float, str]
LayoutEngineResult = tuple[ExtractedDocument, float, float, str, str]
VLMProviderFn = Callable[[Path, DocumentProfile], ExtractionResult | None]
LayoutEngineFn = Callable[[Path, DocumentProfile], LayoutEngineResult]


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _signal(
    signal: str,
    value: float,
    *,
    normalized: float | None = None,
    weight: float | None = None,
    threshold: float | None = None,
    passed: bool | None = None,
) -> ConfidenceSignal:
    return ConfidenceSignal(
        signal=signal,
        value=float(value),
        normalized_value=None if normalized is None else _clamp01(normalized),
        weight=weight,
        threshold=threshold,
        passed=passed,
    )


class BaseExtractor(ABC):
    name = "base"

    @abstractmethod
    def extract(self, pdf_path: Path, profile: DocumentProfile) -> tuple[ExtractedDocument, float, float, str]:
        raise NotImplementedError


def _confidence_from_profile(profile: DocumentProfile) -> float:
    base = 0.9
    base -= min(profile.image_area_ratio, 1.0) * 0.5
    base -= min(profile.whitespace_ratio, 1.0) * 0.2
    if profile.estimated_extraction_cost == "needs_vision_model":
        base -= 0.25
    elif profile.estimated_extraction_cost == "needs_layout_model":
        base -= 0.10
    return max(0.05, min(0.99, base))


def _as_bbox(obj: dict, fallback: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    return (
        float(obj.get("x0", fallback[0])),
        float(obj.get("top", obj.get("y0", fallback[1]))),
        float(obj.get("x1", fallback[2])),
        float(obj.get("bottom", obj.get("y1", fallback[3]))),
    )


class FastTextExtractor(BaseExtractor):
    name = "fast_text"

    def __init__(self, min_chars_per_page: float = 100.0, max_image_area_ratio: float = 0.50):
        self.min_chars_per_page = min_chars_per_page
        self.max_image_area_ratio = max_image_area_ratio

    def extract(self, pdf_path: Path, profile: DocumentProfile) -> tuple[ExtractedDocument, float, float, str]:
        started = time.perf_counter()
        pages: list[ExtractedPage] = []
        pass_pages = 0
        total_pages = 0
        with pdfplumber.open(pdf_path) as pdf, fitz.open(pdf_path) as mu_pdf:
            for pidx, page in enumerate(pdf.pages, start=1):
                total_pages += 1
                words = page.extract_words() or []
                text = page.extract_text() or ""
                char_count = len(text)
                mu_page = mu_pdf[pidx - 1]
                page_area = max(float(page.width) * float(page.height), 1.0)
                img_area = 0.0
                for img in mu_page.get_images(full=True):
                    xref = img[0]
                    rects = mu_page.get_image_rects(xref)
                    img_area += sum(r.width * r.height for r in rects)
                image_ratio = img_area / page_area
                if char_count > self.min_chars_per_page and image_ratio < self.max_image_area_ratio:
                    pass_pages += 1

                blocks: list[TextBlock] = []
                for i, w in enumerate(words):
                    text = (w.get("text") or "").strip()
                    if not text:
                        continue
                    word_bbox = _as_bbox(w, (0.0, 0.0, page.width, page.height))
                    blocks.append(
                        TextBlock(
                            text=text,
                            bbox=word_bbox,
                            reading_order=i,
                            confidence=0.85,
                            confidence_signals=[
                                _signal("extractor_prior", 0.85, normalized=0.85, weight=0.50),
                                _signal(
                                    "page_char_density_gate",
                                    float(char_count),
                                    normalized=min(float(char_count) / max(self.min_chars_per_page, 1.0), 1.0),
                                    weight=0.30,
                                    threshold=self.min_chars_per_page,
                                    passed=char_count > self.min_chars_per_page,
                                ),
                                _signal(
                                    "page_image_ratio_gate",
                                    float(image_ratio),
                                    normalized=1.0 - min(float(image_ratio) / max(self.max_image_area_ratio, 1e-6), 1.0),
                                    weight=0.20,
                                    threshold=self.max_image_area_ratio,
                                    passed=image_ratio < self.max_image_area_ratio,
                                ),
                            ],
                            provenance=ProvenanceRef(
                                doc_name=profile.doc_name,
                                ref_type="pdf_bbox",
                                page_number=pidx,
                                bbox=word_bbox,
                                content_hash="pending",
                            ),
                        )
                    )

                tables: list[TableObject] = []
                for t_idx, table in enumerate(page.extract_tables() or []):
                    if not table:
                        continue
                    headers = [str(c or "").strip() for c in table[0]] if table else []
                    rows = [[str(c or "").strip() for c in row] for row in table[1:]] if len(table) > 1 else []
                    table_bbox = (0.0, 0.0, page.width, page.height)
                    tables.append(
                        TableObject(
                            bbox=table_bbox,
                            headers=headers,
                            rows=rows,
                            reading_order=10000 + t_idx,
                            confidence=0.60,
                            confidence_signals=[
                                _signal("extractor_prior", 0.60, normalized=0.60, weight=0.60),
                                _signal(
                                    "table_row_count",
                                    float(len(rows)),
                                    normalized=min(float(len(rows)) / 6.0, 1.0),
                                    weight=0.25,
                                ),
                                _signal(
                                    "header_presence",
                                    float(1.0 if headers else 0.0),
                                    normalized=1.0 if headers else 0.0,
                                    weight=0.15,
                                    threshold=1.0,
                                    passed=bool(headers),
                                ),
                            ],
                            provenance=ProvenanceRef(
                                doc_name=profile.doc_name,
                                ref_type="pdf_bbox",
                                page_number=pidx,
                                bbox=table_bbox,
                                content_hash="pending",
                            ),
                        )
                    )

                figures: list[FigureObject] = []
                pages.append(
                    ExtractedPage(
                        page_number=pidx,
                        width=page.width,
                        height=page.height,
                        blocks=blocks,
                        tables=tables,
                        figures=figures,
                    )
                )

        extracted = ExtractedDocument(doc_id=profile.doc_id, doc_name=profile.doc_name, pages=pages)
        base = _confidence_from_profile(profile)
        pass_ratio = pass_pages / max(total_pages, 1)
        confidence = max(0.05, min(0.99, min(base, 0.30 + 0.70 * pass_ratio)))
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        notes = (
            f"pages={len(pages)} blocks={sum(len(p.blocks) for p in pages)} tables={sum(len(p.tables) for p in pages)} "
            f"fast_gate={pass_pages}/{max(total_pages, 1)} min_chars>{self.min_chars_per_page:.0f} "
            f"max_image_ratio<{self.max_image_area_ratio:.2f}"
        )
        return extracted, confidence, 0.0, notes + f" elapsed_ms={elapsed_ms}"


class LayoutLiteExtractor(BaseExtractor):
    name = "layout_lite"

    def extract(self, pdf_path: Path, profile: DocumentProfile) -> tuple[ExtractedDocument, float, float, str]:
        started = time.perf_counter()
        pages: list[ExtractedPage] = []
        with pdfplumber.open(pdf_path) as pdf, fitz.open(pdf_path) as mu_pdf:
            for pidx, page in enumerate(pdf.pages, start=1):
                mu_page = mu_pdf[pidx - 1]
                blocks_raw = mu_page.get_text("blocks")
                sorted_blocks = sorted(blocks_raw, key=lambda b: (round(float(b[1]) / 40), float(b[0])))
                blocks = [
                    TextBlock(
                        text=(b[4] or "").strip(),
                        bbox=(float(b[0]), float(b[1]), float(b[2]), float(b[3])),
                        reading_order=i,
                        confidence=0.80,
                        confidence_signals=[
                            _signal("layout_block_prior", 0.80, normalized=0.80, weight=0.70),
                            _signal(
                                "block_text_length",
                                float(len((b[4] or "").strip())),
                                normalized=min(float(len((b[4] or "").strip())) / 40.0, 1.0),
                                weight=0.30,
                            ),
                        ],
                        provenance=ProvenanceRef(
                            doc_name=profile.doc_name,
                            ref_type="pdf_bbox",
                            page_number=pidx,
                            bbox=(float(b[0]), float(b[1]), float(b[2]), float(b[3])),
                            content_hash="pending",
                        ),
                    )
                    for i, b in enumerate(sorted_blocks)
                    if str(b[4] or "").strip()
                ]

                tables: list[TableObject] = []
                for t_idx, table_obj in enumerate(page.find_tables()):
                    extracted = table_obj.extract() or []
                    if not extracted:
                        continue
                    headers = [str(c or "").strip() for c in extracted[0]] if extracted else []
                    rows = [[str(c or "").strip() for c in row] for row in extracted[1:]] if len(extracted) > 1 else []
                    bbox = (
                        float(table_obj.bbox[0]),
                        float(table_obj.bbox[1]),
                        float(table_obj.bbox[2]),
                        float(table_obj.bbox[3]),
                    )
                    tables.append(
                        TableObject(
                            bbox=bbox,
                            headers=headers,
                            rows=rows,
                            reading_order=9000 + t_idx,
                            confidence=0.82,
                            confidence_signals=[
                                _signal("layout_table_prior", 0.82, normalized=0.82, weight=0.60),
                                _signal(
                                    "table_row_count",
                                    float(len(rows)),
                                    normalized=min(float(len(rows)) / 10.0, 1.0),
                                    weight=0.25,
                                ),
                                _signal(
                                    "header_presence",
                                    float(1.0 if headers else 0.0),
                                    normalized=1.0 if headers else 0.0,
                                    weight=0.15,
                                    threshold=1.0,
                                    passed=bool(headers),
                                ),
                            ],
                            provenance=ProvenanceRef(
                                doc_name=profile.doc_name,
                                ref_type="pdf_bbox",
                                page_number=pidx,
                                bbox=bbox,
                                content_hash="pending",
                            ),
                        )
                    )

                figures: list[FigureObject] = []
                caption_candidates = [
                    b for b in blocks if b.text.strip().lower().startswith(("figure", "fig.", "fig "))
                ]
                figure_boxes: list[tuple[float, float, float, float]] = []
                for img in mu_page.get_images(full=True):
                    xref = img[0]
                    for rect in mu_page.get_image_rects(xref):
                        figure_boxes.append((float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1)))

                for f_idx, fig_bbox in enumerate(figure_boxes):
                    caption = None
                    best_gap = float("inf")
                    for candidate in caption_candidates:
                        candidate_box = candidate.bbox
                        if candidate_box[1] >= fig_bbox[1] - 120 and candidate_box[1] <= fig_bbox[3] + 120:
                            gap = min(abs(candidate_box[1] - fig_bbox[3]), abs(fig_bbox[1] - candidate_box[3]))
                            if gap < best_gap:
                                best_gap = gap
                                caption = candidate.text
                    figures.append(
                        FigureObject(
                            bbox=fig_bbox,
                            caption=caption,
                            reading_order=9500 + f_idx,
                            confidence=0.76,
                            confidence_signals=[
                                _signal("layout_figure_prior", 0.76, normalized=0.76, weight=0.65),
                                _signal(
                                    "caption_presence",
                                    float(1.0 if caption else 0.0),
                                    normalized=1.0 if caption else 0.0,
                                    weight=0.35,
                                    threshold=1.0,
                                    passed=bool(caption),
                                ),
                            ],
                            provenance=ProvenanceRef(
                                doc_name=profile.doc_name,
                                ref_type="pdf_bbox",
                                page_number=pidx,
                                bbox=fig_bbox,
                                content_hash="pending",
                            ),
                        )
                    )

                pages.append(
                    ExtractedPage(
                        page_number=pidx,
                        width=page.width,
                        height=page.height,
                        blocks=blocks,
                        tables=tables,
                        figures=figures,
                    )
                )

        extracted = ExtractedDocument(doc_id=profile.doc_id, doc_name=profile.doc_name, pages=pages)
        base = _confidence_from_profile(profile)
        confidence = min(0.95, base + 0.12)
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        notes = f"layout-lite pages={len(pages)} tables={sum(len(p.tables) for p in pages)} elapsed_ms={elapsed_ms}"
        return extracted, confidence, 0.10, notes


class VisionExtractor(BaseExtractor):
    name = "vision"

    def __init__(self, settings: Settings):
        self.settings = settings
        self.docling_adapter = DoclingAdapter()
        self.vlm_providers: dict[str, VLMProviderFn] = {}
        self.register_vlm_provider("openrouter", self._run_openrouter_vlm)

    def register_vlm_provider(self, name: str, provider: VLMProviderFn) -> None:
        key = str(name or "").strip().lower()
        if not key:
            return
        self.vlm_providers[key] = provider

    def _run_vlm_provider(self, provider_name: str, pdf_path: Path, profile: DocumentProfile) -> ExtractionResult | None:
        normalized = (provider_name or "").strip().lower()
        if normalized == "openrouter":
            provider = self._run_openrouter_vlm
        else:
            provider = self.vlm_providers.get(normalized)
        if provider is None:
            return None
        try:
            return provider(pdf_path, profile)
        except Exception:
            return None

    def _select_openrouter_model(self) -> str:
        if self.settings.max_cost_per_doc <= self.settings.vlm_low_budget_cap_usd:
            return self.settings.vlm_model_low_cost
        return self.settings.vlm_model_high_quality

    def _build_vlm_prompt(self, profile: DocumentProfile, page_number: int) -> str:
        return (
            "You are a document structure extraction engine. "
            "Extract text blocks, tables, and figures from this page image. "
            "Return strict JSON only with schema: "
            "{\"blocks\": [{\"text\": str, \"bbox\": [x0,y0,x1,y1], \"reading_order\": int, \"confidence\": float}], "
            "\"tables\": [{\"headers\": [str], \"rows\": [[str]], \"bbox\": [x0,y0,x1,y1], \"reading_order\": int, \"confidence\": float}], "
            "\"figures\": [{\"caption\": str|null, \"bbox\": [x0,y0,x1,y1], \"reading_order\": int, \"confidence\": float}], "
            "\"page_confidence\": float}. "
            f"Document language hint: {profile.language_hint.language} ({profile.language_hint.confidence:.2f}). "
            f"Domain hint: {profile.domain_hint}. Page number: {page_number}."
        )

    def _extract_json_payload(self, content: str) -> dict | None:
        text = (content or "").strip()
        if not text:
            return None
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        try:
            data = json.loads(text)
            return data if isinstance(data, dict) else None
        except Exception:
            pass

        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                data = json.loads(text[start : end + 1])
                return data if isinstance(data, dict) else None
            except Exception:
                return None
        return None

    def _call_openrouter(self, api_key: str, model: str, prompt: str, image_data_url: str) -> dict | None:
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_data_url}},
                    ],
                }
            ],
            "response_format": {"type": "json_object"},
        }
        req = request.Request(
            "https://openrouter.ai/api/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://local.document-intelligence-refinery",
                "X-Title": "Document Intelligence Refinery",
            },
            method="POST",
        )
        with request.urlopen(req, timeout=self.settings.vlm_request_timeout_seconds) as resp:
            raw = resp.read().decode("utf-8", errors="ignore")
        data = json.loads(raw)
        choices = data.get("choices", [])
        if not choices:
            return None
        msg = choices[0].get("message", {})
        content = msg.get("content", "")
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    txt = item.get("text")
                    if txt:
                        parts.append(str(txt))
                elif isinstance(item, str):
                    parts.append(item)
            content_text = "\n".join(parts)
        else:
            content_text = str(content)
        return self._extract_json_payload(content_text)

    def _run_openrouter_vlm(self, pdf_path: Path, profile: DocumentProfile) -> tuple[ExtractedDocument, float, float, str] | None:
        if not (self.settings.use_openrouter_vlm and self.settings.openrouter_api_key):
            return None

        model = self._select_openrouter_model()
        pages: list[ExtractedPage] = []
        page_scores: list[float] = []
        cost_per_page = (
            self.settings.vlm_low_cost_page_cost_estimate
            if model == self.settings.vlm_model_low_cost
            else self.settings.vlm_high_quality_page_cost_estimate
        )

        try:
            with fitz.open(pdf_path) as doc:
                for pidx in range(1, doc.page_count + 1):
                    page = doc.load_page(pidx - 1)
                    pix = page.get_pixmap(dpi=200)
                    png_bytes = pix.tobytes("png")
                    data_url = "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")

                    parsed = self._call_openrouter(
                        api_key=self.settings.openrouter_api_key or "",
                        model=model,
                        prompt=self._build_vlm_prompt(profile, pidx),
                        image_data_url=data_url,
                    )

                    blocks: list[TextBlock] = []
                    tables: list[TableObject] = []
                    figures: list[FigureObject] = []

                    if isinstance(parsed, dict):
                        for i, b in enumerate(parsed.get("blocks", []) or []):
                            if not isinstance(b, dict):
                                continue
                            text = str(b.get("text", "")).strip()
                            if not text:
                                continue
                            bbox_raw = b.get("bbox", [0.0, 0.0, float(pix.width), float(pix.height)])
                            if not isinstance(bbox_raw, list) or len(bbox_raw) != 4:
                                bbox_raw = [0.0, 0.0, float(pix.width), float(pix.height)]
                            bbox = (
                                float(bbox_raw[0]),
                                float(bbox_raw[1]),
                                float(bbox_raw[2]),
                                float(bbox_raw[3]),
                            )
                            conf = float(b.get("confidence", 0.75))
                            blocks.append(
                                TextBlock(
                                    text=text,
                                    bbox=bbox,
                                    reading_order=int(b.get("reading_order", i)),
                                    confidence=max(0.05, min(1.0, conf)),
                                    confidence_signals=[
                                        _signal("vlm_element_confidence", conf, normalized=conf, weight=0.80),
                                        _signal("vlm_provider", 1.0, normalized=1.0, weight=0.20, passed=True),
                                    ],
                                    provenance=ProvenanceRef(
                                        doc_name=profile.doc_name,
                                        ref_type="pdf_bbox",
                                        page_number=pidx,
                                        bbox=bbox,
                                        content_hash="pending",
                                    ),
                                )
                            )

                        for i, t in enumerate(parsed.get("tables", []) or []):
                            if not isinstance(t, dict):
                                continue
                            bbox_raw = t.get("bbox", [0.0, 0.0, float(pix.width), float(pix.height)])
                            if not isinstance(bbox_raw, list) or len(bbox_raw) != 4:
                                bbox_raw = [0.0, 0.0, float(pix.width), float(pix.height)]
                            bbox = (
                                float(bbox_raw[0]),
                                float(bbox_raw[1]),
                                float(bbox_raw[2]),
                                float(bbox_raw[3]),
                            )
                            tables.append(
                                TableObject(
                                    bbox=bbox,
                                    headers=[str(h) for h in (t.get("headers", []) or [])],
                                    rows=[[str(c) for c in row] for row in (t.get("rows", []) or [])],
                                    reading_order=int(t.get("reading_order", 9000 + i)),
                                    confidence=max(0.05, min(1.0, float(t.get("confidence", 0.72)))),
                                    confidence_signals=[
                                        _signal(
                                            "vlm_element_confidence",
                                            float(t.get("confidence", 0.72)),
                                            normalized=float(t.get("confidence", 0.72)),
                                            weight=0.70,
                                        ),
                                        _signal(
                                            "header_presence",
                                            float(1.0 if (t.get("headers", []) or []) else 0.0),
                                            normalized=1.0 if (t.get("headers", []) or []) else 0.0,
                                            weight=0.30,
                                            threshold=1.0,
                                            passed=bool((t.get("headers", []) or [])),
                                        ),
                                    ],
                                    provenance=ProvenanceRef(
                                        doc_name=profile.doc_name,
                                        ref_type="pdf_bbox",
                                        page_number=pidx,
                                        bbox=bbox,
                                        content_hash="pending",
                                    ),
                                )
                            )

                        for i, f in enumerate(parsed.get("figures", []) or []):
                            if not isinstance(f, dict):
                                continue
                            bbox_raw = f.get("bbox", [0.0, 0.0, float(pix.width), float(pix.height)])
                            if not isinstance(bbox_raw, list) or len(bbox_raw) != 4:
                                bbox_raw = [0.0, 0.0, float(pix.width), float(pix.height)]
                            bbox = (
                                float(bbox_raw[0]),
                                float(bbox_raw[1]),
                                float(bbox_raw[2]),
                                float(bbox_raw[3]),
                            )
                            cap = f.get("caption", None)
                            figures.append(
                                FigureObject(
                                    bbox=bbox,
                                    caption=str(cap) if cap is not None else None,
                                    reading_order=int(f.get("reading_order", 9500 + i)),
                                    confidence=max(0.05, min(1.0, float(f.get("confidence", 0.72)))),
                                    confidence_signals=[
                                        _signal(
                                            "vlm_element_confidence",
                                            float(f.get("confidence", 0.72)),
                                            normalized=float(f.get("confidence", 0.72)),
                                            weight=0.70,
                                        ),
                                        _signal(
                                            "caption_presence",
                                            float(1.0 if cap else 0.0),
                                            normalized=1.0 if cap else 0.0,
                                            weight=0.30,
                                            threshold=1.0,
                                            passed=bool(cap),
                                        ),
                                    ],
                                    provenance=ProvenanceRef(
                                        doc_name=profile.doc_name,
                                        ref_type="pdf_bbox",
                                        page_number=pidx,
                                        bbox=bbox,
                                        content_hash="pending",
                                    ),
                                )
                            )

                    unresolved = not (blocks or tables or figures)
                    page_conf = float(parsed.get("page_confidence", 0.75)) if isinstance(parsed, dict) else 0.35
                    page_scores.append(max(0.05, min(1.0, page_conf)))
                    pages.append(
                        ExtractedPage(
                            page_number=pidx,
                            width=float(pix.width),
                            height=float(pix.height),
                            blocks=blocks,
                            tables=tables,
                            figures=figures,
                            unresolved_needs_vision=unresolved,
                        )
                    )

            confidence = sum(page_scores) / max(len(page_scores), 1)
            est_cost = cost_per_page * max(len(pages), 1)
            notes = f"openrouter_vlm model={model} pages={len(pages)}"
            return ExtractedDocument(doc_id=profile.doc_id, doc_name=profile.doc_name, pages=pages), confidence, est_cost, notes
        except Exception:
            return None

    def _run_docling_full_page_ocr(self, pdf_path: Path, profile: DocumentProfile) -> tuple[ExtractedDocument, float, float, str] | None:
        if not self.docling_adapter.available:
            return None
        try:
            extracted = self.docling_adapter.extract(pdf_path, profile)
            total_blocks = sum(len(p.blocks) for p in extracted.pages)
            if total_blocks == 0:
                return None
            confidence = 0.84 if profile.origin_type == "scanned_image" else 0.78
            return extracted, confidence, 0.12, "docling_full_page_ocr local"
        except Exception:
            return None

    def pixel_bbox_to_pdf_bbox(
        self,
        bbox_px: tuple[float, float, float, float],
        img_w_px: float,
        img_h_px: float,
        page_w_pt: float,
        page_h_pt: float,
    ) -> tuple[float, float, float, float]:
        x0_px, y0_px, x1_px, y1_px = bbox_px
        sx = page_w_pt / max(img_w_px, 1.0)
        sy = page_h_pt / max(img_h_px, 1.0)
        x0 = max(0.0, min(page_w_pt, x0_px * sx))
        y0 = max(0.0, min(page_h_pt, y0_px * sy))
        x1 = max(0.0, min(page_w_pt, x1_px * sx))
        y1 = max(0.0, min(page_h_pt, y1_px * sy))
        if x1 < x0:
            x0, x1 = x1, x0
        if y1 < y0:
            y0, y1 = y1, y0
        return (x0, y0, x1, y1)

    def _ocr_words_from_data(self, data: dict, *, img_w_px: float, img_h_px: float, page_w_pt: float, page_h_pt: float) -> list[dict]:
        words: list[dict] = []
        texts = data.get("text", []) or []
        n = len(texts)
        for i in range(n):
            txt = str(texts[i] or "").strip()
            if not txt:
                continue
            conf_raw = data.get("conf", [0] * n)[i] if i < len(data.get("conf", [])) else 0
            conf_txt = str(conf_raw).strip()
            conf = 0.0 if conf_txt in {"", "-1"} else float(conf_raw)
            if conf < 20.0:
                continue
            left = float(data.get("left", [0] * n)[i] if i < len(data.get("left", [])) else 0)
            top = float(data.get("top", [0] * n)[i] if i < len(data.get("top", [])) else 0)
            width = float(data.get("width", [0] * n)[i] if i < len(data.get("width", [])) else 0)
            height = float(data.get("height", [0] * n)[i] if i < len(data.get("height", [])) else 0)
            if width <= 0 or height <= 0:
                continue
            x0_pt, y0_pt, x1_pt, y1_pt = self.pixel_bbox_to_pdf_bbox(
                (left, top, left + width, top + height),
                img_w_px,
                img_h_px,
                page_w_pt,
                page_h_pt,
            )
            words.append(
                {
                    "index": i,
                    "text": txt,
                    "x": x0_pt,
                    "y": y0_pt,
                    "w": max(x1_pt - x0_pt, 0.1),
                    "h": max(y1_pt - y0_pt, 0.1),
                    "conf": conf,
                    "x_center": (x0_pt + x1_pt) / 2.0,
                    "y_center": (y0_pt + y1_pt) / 2.0,
                }
            )
        return words

    def _percentile(self, values: list[float], p: float) -> float:
        if not values:
            return 0.0
        v = sorted(values)
        pos = int(max(0, min(len(v) - 1, round((p / 100.0) * (len(v) - 1)))))
        return float(v[pos])

    def _compute_ocr_quality_metrics(self, words: list[dict], page_text: str) -> dict[str, float]:
        text = page_text or ""
        total_chars = max(len(text), 1)
        letters = sum(1 for ch in text if ch.isalpha())
        digits = sum(1 for ch in text if ch.isdigit())
        ethiopic_chars = len(re.findall(r"[\u1200-\u137F]", text))

        tokens = [t.lower() for t in re.findall(r"\w+", text, flags=re.UNICODE)]
        token_count = len(tokens)
        unique_count = len(set(tokens))
        avg_word_len = sum(len(t) for t in tokens) / max(token_count, 1)
        token_diversity = unique_count / max(token_count, 1)

        short_token_counts: dict[str, int] = {}
        for token in tokens:
            if len(token) <= 2:
                short_token_counts[token] = short_token_counts.get(token, 0) + 1
        repeated_short_chars = sum(len(tok) * (cnt - 1) for tok, cnt in short_token_counts.items() if cnt > 1)

        symbols = sum(1 for ch in text if not ch.isalnum() and not ch.isspace())
        garbage_ratio = (symbols + repeated_short_chars) / max(total_chars, 1)

        confs = [float(w.get("conf", 0.0)) for w in words if float(w.get("conf", 0.0)) >= 0.0]
        ocr_word_conf_mean = (sum(confs) / max(len(confs), 1)) if confs else 0.0
        ocr_word_conf_p10 = self._percentile(confs, 10.0) if confs else 0.0

        metrics = {
            "ocr_word_conf_mean": float(ocr_word_conf_mean),
            "ocr_word_conf_p10": float(ocr_word_conf_p10),
            "ethiopic_ratio": float(ethiopic_chars / max(letters, 1)),
            "alpha_ratio": float(letters / max(total_chars, 1)),
            "digit_ratio": float(digits / max(total_chars, 1)),
            "garbage_ratio": float(garbage_ratio),
            "avg_word_len": float(avg_word_len),
            "token_diversity": float(token_diversity),
        }
        return metrics

    def _quality_score_from_metrics(self, metrics: dict[str, float], origin_type: str) -> float:
        score = 1.0
        if metrics.get("ocr_word_conf_mean", 0.0) < 60.0:
            score *= 0.7
        if metrics.get("ocr_word_conf_p10", 0.0) < 30.0:
            score *= 0.8
        if metrics.get("garbage_ratio", 0.0) > 0.25:
            score *= 0.7
        if metrics.get("avg_word_len", 0.0) < 3.0:
            score *= 0.85
        if metrics.get("token_diversity", 0.0) < 0.25:
            score *= 0.85
        if origin_type == "scanned_image":
            score *= 0.9
        return max(0.05, min(0.95, float(score)))

    def _cluster_rows(self, words: list[dict]) -> list[list[dict]]:
        if not words:
            return []
        heights = [max(float(w["h"]), 1.0) for w in words]
        row_tol = 0.5 * median(heights)
        rows: list[dict] = []
        for word in sorted(words, key=lambda w: w["y_center"]):
            assigned = False
            for row in rows:
                if abs(float(word["y_center"]) - float(row["center"])) < row_tol:
                    row_words = row["words"]
                    row_words.append(word)
                    row["center"] = sum(float(w["y_center"]) for w in row_words) / len(row_words)
                    assigned = True
                    break
            if not assigned:
                rows.append({"center": float(word["y_center"]), "words": [word]})

        out: list[list[dict]] = []
        for row in sorted(rows, key=lambda r: float(r["center"])):
            out.append(sorted(row["words"], key=lambda w: float(w["x"])))
        return out

    def _infer_columns(self, rows: list[list[dict]]) -> list[float]:
        xs: list[float] = [float(w["x"]) for row in rows for w in row]
        widths: list[float] = [max(float(w["w"]), 1.0) for row in rows for w in row]
        if not xs or not widths:
            return []
        col_tol = 0.8 * median(widths)
        clusters: list[dict] = []
        for x in sorted(xs):
            assigned = False
            for cluster in clusters:
                if abs(x - float(cluster["center"])) < col_tol:
                    vals = cluster["vals"]
                    vals.append(x)
                    cluster["center"] = sum(vals) / len(vals)
                    assigned = True
                    break
            if not assigned:
                clusters.append({"center": x, "vals": [x]})
        return sorted(float(c["center"]) for c in clusters)

    def _looks_like_header_row(self, row_cells: list[str]) -> bool:
        non_empty = [c for c in row_cells if c.strip()]
        if not non_empty:
            return False
        numeric_like = 0
        for cell in non_empty:
            compact = re.sub(r"[\s,.-]", "", cell)
            if compact.isdigit():
                numeric_like += 1
        return (numeric_like / max(len(non_empty), 1)) < 0.5

    def _reconstruct_table_from_words(
        self,
        words: list[dict],
        *,
        page_number: int,
        doc_name: str,
        page_width: float,
        page_height: float,
    ) -> tuple[TableObject | None, set[int]]:
        if len(words) < 12:
            return None, set()

        rows_all = self._cluster_rows(words)
        rows = [row for row in rows_all if len(row) >= 3]
        if len(rows) < 4:
            return None, set()

        col_centers = self._infer_columns(rows)
        if len(col_centers) < 3:
            return None, set()

        grid: list[list[str]] = []
        used_indices: set[int] = set()
        occupied_counts: list[int] = []

        for row in rows:
            cells: list[list[str]] = [[] for _ in col_centers]
            occupied: set[int] = set()
            for word in row:
                nearest_col = min(range(len(col_centers)), key=lambda i: abs(float(word["x"]) - col_centers[i]))
                cells[nearest_col].append(str(word["text"]))
                occupied.add(nearest_col)
                used_indices.add(int(word["index"]))
            row_cells = [" ".join(parts).strip() for parts in cells]
            grid.append(row_cells)
            occupied_counts.append(sum(1 for c in row_cells if c))

        populated_rows = sum(1 for c in occupied_counts if c >= 3)
        if populated_rows / max(len(grid), 1) < 0.35:
            return None, set()

        col_count_per_row = [c for c in occupied_counts if c > 0]
        if not col_count_per_row:
            return None, set()
        stable_target = median(col_count_per_row)
        stable_rows = sum(1 for c in col_count_per_row if abs(c - stable_target) <= 1)
        if stable_rows / max(len(col_count_per_row), 1) < 0.35:
            return None, set()

        used_words = [w for w in words if int(w["index"]) in used_indices]
        if not used_words:
            return None, set()

        headers: list[str] = []
        data_rows = grid
        if self._looks_like_header_row(grid[0]):
            headers = grid[0]
            data_rows = grid[1:] if len(grid) > 1 else []

        x0 = min(float(w["x"]) for w in used_words)
        y0 = min(float(w["y"]) for w in used_words)
        x1 = max(float(w["x"]) + float(w["w"]) for w in used_words)
        y1 = max(float(w["y"]) + float(w["h"]) for w in used_words)
        mean_conf = sum(float(w["conf"]) for w in used_words) / max(len(used_words), 1)

        table = TableObject(
            bbox=(x0, y0, x1, y1),
            headers=headers,
            rows=data_rows,
            reading_order=9000,
            confidence=max(0.05, min(1.0, mean_conf / 100.0)),
            confidence_signals=[
                _signal("ocr_mean_word_conf", mean_conf, normalized=mean_conf / 100.0, weight=0.70),
                _signal(
                    "table_grid_stability",
                    float(stable_rows / max(len(col_count_per_row), 1)),
                    normalized=float(stable_rows / max(len(col_count_per_row), 1)),
                    weight=0.30,
                    threshold=0.35,
                    passed=(stable_rows / max(len(col_count_per_row), 1)) >= 0.35,
                ),
            ],
            provenance=ProvenanceRef(
                doc_name=doc_name,
                ref_type="pdf_bbox",
                page_number=page_number,
                bbox=(x0, y0, x1, y1),
                content_hash="pending",
            ),
        )
        return table, used_indices

    def _word_in_table_bbox(self, word_bbox: tuple[float, float, float, float], table_bbox: tuple[float, float, float, float]) -> bool:
        wx0, wy0, wx1, wy1 = word_bbox
        tx0, ty0, tx1, ty1 = table_bbox
        inter_w = max(0.0, min(wx1, tx1) - max(wx0, tx0))
        inter_h = max(0.0, min(wy1, ty1) - max(wy0, ty0))
        inter_area = inter_w * inter_h
        word_area = max((wx1 - wx0) * (wy1 - wy0), 1.0)
        return (inter_area / word_area) >= 0.6

    def _run_tesseract_ocr(self, pdf_path: Path, profile: DocumentProfile) -> tuple[ExtractedDocument, float, float, str] | None:
        if not (self.settings.ocr_enabled and self.settings.ocr_engine == "tesseract"):
            return None

        ocr_lang_used = select_ocr_lang(
            profile.language_hint.language,
            profile.language_hint.confidence,
            ocr_amharic_enabled=self.settings.ocr_amharic_enabled,
            ocr_lang_default=self.settings.ocr_lang_default,
            ocr_lang_fallback=self.settings.ocr_lang_fallback,
        )
        try:
            import pytesseract

            pages: list[ExtractedPage] = []
            with fitz.open(pdf_path) as doc:
                all_scores: list[float] = []
                page_metrics_accum: list[dict[str, float]] = []
                for pidx in range(1, doc.page_count + 1):
                    page = doc.load_page(pidx - 1)
                    pix = page.get_pixmap(dpi=200)
                    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                    try:
                        data = pytesseract.image_to_data(img, lang=ocr_lang_used, output_type=pytesseract.Output.DICT)
                    except Exception:
                        data = pytesseract.image_to_data(img, lang=self.settings.ocr_lang_default, output_type=pytesseract.Output.DICT)
                        ocr_lang_used = f"{self.settings.ocr_lang_default}(fallback)"
                    page_w_pt = float(page.rect.width)
                    page_h_pt = float(page.rect.height)
                    ocr_words = self._ocr_words_from_data(
                        data,
                        img_w_px=float(pix.width),
                        img_h_px=float(pix.height),
                        page_w_pt=page_w_pt,
                        page_h_pt=page_h_pt,
                    )
                    table_obj, _table_word_indices = self._reconstruct_table_from_words(
                        ocr_words,
                        page_number=pidx,
                        doc_name=profile.doc_name,
                        page_width=page_w_pt,
                        page_height=page_h_pt,
                    )

                    blocks: list[TextBlock] = []
                    confidences: list[float] = []
                    non_empty = 0
                    for i in range(len(data.get("text", []))):
                        txt = (data["text"][i] or "").strip()
                        if not txt:
                            continue
                        x = float(data["left"][i])
                        y = float(data["top"][i])
                        w = float(data["width"][i])
                        h = float(data["height"][i])
                        bx0, by0, bx1, by1 = self.pixel_bbox_to_pdf_bbox(
                            (x, y, x + w, y + h),
                            float(pix.width),
                            float(pix.height),
                            page_w_pt,
                            page_h_pt,
                        )
                        c = float(data.get("conf", [0])[i]) if str(data.get("conf", [0])[i]).strip() not in {"-1", ""} else 0.0
                        confidences.append(max(0.0, min(100.0, c)))
                        non_empty += 1
                        blocks.append(
                            TextBlock(
                                text=txt,
                                bbox=(bx0, by0, bx1, by1),
                                reading_order=i,
                                confidence=max(0.05, min(1.0, c / 100.0)),
                                confidence_signals=[
                                    _signal("ocr_word_conf", c, normalized=c / 100.0, weight=0.80),
                                    _signal("ocr_word_present", 1.0, normalized=1.0, weight=0.20, passed=True),
                                ],
                                provenance=ProvenanceRef(
                                    doc_name=profile.doc_name,
                                    ref_type="pdf_bbox",
                                    page_number=pidx,
                                    bbox=(bx0, by0, bx1, by1),
                                    content_hash="pending",
                                ),
                            )
                        )
                    page_text = " ".join((b.text or "").strip() for b in blocks if (b.text or "").strip())
                    page_metrics = self._compute_ocr_quality_metrics(ocr_words, page_text)
                    page_score = self._quality_score_from_metrics(page_metrics, profile.origin_type)
                    all_scores.append(page_score)
                    page_metrics_accum.append(page_metrics)
                    pages.append(
                        ExtractedPage(
                            page_number=pidx,
                            width=page_w_pt,
                            height=page_h_pt,
                            blocks=blocks,
                            tables=[table_obj] if table_obj is not None else [],
                            figures=[],
                            quality=page_metrics,
                            unresolved_needs_vision=False,
                        )
                    )
            doc_score = sum(all_scores) / max(len(all_scores), 1)
            table_pages = sum(1 for p in pages if p.tables)
            if page_metrics_accum:
                mean_conf_doc = sum(m["ocr_word_conf_mean"] for m in page_metrics_accum) / len(page_metrics_accum)
                garbage_doc = sum(m["garbage_ratio"] for m in page_metrics_accum) / len(page_metrics_accum)
                token_div_doc = sum(m["token_diversity"] for m in page_metrics_accum) / len(page_metrics_accum)
                notes = (
                    f"ocr_engine=tesseract ocr_lang_used={ocr_lang_used}; table_pages={table_pages}; "
                    f"ocr_word_conf_mean={mean_conf_doc:.2f}; garbage_ratio={garbage_doc:.3f}; token_diversity={token_div_doc:.3f}"
                )
            else:
                notes = f"ocr_engine=tesseract ocr_lang_used={ocr_lang_used}; table_pages={table_pages}"
            return ExtractedDocument(doc_id=profile.doc_id, doc_name=profile.doc_name, pages=pages), float(doc_score), 0.0, notes
        except Exception:
            return None

    def _run_unresolved_local_fallback(self, pdf_path: Path, profile: DocumentProfile) -> tuple[ExtractedDocument, float, float, str]:
        pages: list[ExtractedPage] = []
        with pdfplumber.open(pdf_path) as pdf:
            for pidx, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                blocks = [
                    TextBlock(
                        text=text.strip()[:500],
                        bbox=(0.0, 0.0, float(page.width), float(page.height)),
                        reading_order=0,
                        confidence=0.35,
                        confidence_signals=[
                            _signal("fallback_unresolved", 1.0, normalized=1.0, weight=1.0, passed=False),
                        ],
                        provenance=ProvenanceRef(
                            doc_name=profile.doc_name,
                            ref_type="pdf_bbox",
                            page_number=pidx,
                            bbox=(0.0, 0.0, float(page.width), float(page.height)),
                            content_hash="pending",
                        ),
                    )
                ] if text.strip() else []
                pages.append(
                    ExtractedPage(
                        page_number=pidx,
                        width=float(page.width),
                        height=float(page.height),
                        blocks=blocks,
                        tables=[],
                        figures=[],
                        unresolved_needs_vision=True,
                    )
                )
        extracted = ExtractedDocument(doc_id=profile.doc_id, doc_name=profile.doc_name, pages=pages)
        return extracted, 0.30, 0.0, "local_ocr_unavailable; unresolved pages preserved with provenance"

    def _run_local_strategy_c(self, pdf_path: Path, profile: DocumentProfile) -> tuple[ExtractedDocument, float, float, str]:
        if profile.origin_type == "scanned_image":
            tesseract = self._run_tesseract_ocr(pdf_path, profile)
            if tesseract is not None:
                return tesseract
            docling = self._run_docling_full_page_ocr(pdf_path, profile)
            if docling is not None:
                return docling
            return self._run_unresolved_local_fallback(pdf_path, profile)

        docling = self._run_docling_full_page_ocr(pdf_path, profile)
        if docling is not None:
            return docling
        tesseract = self._run_tesseract_ocr(pdf_path, profile)
        if tesseract is not None:
            return tesseract
        return self._run_unresolved_local_fallback(pdf_path, profile)

    def extract(self, pdf_path: Path, profile: DocumentProfile) -> tuple[ExtractedDocument, float, float, str]:
        local_result = self._run_local_strategy_c(pdf_path, profile)

        openrouter_allowed = self.settings.use_openrouter_vlm and bool(self.settings.openrouter_api_key)
        has_unresolved = any(p.unresolved_needs_vision for p in local_result[0].pages)
        if openrouter_allowed and (has_unresolved or local_result[1] < 0.55):
            provider_name = (self.settings.vlm_provider or "openrouter").strip().lower()
            vlm_result = self._run_vlm_provider(provider_name, pdf_path, profile)
            if vlm_result is not None:
                return vlm_result

        return local_result


class ExtractionRouter:
    def __init__(self, settings: Settings, store: ArtifactStore):
        self.settings = settings
        self.store = store
        self.rules = load_runtime_rules(settings)
        self.fast_confidence_floor = self._rule("fast_confidence_floor", 0.60)
        self.layout_confidence_floor = self._rule("layout_confidence_floor", 0.72)
        self.escalate_to_vision_floor = self._rule("escalate_to_vision_floor", 0.50)
        self.handwriting_whitespace_threshold = self._rule("handwriting_whitespace_threshold", 0.92)
        self.handwriting_char_density_threshold = self._rule("handwriting_char_density_threshold", 0.00005)
        self.handwriting_image_ratio_threshold = self._rule("handwriting_image_ratio_threshold", 0.55)
        self.docling_adapter = DoclingAdapter()
        self.fast = FastTextExtractor(
            min_chars_per_page=self._rule("fast_min_chars_per_page", 100.0),
            max_image_area_ratio=self._rule("fast_max_image_area_ratio", 0.50),
        )
        self.layout = LayoutLiteExtractor()
        self.vision = VisionExtractor(settings)
        self.layout_engines: dict[str, LayoutEngineFn] = {}
        self.register_layout_engine("default", self._run_layout_strategy_default)
        self.register_layout_engine("docling", self._run_layout_strategy_default)
        self.register_layout_engine("layout_lite", self._run_layout_lite_only)

    def _rule(self, key: str, default: float) -> float:
        ext = self.rules.get("extraction", {})
        value = ext.get(key, default)
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def register_layout_engine(self, name: str, engine: LayoutEngineFn) -> None:
        key = str(name or "").strip().lower()
        if not key:
            return
        self.layout_engines[key] = engine

    def _run_layout_strategy(self, pdf_path: Path, profile: DocumentProfile) -> tuple[ExtractedDocument, float, float, str, str]:
        engine_key = (self.settings.layout_engine or "default").strip().lower()
        engine = self.layout_engines.get(engine_key) or self.layout_engines.get("default")
        if engine is None:
            return self._run_layout_strategy_default(pdf_path, profile)
        try:
            return engine(pdf_path, profile)
        except Exception:
            return self._run_layout_strategy_default(pdf_path, profile)

    def _run_layout_strategy_default(self, pdf_path: Path, profile: DocumentProfile) -> tuple[ExtractedDocument, float, float, str, str]:
        if self.docling_adapter.available:
            try:
                extracted = self.docling_adapter.extract(pdf_path, profile)
                layout_doc, _, _, _ = self.layout.extract(pdf_path, profile)
                extracted = self._merge_layout_structures(extracted, layout_doc)
                return extracted, 0.90, 0.20, "docling adapter used + layout structure enrichment", "docling"
            except Exception:
                pass
        extracted, conf, cost, notes = self.layout.extract(pdf_path, profile)
        return extracted, conf, cost, notes, self.layout.name

    def _run_layout_lite_only(self, pdf_path: Path, profile: DocumentProfile) -> tuple[ExtractedDocument, float, float, str, str]:
        extracted, conf, cost, notes = self.layout.extract(pdf_path, profile)
        return extracted, conf, cost, notes, self.layout.name

    def _merge_layout_structures(self, primary: ExtractedDocument, structure: ExtractedDocument) -> ExtractedDocument:
        merged_pages: list[ExtractedPage] = []
        page_count = max(len(primary.pages), len(structure.pages))

        for i in range(page_count):
            p_page = primary.pages[i] if i < len(primary.pages) else None
            s_page = structure.pages[i] if i < len(structure.pages) else None

            if p_page is None and s_page is not None:
                merged_pages.append(s_page)
                continue
            if p_page is not None and s_page is None:
                merged_pages.append(p_page)
                continue
            if p_page is None or s_page is None:
                continue

            merged_pages.append(
                ExtractedPage(
                    page_number=p_page.page_number,
                    width=p_page.width,
                    height=p_page.height,
                    blocks=p_page.blocks if p_page.blocks else s_page.blocks,
                    tables=p_page.tables if p_page.tables else s_page.tables,
                    figures=p_page.figures if p_page.figures else s_page.figures,
                    unresolved_needs_vision=p_page.unresolved_needs_vision or s_page.unresolved_needs_vision,
                )
            )

        return ExtractedDocument(doc_id=primary.doc_id, doc_name=primary.doc_name, pages=merged_pages)

    def _profile_prefers_fast(self, profile: DocumentProfile) -> bool:
        return profile.origin_type == "native_digital" and profile.layout_complexity == "single_column"

    def _resolve_detected_language(self, profile: DocumentProfile, extracted: ExtractedDocument, ocr_lang_used: str | None) -> str:
        sample_parts: list[str] = []
        for page in extracted.pages:
            for block in page.blocks:
                txt = (block.text or "").strip()
                if txt:
                    sample_parts.append(txt)
                if len(sample_parts) >= 120:
                    break
            if len(sample_parts) >= 120:
                break
        sample_text = "\n".join(sample_parts)[:12000]
        if sample_text.strip():
            detected = detect_language(sample_text, mode=self.settings.language_detection_mode)
            lang = str(detected.get("language", "unknown"))
            if lang in {"am", "en"}:
                return lang

        # Keep OCR language logged separately, but do not force document language from OCR config hints.
        profile_lang = (profile.language_hint.language or "unknown").lower()
        if profile_lang in {"am", "en"}:
            return profile.language_hint.language
        return profile.language_hint.language

    def _profile_prefers_layout(self, profile: DocumentProfile) -> bool:
        return (
            profile.layout_complexity in {"multi_column", "table_heavy", "mixed"}
            or profile.origin_type in {"mixed", "form_fillable"}
        )

    def _handwriting_detected(self, profile: DocumentProfile) -> bool:
        return (
            profile.whitespace_ratio >= self.handwriting_whitespace_threshold
            and profile.avg_char_density <= self.handwriting_char_density_threshold
            and profile.image_area_ratio >= self.handwriting_image_ratio_threshold
        )

    def _domain_risk_boost(self, profile: DocumentProfile) -> float:
        if profile.domain_confidence < self.settings.router_domain_risk_confidence_gate:
            return 0.0
        if profile.domain_hint in {"legal", "medical"}:
            return self.settings.router_domain_risk_boost_high
        if profile.domain_hint in {"financial", "technical"}:
            return self.settings.router_domain_risk_boost_medium
        return 0.0

    def _build_policy_context(self, profile: DocumentProfile, budget_spent: float) -> RoutingPolicyContext:
        budget_headroom = self.settings.max_cost_per_doc - budget_spent
        risk_boost = self._domain_risk_boost(profile)
        adjusted_layout_floor = max(0.0, min(1.0, self.layout_confidence_floor + risk_boost))
        adjusted_vision_floor = max(0.0, min(1.0, self.escalate_to_vision_floor + risk_boost))

        allowed = budget_headroom >= self.settings.router_budget_min_vision_headroom
        if not self.settings.enable_vision:
            reason = "vision_disabled"
        elif allowed:
            reason = "budget_headroom_ok"
        else:
            reason = "insufficient_budget_headroom"

        return RoutingPolicyContext(
            domain_hint=profile.domain_hint,
            domain_confidence=profile.domain_confidence,
            max_cost_per_doc=self.settings.max_cost_per_doc,
            budget_spent=budget_spent,
            budget_headroom=budget_headroom,
            min_vision_headroom=self.settings.router_budget_min_vision_headroom,
            layout_confidence_floor=adjusted_layout_floor,
            vision_confidence_floor=adjusted_vision_floor,
            domain_risk_boost=risk_boost,
            vision_allowed=bool(self.settings.enable_vision and allowed),
            vision_policy_reason=reason,
        )

    def _build_page_strategy_history(self, extracted: ExtractedDocument, attempts: list[RoutingAttempt]) -> list[PageStrategyHistory]:
        attempted = [a.strategy for a in attempts]
        final_strategy = attempted[-1] if attempted else "unknown"
        history: list[PageStrategyHistory] = []
        for page in extracted.pages:
            ocr_word_conf_mean = None
            if isinstance(page.quality, dict) and "ocr_word_conf_mean" in page.quality:
                ocr_word_conf_mean = float(page.quality.get("ocr_word_conf_mean", 0.0))
            history.append(
                PageStrategyHistory(
                    page_number=page.page_number,
                    attempted_strategies=attempted,
                    final_strategy=final_strategy,
                    unresolved_needs_vision=page.unresolved_needs_vision,
                    block_count=len(page.blocks),
                    table_count=len(page.tables),
                    figure_count=len(page.figures),
                    ocr_word_conf_mean=ocr_word_conf_mean,
                )
            )
        return history

    def _write_ledger(
        self,
        *,
        doc_id: str,
        origin_type: str | None,
        layout_complexity: str | None,
        strategy: str,
        confidence: float,
        cost_estimate: float,
        duration_ms: int,
        pages_processed: int | None,
        blocks_extracted: int | None,
        tables_extracted: int | None,
        table_pages_count: int | None,
        ocr_word_conf_mean: float | None,
        ethiopic_ratio: float | None,
        garbage_ratio: float | None,
        avg_word_len: float | None,
        alpha_ratio: float | None,
        token_diversity: float | None,
        escalations: list[str],
        notes: str,
        detected_language: str | None = None,
        ocr_lang_used: str | None = None,
        policy_context: RoutingPolicyContext | None = None,
        routing_attempts: list[RoutingAttempt] | None = None,
        page_strategy_history: list[PageStrategyHistory] | None = None,
    ) -> None:
        strategy_label = self._normalize_strategy_label(strategy, notes)
        notes_final = self._augment_ledger_notes(
            notes,
            strategy=strategy_label,
            detected_language=detected_language,
            pages_processed=pages_processed,
            blocks_extracted=blocks_extracted,
            tables_extracted=tables_extracted,
        )
        entry = LedgerEntry(
            doc_id=doc_id,
            origin_type=origin_type,
            layout_complexity=layout_complexity,
            strategy_used=strategy_label,
            confidence_score=confidence,
            cost_estimate=cost_estimate,
            processing_time_ms=duration_ms,
            pages_processed=pages_processed,
            blocks_extracted=blocks_extracted,
            tables_extracted=tables_extracted,
            table_pages_count=table_pages_count,
            ocr_word_conf_mean=ocr_word_conf_mean,
            ethiopic_ratio=ethiopic_ratio,
            garbage_ratio=garbage_ratio,
            avg_word_len=avg_word_len,
            alpha_ratio=alpha_ratio,
            token_diversity=token_diversity,
            escalations=escalations,
            notes=notes_final,
            detected_language=detected_language,
            ocr_lang_used=ocr_lang_used,
            policy_context=policy_context,
            routing_attempts=routing_attempts or [],
            page_strategy_history=page_strategy_history or [],
        )
        self.store.append_jsonl(self.store.ledger_file, entry.model_dump(mode="json"))

    def _aggregate_page_quality(self, extracted: ExtractedDocument) -> dict[str, float | None]:
        quality_pages = [p.quality for p in extracted.pages if isinstance(p.quality, dict)]
        if not quality_pages:
            return {
                "ocr_word_conf_mean": None,
                "ethiopic_ratio": None,
                "garbage_ratio": None,
                "avg_word_len": None,
                "alpha_ratio": None,
                "token_diversity": None,
            }

        def avg(key: str) -> float:
            vals = [float(p.get(key, 0.0)) for p in quality_pages]
            return sum(vals) / max(len(vals), 1)

        return {
            "ocr_word_conf_mean": avg("ocr_word_conf_mean"),
            "ethiopic_ratio": avg("ethiopic_ratio"),
            "garbage_ratio": avg("garbage_ratio"),
            "avg_word_len": avg("avg_word_len"),
            "alpha_ratio": avg("alpha_ratio"),
            "token_diversity": avg("token_diversity"),
        }

    def _normalize_strategy_label(self, strategy: str, notes: str) -> str:
        value = (strategy or "").strip().lower()
        if value.startswith("strategy_"):
            return strategy

        notes_lower = (notes or "").lower()
        local_ocr_signals = (
            "docling_full_page_ocr local",
            "local_ocr_unavailable",
            "ocr_engine=tesseract",
            "ocr_engine=docling",
        )

        if value in {"fast_text"}:
            return "strategy_a_fast_text"

        if (
            value in {"vision_disabled", "docling+vision_disabled"}
            or value.endswith("+vision_disabled")
            or any(signal in notes_lower for signal in local_ocr_signals)
        ):
            return "strategy_c_local_ocr"

        if value in {"layout_lite", "docling"}:
            return "strategy_b_layout_docling"

        return strategy

    def _extract_ocr_engine(self, notes: str) -> str | None:
        match = re.search(r"\\bocr_engine=([^;\\s]+)", notes)
        if match:
            return match.group(1).strip()
        lower = notes.lower()
        if "tesseract" in lower:
            return "tesseract"
        if "docling_full_page_ocr" in lower or "docling adapter used" in lower:
            return "docling"
        return None

    def _augment_ledger_notes(
        self,
        notes: str,
        *,
        strategy: str,
        detected_language: str | None,
        pages_processed: int | None,
        blocks_extracted: int | None,
        tables_extracted: int | None,
    ) -> str:
        base = (notes or "").strip()
        extra: list[str] = []
        docling_used = "docling" in strategy.lower() or "docling" in base.lower()
        if "docling_adapter_used=" not in base:
            extra.append(f"docling_adapter_used={'true' if docling_used else 'false'}")
        if "ocr_engine=" not in base:
            ocr_engine = self._extract_ocr_engine(base)
            if ocr_engine:
                extra.append(f"ocr_engine={ocr_engine}")
        if detected_language and "language_detected=" not in base:
            extra.append(f"language_detected={detected_language}")
        if blocks_extracted is not None and "blocks=" not in base:
            extra.append(f"blocks={blocks_extracted}")
        if tables_extracted is not None and "tables=" not in base:
            extra.append(f"tables={tables_extracted}")
        if pages_processed is not None and "pages=" not in base and "pages_processed=" not in base:
            extra.append(f"pages={pages_processed}")
        if not base:
            return "; ".join(extra)
        if not extra:
            return base
        return base + "; " + "; ".join(extra)

    def extract(self, pdf_path: Path, profile: DocumentProfile) -> ExtractedDocument:
        started = time.perf_counter()
        logger.info(
            "stage=extraction start doc=%s strategy_hint=%s domain=%s domain_conf=%.2f budget_max=%.3f",
            profile.doc_name,
            profile.estimated_extraction_cost,
            profile.domain_hint,
            profile.domain_confidence,
            self.settings.max_cost_per_doc,
        )

        escalations: list[str] = []
        attempts: list[RoutingAttempt] = []
        extracted: ExtractedDocument
        conf: float
        cost: float
        notes: str
        strategy: str

        def _record_attempt(stage: str, strategy_name: str, confidence: float, cost_delta: float, cumulative_cost: float, note: str) -> None:
            attempts.append(
                RoutingAttempt(
                    stage=stage,
                    strategy=self._normalize_strategy_label(strategy_name, note),
                    confidence=confidence,
                    cost_delta=max(0.0, float(cost_delta)),
                    cumulative_cost=max(0.0, float(cumulative_cost)),
                    notes=note,
                )
            )

        handwriting = self._handwriting_detected(profile)
        if profile.origin_type == "scanned_image" or handwriting:
            extracted, conf, cost, notes = self.vision.extract(pdf_path, profile)
            strategy = self.vision.name if self.settings.enable_vision else f"{self.vision.name}_disabled"
            if handwriting:
                notes = f"{notes}; handwriting_like=true"
            _record_attempt("initial", strategy, conf, cost, cost, notes)
        elif self._profile_prefers_fast(profile):
            extracted, conf, cost, notes = self.fast.extract(pdf_path, profile)
            strategy = self.fast.name
            _record_attempt("initial", strategy, conf, cost, cost, notes)
            if conf < self.fast_confidence_floor:
                escalations.append("fast_to_layout")
                extracted_l, conf_l, cost_l, notes_l, strategy_l = self._run_layout_strategy(pdf_path, profile)
                extracted, conf, strategy = extracted_l, conf_l, strategy_l
                cost += cost_l
                notes = f"{notes}; {notes_l}"
                _record_attempt("escalation_fast_to_layout", strategy, conf, cost_l, cost, notes_l)
        else:
            extracted, conf, cost, notes, strategy = self._run_layout_strategy(pdf_path, profile)
            _record_attempt("initial", strategy, conf, cost, cost, notes)

        policy = self._build_policy_context(profile, cost)

        if conf < policy.layout_confidence_floor and policy.vision_allowed:
            if strategy != self.vision.name and not strategy.startswith(f"{self.vision.name}_"):
                escalations.append("layout_to_vision")
                extracted_v, conf_v, cost_v, notes_v = self.vision.extract(pdf_path, profile)
                extracted = extracted_v
                conf = conf_v
                strategy = self.vision.name if self.settings.enable_vision else f"{strategy}+vision_disabled"
                cost += cost_v
                notes = f"{notes}; {notes_v}"
                _record_attempt("escalation_layout_to_vision", strategy, conf, cost_v, cost, notes_v)
                policy = self._build_policy_context(profile, cost)
        elif conf < policy.layout_confidence_floor and not policy.vision_allowed:
            escalations.append("layout_to_vision_blocked")

        if conf < policy.vision_confidence_floor and policy.vision_allowed:
            if strategy != self.vision.name and not strategy.startswith(f"{self.vision.name}_"):
                escalations.append("low_conf_to_vision")
                extracted_v, conf_v, cost_v, notes_v = self.vision.extract(pdf_path, profile)
                extracted = extracted_v
                conf = conf_v
                strategy = self.vision.name if self.settings.enable_vision else f"{strategy}+vision_disabled"
                cost += cost_v
                notes = f"{notes}; {notes_v}"
                _record_attempt("escalation_low_conf_to_vision", strategy, conf, cost_v, cost, notes_v)
                policy = self._build_policy_context(profile, cost)
        elif conf < policy.vision_confidence_floor and not policy.vision_allowed:
            escalations.append("low_conf_to_vision_blocked")

        elapsed = int((time.perf_counter() - started) * 1000)
        pages_processed = len(extracted.pages) if extracted is not None else None
        blocks_extracted = sum(len(p.blocks) for p in extracted.pages) if extracted is not None else None
        tables_extracted = sum(len(p.tables) for p in extracted.pages) if extracted is not None else None
        table_pages_count = sum(1 for p in extracted.pages if p.tables) if extracted is not None else None
        quality_agg = self._aggregate_page_quality(extracted) if extracted is not None else {}
        ocr_lang_used = None
        if "ocr_lang_used=" in notes:
            ocr_lang_used = notes.split("ocr_lang_used=", 1)[1].split(";", 1)[0].strip()
        detected_language = self._resolve_detected_language(profile, extracted, ocr_lang_used)
        page_history = self._build_page_strategy_history(extracted, attempts)
        self._write_ledger(
            doc_id=profile.doc_id,
            origin_type=profile.origin_type,
            layout_complexity=profile.layout_complexity,
            strategy=strategy,
            confidence=conf,
            cost_estimate=cost,
            duration_ms=elapsed,
            pages_processed=pages_processed,
            blocks_extracted=blocks_extracted,
            tables_extracted=tables_extracted,
            table_pages_count=table_pages_count,
            ocr_word_conf_mean=quality_agg.get("ocr_word_conf_mean"),
            ethiopic_ratio=quality_agg.get("ethiopic_ratio"),
            garbage_ratio=quality_agg.get("garbage_ratio"),
            avg_word_len=quality_agg.get("avg_word_len"),
            alpha_ratio=quality_agg.get("alpha_ratio"),
            token_diversity=quality_agg.get("token_diversity"),
            escalations=escalations,
            notes=notes,
            detected_language=detected_language,
            ocr_lang_used=ocr_lang_used,
            policy_context=policy,
            routing_attempts=attempts,
            page_strategy_history=page_history,
        )
        logger.info(
            "stage=extraction end doc=%s strategy=%s pages=%s confidence=%.3f duration_ms=%s budget_spent=%.3f budget_headroom=%.3f escalations=%s",
            profile.doc_name,
            strategy,
            len(extracted.pages),
            conf,
            elapsed,
            cost,
            policy.budget_headroom,
            "|".join(escalations) if escalations else "none",
        )
        return extracted
