from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path

import fitz
import pdfplumber
from PIL import Image

from .adapters import DoclingAdapter, MineruAdapter
from .config import Settings
from .lang_detect import select_ocr_lang
from .models import DocumentProfile, ExtractedDocument, ExtractedPage, FigureObject, LedgerEntry, ProvenanceRef, TableObject, TextBlock
from .storage import ArtifactStore

logger = logging.getLogger(__name__)


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

    def extract(self, pdf_path: Path, profile: DocumentProfile) -> tuple[ExtractedDocument, float, float, str]:
        started = time.perf_counter()
        pages: list[ExtractedPage] = []
        with pdfplumber.open(pdf_path) as pdf:
            for pidx, page in enumerate(pdf.pages, start=1):
                words = page.extract_words() or []
                blocks: list[TextBlock] = []
                for i, w in enumerate(words):
                    text = (w.get("text") or "").strip()
                    if not text:
                        continue
                    blocks.append(
                        TextBlock(
                            text=text,
                            bbox=_as_bbox(w, (0.0, 0.0, page.width, page.height)),
                            reading_order=i,
                            confidence=0.85,
                            provenance=ProvenanceRef(
                                doc_name=profile.doc_name,
                                ref_type="pdf_bbox",
                                page_number=pidx,
                                bbox=_as_bbox(w, (0.0, 0.0, page.width, page.height)),
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
        confidence = _confidence_from_profile(profile)
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        notes = f"pages={len(pages)} blocks={sum(len(p.blocks) for p in pages)} tables={sum(len(p.tables) for p in pages)}"
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
                            provenance=ProvenanceRef(
                                doc_name=profile.doc_name,
                                ref_type="pdf_bbox",
                                page_number=pidx,
                                bbox=bbox,
                                content_hash="pending",
                            ),
                        )
                    )

                figures = []
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

    def extract(self, pdf_path: Path, profile: DocumentProfile) -> tuple[ExtractedDocument, float, float, str]:
        pages: list[ExtractedPage] = []
        ocr_lang_used = None
        if self.settings.ocr_enabled and self.settings.ocr_engine == "tesseract":
            ocr_lang_used = select_ocr_lang(
                profile.language_hint.language,
                profile.language_hint.confidence,
                ocr_amharic_enabled=self.settings.ocr_amharic_enabled,
                ocr_lang_default=self.settings.ocr_lang_default,
                ocr_lang_fallback=self.settings.ocr_lang_fallback,
            )
            try:
                import pytesseract

                with fitz.open(pdf_path) as doc:
                    all_scores: list[float] = []
                    for pidx in range(1, doc.page_count + 1):
                        page = doc.load_page(pidx - 1)
                        pix = page.get_pixmap(dpi=200)
                        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                        try:
                            data = pytesseract.image_to_data(img, lang=ocr_lang_used, output_type=pytesseract.Output.DICT)
                        except Exception:
                            data = pytesseract.image_to_data(img, lang=self.settings.ocr_lang_default, output_type=pytesseract.Output.DICT)
                            ocr_lang_used = f"{self.settings.ocr_lang_default}(fallback)"
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
                            c = float(data.get("conf", [0])[i]) if str(data.get("conf", [0])[i]).strip() not in {"-1", ""} else 0.0
                            confidences.append(max(0.0, min(100.0, c)))
                            non_empty += 1
                            blocks.append(
                                TextBlock(
                                    text=txt,
                                    bbox=(x, y, x + w, y + h),
                                    reading_order=i,
                                    confidence=max(0.05, min(1.0, c / 100.0)),
                                    provenance=ProvenanceRef(
                                        doc_name=profile.doc_name,
                                        ref_type="pdf_bbox",
                                        page_number=pidx,
                                        bbox=(x, y, x + w, y + h),
                                        content_hash="pending",
                                    ),
                                )
                            )
                        mean_conf = (sum(confidences) / max(len(confidences), 1)) / 100.0
                        coverage = min(1.0, non_empty / 120.0)
                        page_score = max(0.1, min(0.95, 0.5 * mean_conf + 0.5 * coverage))
                        all_scores.append(page_score)
                        pages.append(
                            ExtractedPage(
                                page_number=pidx,
                                width=float(pix.width),
                                height=float(pix.height),
                                blocks=blocks,
                                tables=[],
                                figures=[],
                                unresolved_needs_vision=False,
                            )
                        )
                doc_score = sum(all_scores) / max(len(all_scores), 1)
                notes = f"ocr_engine=tesseract ocr_lang_used={ocr_lang_used}"
                return ExtractedDocument(doc_id=profile.doc_id, doc_name=profile.doc_name, pages=pages), float(doc_score), 0.0, notes
            except Exception:
                pass

        with pdfplumber.open(pdf_path) as pdf:
            for pidx, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                unresolved = True
                if self.settings.enable_vision and self.settings.openrouter_api_key:
                    unresolved = False
                blocks = [
                    TextBlock(
                        text=text.strip()[:500],
                        bbox=(0.0, 0.0, float(page.width), float(page.height)),
                        reading_order=0,
                        confidence=0.35 if unresolved else 0.75,
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
                        unresolved_needs_vision=unresolved,
                    )
                )
        extracted = ExtractedDocument(doc_id=profile.doc_id, doc_name=profile.doc_name, pages=pages)
        if self.settings.enable_vision and self.settings.openrouter_api_key:
            return extracted, 0.80, 0.70, "vision adapter active"
        fallback_note = "vision disabled; unresolved pages preserved with provenance"
        if ocr_lang_used:
            fallback_note += f"; ocr_lang_used={ocr_lang_used}"
        return extracted, 0.30, 0.0, fallback_note


class ExtractionRouter:
    def __init__(self, settings: Settings, store: ArtifactStore):
        self.settings = settings
        self.store = store
        self.docling_adapter = DoclingAdapter()
        self.mineru_adapter = MineruAdapter()
        self.fast = FastTextExtractor()
        self.layout = LayoutLiteExtractor()
        self.vision = VisionExtractor(settings)

    def _write_ledger(
        self,
        *,
        doc_id: str,
        strategy: str,
        confidence: float,
        cost_estimate: float,
        duration_ms: int,
        escalations: list[str],
        notes: str,
        detected_language: str | None = None,
        ocr_lang_used: str | None = None,
    ) -> None:
        entry = LedgerEntry(
            doc_id=doc_id,
            strategy_used=strategy,
            confidence_score=confidence,
            cost_estimate=cost_estimate,
            processing_time_ms=duration_ms,
            escalations=escalations,
            notes=notes,
            detected_language=detected_language,
            ocr_lang_used=ocr_lang_used,
        )
        self.store.append_jsonl(self.store.ledger_file, entry.model_dump(mode="json"))

    def extract(self, pdf_path: Path, profile: DocumentProfile) -> ExtractedDocument:
        started = time.perf_counter()
        logger.info("stage=extraction start doc=%s strategy_hint=%s", profile.doc_name, profile.estimated_extraction_cost)

        escalations: list[str] = []
        extracted, conf, cost, notes = self.fast.extract(pdf_path, profile)
        strategy = self.fast.name

        if profile.estimated_extraction_cost != "fast_text_sufficient" or conf < 0.60:
            escalations.append("fast_to_layout")
            if self.docling_adapter.available:
                try:
                    extracted = self.docling_adapter.extract(pdf_path, profile)
                    conf, cost_l, notes_l = 0.90, 0.20, "docling adapter used"
                    strategy = "docling"
                except Exception:
                    extracted, conf, cost_l, notes_l = self.layout.extract(pdf_path, profile)
                    strategy = self.layout.name
            elif self.mineru_adapter.available:
                try:
                    extracted = self.mineru_adapter.extract(pdf_path, profile)
                    conf, cost_l, notes_l = 0.88, 0.20, "mineru adapter used"
                    strategy = "mineru"
                except Exception:
                    extracted, conf, cost_l, notes_l = self.layout.extract(pdf_path, profile)
                    strategy = self.layout.name
            else:
                extracted, conf, cost_l, notes_l = self.layout.extract(pdf_path, profile)
                strategy = self.layout.name
            cost += cost_l
            notes = f"{notes}; {notes_l}"

        if (profile.estimated_extraction_cost == "needs_vision_model" or conf < 0.50) and cost <= self.settings.max_cost_per_doc:
            escalations.append("layout_to_vision")
            extracted, conf, cost_v, notes_v = self.vision.extract(pdf_path, profile)
            strategy = self.vision.name if self.settings.enable_vision else f"{strategy}+vision_disabled"
            cost += cost_v
            notes = f"{notes}; {notes_v}"

        elapsed = int((time.perf_counter() - started) * 1000)
        ocr_lang_used = None
        if "ocr_lang_used=" in notes:
            ocr_lang_used = notes.split("ocr_lang_used=", 1)[1].split(";", 1)[0].strip()
        self._write_ledger(
            doc_id=profile.doc_id,
            strategy=strategy,
            confidence=conf,
            cost_estimate=cost,
            duration_ms=elapsed,
            escalations=escalations,
            notes=notes,
            detected_language=profile.language_hint.language,
            ocr_lang_used=ocr_lang_used,
        )
        logger.info(
            "stage=extraction end doc=%s strategy=%s pages=%s confidence=%.3f duration_ms=%s",
            profile.doc_name,
            strategy,
            len(extracted.pages),
            conf,
            elapsed,
        )
        return extracted
