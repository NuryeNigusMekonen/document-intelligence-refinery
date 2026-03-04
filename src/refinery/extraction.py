from __future__ import annotations

import base64
import json
import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from urllib import request

import fitz
import pdfplumber
import yaml
from PIL import Image

from .adapters import DoclingAdapter
from .config import Settings
from .lang_detect import select_ocr_lang
from .models import DocumentProfile, ExtractedDocument, ExtractedPage, FigureObject, LedgerEntry, ProvenanceRef, TableObject, TextBlock
from .storage import ArtifactStore

logger = logging.getLogger(__name__)


DEFAULT_EXTRACTION_RULES: dict[str, dict[str, float]] = {
    "extraction": {
        "fast_min_chars_per_page": 100.0,
        "fast_max_image_area_ratio": 0.50,
        "fast_confidence_floor": 0.60,
        "layout_confidence_floor": 0.72,
        "escalate_to_vision_floor": 0.50,
        "handwriting_whitespace_threshold": 0.92,
        "handwriting_char_density_threshold": 0.00005,
        "handwriting_image_ratio_threshold": 0.55,
    }
}


def _load_rules(settings: Settings) -> dict[str, dict[str, float]]:
    rules_path = settings.workspace_root / "extraction_rules.yaml"
    if not rules_path.exists():
        return DEFAULT_EXTRACTION_RULES
    try:
        loaded = yaml.safe_load(rules_path.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            return DEFAULT_EXTRACTION_RULES
        out: dict[str, dict[str, float]] = {"extraction": dict(DEFAULT_EXTRACTION_RULES["extraction"])}
        ext = loaded.get("extraction", {})
        if isinstance(ext, dict):
            for key, value in ext.items():
                try:
                    out["extraction"][str(key)] = float(value)
                except (TypeError, ValueError):
                    continue
        return out
    except Exception:
        return DEFAULT_EXTRACTION_RULES


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

    def _select_openrouter_model(self) -> str:
        if self.settings.max_cost_per_doc <= 1.0:
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
        with request.urlopen(req, timeout=60) as resp:
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
        if self.settings.vlm_provider.lower() != "openrouter":
            return None
        if not (self.settings.use_openrouter_vlm and self.settings.openrouter_api_key):
            return None

        model = self._select_openrouter_model()
        pages: list[ExtractedPage] = []
        page_scores: list[float] = []
        cost_per_page = 0.08 if model == "openai/gpt-4o-mini" else 0.12

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
            vlm_result = self._run_openrouter_vlm(pdf_path, profile)
            if vlm_result is not None:
                return vlm_result

        return local_result


class ExtractionRouter:
    def __init__(self, settings: Settings, store: ArtifactStore):
        self.settings = settings
        self.store = store
        self.rules = _load_rules(settings)
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

    def _rule(self, key: str, default: float) -> float:
        ext = self.rules.get("extraction", {})
        value = ext.get(key, default)
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _run_layout_strategy(self, pdf_path: Path, profile: DocumentProfile) -> tuple[ExtractedDocument, float, float, str, str]:
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
        extracted: ExtractedDocument
        conf: float
        cost: float
        notes: str
        strategy: str

        handwriting = self._handwriting_detected(profile)
        if profile.origin_type == "scanned_image" or handwriting:
            extracted, conf, cost, notes = self.vision.extract(pdf_path, profile)
            strategy = self.vision.name if self.settings.enable_vision else f"{self.vision.name}_disabled"
            if handwriting:
                notes = f"{notes}; handwriting_like=true"
        elif self._profile_prefers_fast(profile):
            extracted, conf, cost, notes = self.fast.extract(pdf_path, profile)
            strategy = self.fast.name
            if conf < self.fast_confidence_floor:
                escalations.append("fast_to_layout")
                extracted_l, conf_l, cost_l, notes_l, strategy_l = self._run_layout_strategy(pdf_path, profile)
                extracted, conf, strategy = extracted_l, conf_l, strategy_l
                cost += cost_l
                notes = f"{notes}; {notes_l}"
        else:
            extracted, conf, cost, notes, strategy = self._run_layout_strategy(pdf_path, profile)

        if conf < self.layout_confidence_floor and cost <= self.settings.max_cost_per_doc:
            if strategy != self.vision.name and not strategy.startswith(f"{self.vision.name}_"):
                escalations.append("layout_to_vision")
                extracted_v, conf_v, cost_v, notes_v = self.vision.extract(pdf_path, profile)
                extracted = extracted_v
                conf = conf_v
                strategy = self.vision.name if self.settings.enable_vision else f"{strategy}+vision_disabled"
                cost += cost_v
                notes = f"{notes}; {notes_v}"

        if conf < self.escalate_to_vision_floor and cost <= self.settings.max_cost_per_doc:
            if strategy != self.vision.name and not strategy.startswith(f"{self.vision.name}_"):
                escalations.append("low_conf_to_vision")
                extracted_v, conf_v, cost_v, notes_v = self.vision.extract(pdf_path, profile)
                extracted = extracted_v
                conf = conf_v
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
