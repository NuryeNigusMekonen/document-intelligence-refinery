from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import fitz
import pdfplumber

from .config import Settings
from .lang_detect import detect_language
from .models import DocumentProfile, LanguageHint
from .storage import ArtifactStore
from .utils import deterministic_id, sha256_file

logger = logging.getLogger(__name__)


@dataclass
class TriageMetrics:
    page_count: int
    avg_char_density: float
    image_area_ratio: float
    whitespace_ratio: float
    low_char_pages_fraction: float
    estimated_columns: int
    table_signal_ratio: float
    text_sample: str
    form_field_count: int = 0
    multi_column_fraction: float = 0.0


def classify_profile(doc_name: str, sha256: str, metrics: TriageMetrics, language_hint: LanguageHint | None = None) -> DocumentProfile:
    if metrics.form_field_count > 0:
        origin_type = "form_fillable"
    elif metrics.image_area_ratio > 0.75 and metrics.avg_char_density < 0.00008:
        origin_type = "scanned_image"
    elif metrics.image_area_ratio > 0.20 and metrics.avg_char_density < 0.0002:
        origin_type = "mixed"
    else:
        origin_type = "native_digital"

    table_heavy = metrics.table_signal_ratio > 0.25
    figure_heavy = metrics.image_area_ratio > 0.50
    multi_column = metrics.estimated_columns >= 2 or metrics.multi_column_fraction > 0.45

    signal_count = sum(1 for flag in (table_heavy, figure_heavy, multi_column) if flag)

    if signal_count >= 2:
        layout_complexity = "mixed"
    elif table_heavy:
        layout_complexity = "table_heavy"
    elif multi_column:
        layout_complexity = "multi_column"
    elif figure_heavy:
        layout_complexity = "figure_heavy"
    else:
        layout_complexity = "single_column"

    if metrics.avg_char_density < 0.0001 and metrics.image_area_ratio > 0.60:
        estimated_extraction_cost = "needs_vision_model"
    elif (
        layout_complexity in {"multi_column", "table_heavy", "figure_heavy", "mixed"}
        or metrics.low_char_pages_fraction > 0.3
        or metrics.form_field_count > 0
    ):
        estimated_extraction_cost = "needs_layout_model"
    else:
        estimated_extraction_cost = "fast_text_sufficient"

    lower = f"{doc_name} {metrics.text_sample[:5000]}".lower()
    if any(k in lower for k in ["balance", "income", "financial", "earnings", "assets", "revenue", "expense"]):
        domain_hint = "financial"
    elif any(k in lower for k in ["contract", "agreement", "terms", "legal", "whereas", "party", "liability"]):
        domain_hint = "legal"
    elif any(k in lower for k in ["manual", "spec", "api", "technical", "architecture", "endpoint", "algorithm"]):
        domain_hint = "technical"
    elif any(k in lower for k in ["clinical", "patient", "medical", "diagnosis", "treatment", "prescription"]):
        domain_hint = "medical"
    else:
        domain_hint = "general"

    language_hint = language_hint or LanguageHint(language="en", confidence=0.6)
    doc_id = deterministic_id("doc", {"name": doc_name, "sha256": sha256})
    return DocumentProfile(
        doc_id=doc_id,
        doc_name=doc_name,
        sha256=sha256,
        origin_type=origin_type,
        layout_complexity=layout_complexity,
        language_hint=language_hint,
        domain_hint=domain_hint,
        estimated_extraction_cost=estimated_extraction_cost,
        page_count=metrics.page_count,
        avg_char_density=metrics.avg_char_density,
        image_area_ratio=metrics.image_area_ratio,
        whitespace_ratio=metrics.whitespace_ratio,
    )


class TriageAgent:
    def __init__(self, store: ArtifactStore, settings: Settings | None = None):
        self.store = store
        self.settings = settings or Settings(workspace_root=store.settings.workspace_root)

    def _compute_metrics(self, pdf_path: Path) -> TriageMetrics:
        total_chars = 0
        total_page_area = 0.0
        total_image_area = 0.0
        low_char_pages = 0
        col_votes: list[int] = []
        table_hits = 0
        form_field_count = 0
        page_count = 0
        text_parts: list[str] = []

        with pdfplumber.open(pdf_path) as plumber_pdf, fitz.open(pdf_path) as mu_pdf:
            page_count = len(plumber_pdf.pages)
            for idx, page in enumerate(plumber_pdf.pages):
                width, height = page.width, page.height
                page_area = max(width * height, 1.0)
                total_page_area += page_area

                text = page.extract_text() or ""
                if idx < 5 and text.strip():
                    text_parts.append(text[:2000])
                chars = len(text)
                total_chars += chars
                if chars < 80:
                    low_char_pages += 1

                words = page.extract_words() or []
                if words:
                    x_positions = [w.get("x0", 0.0) for w in words]
                    median_x = sorted(x_positions)[len(x_positions) // 2]
                    left = sum(1 for x in x_positions if x < median_x - 20)
                    right = sum(1 for x in x_positions if x > median_x + 20)
                    col_votes.append(2 if left > 15 and right > 15 else 1)
                else:
                    col_votes.append(1)

                if page.find_tables():
                    table_hits += 1

                mu_page = mu_pdf[idx]
                widgets = mu_page.widgets()
                form_field_count += sum(1 for _ in widgets) if widgets is not None else 0
                for img in mu_page.get_images(full=True):
                    xref = img[0]
                    rects = mu_page.get_image_rects(xref)
                    total_image_area += sum(r.width * r.height for r in rects)

        avg_char_density = total_chars / max(total_page_area, 1.0)
        image_area_ratio = total_image_area / max(total_page_area, 1.0)
        whitespace_ratio = max(0.0, min(1.0, 1.0 - (avg_char_density / 0.0015)))
        multi_column_votes = sum(1 for c in col_votes if c == 2)
        estimated_columns = 2 if multi_column_votes > (len(col_votes) / 2) else 1
        table_signal_ratio = table_hits / max(page_count, 1)
        multi_column_fraction = multi_column_votes / max(page_count, 1)

        return TriageMetrics(
            page_count=page_count,
            avg_char_density=avg_char_density,
            image_area_ratio=image_area_ratio,
            whitespace_ratio=whitespace_ratio,
            low_char_pages_fraction=low_char_pages / max(page_count, 1),
            estimated_columns=estimated_columns,
            table_signal_ratio=table_signal_ratio,
            text_sample="\n".join(text_parts)[:8000],
            form_field_count=form_field_count,
            multi_column_fraction=multi_column_fraction,
        )

    def run(self, pdf_path: Path) -> DocumentProfile:
        started = time.perf_counter()
        logger.info("stage=triage start doc=%s", pdf_path.name)
        sha = sha256_file(pdf_path)
        metrics = self._compute_metrics(pdf_path)
        lang = detect_language(metrics.text_sample, mode=self.settings.language_detection_mode)
        language_hint = LanguageHint(language=lang["language"], confidence=float(lang["confidence"]))
        profile = classify_profile(pdf_path.name, sha, metrics, language_hint=language_hint)
        out = self.store.profiles_dir / f"{profile.doc_id}.json"
        self.store.save_json(out, profile.model_dump(mode="json"))
        elapsed = int((time.perf_counter() - started) * 1000)
        logger.info("stage=triage end doc=%s pages=%s duration_ms=%s", pdf_path.name, metrics.page_count, elapsed)
        return profile
