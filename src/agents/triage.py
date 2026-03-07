from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

import fitz
import pdfplumber

from models import DocumentProfile, LanguageHint
from refinery.config import Settings
from refinery.lang_detect import detect_language
from refinery.runtime_rules import load_runtime_rules
from refinery.storage import ArtifactStore
from refinery.utils import deterministic_id, sha256_file

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
    mixed_mode_pages_fraction: float = 0.0
    zero_text_document: bool = False


@dataclass
class DomainClassification:
    domain_hint: Literal["financial", "legal", "technical", "medical", "general"]
    confidence: float


class DomainClassifier(Protocol):
    def classify(self, doc_name: str, text_sample: str) -> DomainClassification:
        ...


class KeywordDomainClassifier:
    def __init__(self, domain_keywords: dict[str, list[str]] | None = None):
        self.domain_keywords = domain_keywords or {}

    def classify(self, doc_name: str, text_sample: str) -> DomainClassification:
        lower = f"{doc_name} {text_sample[:5000]}".lower()
        configured: list[tuple[Literal["financial", "legal", "technical", "medical"], float]] = [
            ("financial", 0.85),
            ("legal", 0.85),
            ("technical", 0.82),
            ("medical", 0.82),
        ]
        for domain, confidence in configured:
            terms = [str(k).lower() for k in self.domain_keywords.get(domain, [])]
            if terms and any(term in lower for term in terms):
                return DomainClassification(domain_hint=domain, confidence=confidence)
        return DomainClassification(domain_hint="general", confidence=0.55)


def classify_profile(
    doc_name: str,
    sha256: str,
    metrics: TriageMetrics,
    language_hint: LanguageHint | None = None,
    settings: Settings | None = None,
    domain_classifier: DomainClassifier | None = None,
) -> DocumentProfile:
    settings = settings or Settings()
    rules = load_runtime_rules(settings)
    domain_keywords = rules.get("triage", {}).get("domain_keywords", {})
    domain_classifier = domain_classifier or KeywordDomainClassifier(domain_keywords=domain_keywords if isinstance(domain_keywords, dict) else None)

    if metrics.form_field_count >= settings.triage_form_fillable_min_fields:
        origin_type = "form_fillable"
        origin_confidence = 0.92
    elif metrics.zero_text_document:
        origin_type = "scanned_image"
        origin_confidence = 0.88
    elif (
        metrics.image_area_ratio >= settings.triage_scanned_image_min_image_ratio
        and metrics.avg_char_density <= settings.triage_scanned_image_max_char_density
    ):
        origin_type = "scanned_image"
        origin_confidence = 0.82
    elif (
        (metrics.image_area_ratio >= settings.triage_mixed_min_image_ratio and metrics.avg_char_density <= settings.triage_mixed_max_char_density)
        or metrics.mixed_mode_pages_fraction >= settings.triage_mixed_mode_pages_fraction_threshold
    ):
        origin_type = "mixed"
        origin_confidence = 0.74
    else:
        origin_type = "native_digital"
        origin_confidence = 0.72

    table_heavy = metrics.table_signal_ratio >= settings.triage_table_heavy_ratio
    figure_heavy = metrics.image_area_ratio >= settings.triage_figure_heavy_image_ratio
    multi_column = (
        metrics.estimated_columns >= settings.triage_multi_column_min_columns
        or metrics.multi_column_fraction >= settings.triage_multi_column_fraction
    )

    signal_count = sum(1 for flag in (table_heavy, figure_heavy, multi_column) if flag)

    if metrics.zero_text_document and metrics.image_area_ratio > 0:
        layout_complexity = "figure_heavy"
        layout_confidence = 0.75
    elif signal_count >= settings.triage_layout_mixed_signal_count:
        layout_complexity = "mixed"
        layout_confidence = 0.78
    elif table_heavy:
        layout_complexity = "table_heavy"
        layout_confidence = min(0.95, 0.6 + metrics.table_signal_ratio)
    elif multi_column:
        layout_complexity = "multi_column"
        layout_confidence = min(0.92, 0.55 + metrics.multi_column_fraction)
    elif figure_heavy:
        layout_complexity = "figure_heavy"
        layout_confidence = min(0.92, 0.55 + metrics.image_area_ratio)
    else:
        layout_complexity = "single_column"
        layout_confidence = 0.70

    if metrics.zero_text_document:
        estimated_extraction_cost = "needs_vision_model"
        extraction_cost_confidence = 0.90
    elif (
        metrics.avg_char_density <= settings.triage_vision_max_char_density
        and metrics.image_area_ratio >= settings.triage_vision_min_image_ratio
    ):
        estimated_extraction_cost = "needs_vision_model"
        extraction_cost_confidence = 0.84
    elif (
        layout_complexity in {"multi_column", "table_heavy", "figure_heavy", "mixed"}
        or metrics.low_char_pages_fraction >= settings.triage_layout_low_char_pages_fraction
        or metrics.form_field_count >= settings.triage_form_fillable_min_fields
        or metrics.mixed_mode_pages_fraction >= settings.triage_mixed_mode_pages_fraction_threshold
    ):
        estimated_extraction_cost = "needs_layout_model"
        extraction_cost_confidence = 0.76
    else:
        estimated_extraction_cost = "fast_text_sufficient"
        extraction_cost_confidence = 0.72

    if metrics.zero_text_document:
        domain_hint = "general"
        domain_confidence = 0.20
    else:
        domain = domain_classifier.classify(doc_name=doc_name, text_sample=metrics.text_sample)
        domain_hint = domain.domain_hint
        domain_confidence = max(0.0, min(1.0, float(domain.confidence)))

    language_hint = language_hint or LanguageHint(language="unknown" if metrics.zero_text_document else "en", confidence=0.0 if metrics.zero_text_document else 0.6)
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
        origin_confidence=origin_confidence,
        layout_confidence=layout_confidence,
        domain_confidence=domain_confidence,
        extraction_cost_confidence=extraction_cost_confidence,
        page_count=metrics.page_count,
        avg_char_density=metrics.avg_char_density,
        image_area_ratio=metrics.image_area_ratio,
        whitespace_ratio=metrics.whitespace_ratio,
    )


class TriageAgent:
    def __init__(self, store: ArtifactStore, settings: Settings | None = None, domain_classifier: DomainClassifier | None = None):
        self.store = store
        self.settings = settings or Settings(workspace_root=store.settings.workspace_root)
        rules = load_runtime_rules(self.settings)
        domain_keywords = rules.get("triage", {}).get("domain_keywords", {})
        self.domain_classifier = domain_classifier or KeywordDomainClassifier(domain_keywords=domain_keywords if isinstance(domain_keywords, dict) else None)

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
        mixed_mode_pages = 0

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
                if chars < self.settings.triage_low_char_page_threshold:
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
                page_image_area = 0.0
                for img in mu_page.get_images(full=True):
                    xref = img[0]
                    rects = mu_page.get_image_rects(xref)
                    img_area = sum(r.width * r.height for r in rects)
                    page_image_area += img_area
                    total_image_area += img_area

                page_image_ratio = page_image_area / max(page_area, 1.0)
                if chars >= self.settings.triage_low_char_page_threshold and page_image_ratio >= self.settings.triage_mixed_min_image_ratio:
                    mixed_mode_pages += 1

        avg_char_density = total_chars / max(total_page_area, 1.0)
        # Floating point accumulation can push the ratio a hair above 1.0 (e.g., 1.00000001)
        # Clamp to [0, 1] to satisfy schema constraints and reflect a true ratio.
        image_area_ratio = max(0.0, min(1.0, total_image_area / max(total_page_area, 1.0)))
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
            mixed_mode_pages_fraction=mixed_mode_pages / max(page_count, 1),
            zero_text_document=total_chars == 0,
        )

    def run(self, pdf_path: Path) -> DocumentProfile:
        started = time.perf_counter()
        logger.info("stage=triage start doc=%s", pdf_path.name)
        sha = sha256_file(pdf_path)
        metrics = self._compute_metrics(pdf_path)
        if metrics.zero_text_document:
            language_hint = LanguageHint(language="unknown", confidence=0.0)
        else:
            lang = detect_language(metrics.text_sample, mode=self.settings.language_detection_mode)
            language_hint = LanguageHint(language=lang["language"], confidence=float(lang["confidence"]))
        profile = classify_profile(
            pdf_path.name,
            sha,
            metrics,
            language_hint=language_hint,
            settings=self.settings,
            domain_classifier=self.domain_classifier,
        )
        out = self.store.profiles_dir / f"{profile.doc_id}.json"
        self.store.save_json(out, profile.model_dump(mode="json"))
        elapsed = int((time.perf_counter() - started) * 1000)
        logger.info("stage=triage end doc=%s pages=%s duration_ms=%s", pdf_path.name, metrics.page_count, elapsed)
        return profile
