from __future__ import annotations

from pathlib import Path

from .lang_detect import detect_language
from .file_router import FileRouter
from .models import DocumentProfile, LanguageHint
from .utils import deterministic_id, sha256_file


def doc_id_from_file(path: Path) -> str:
    return deterministic_id("doc", {"name": path.name, "sha256": sha256_file(path)})


def build_non_pdf_profile(path: Path, router: FileRouter) -> DocumentProfile:
    ftype = router.detect_type(path)
    sha = sha256_file(path)
    doc_id = deterministic_id("doc", {"name": path.name, "sha256": sha})

    if ftype == "image":
        origin = "scanned_image"
        layout = "figure_heavy"
        cost = "needs_vision_model"
    elif ftype == "excel":
        origin = "native_digital"
        layout = "table_heavy"
        cost = "needs_layout_model"
    elif ftype in {"docx", "markdown"}:
        origin = "native_digital"
        layout = "mixed"
        cost = "fast_text_sufficient"
    else:
        origin = "mixed"
        layout = "mixed"
        cost = "needs_layout_model"

    sample_text = ""
    if ftype == "markdown":
        sample_text = path.read_text(encoding="utf-8", errors="ignore")[:8000]
    elif ftype == "docx":
        try:
            from docx import Document

            sample_text = "\n".join((p.text or "") for p in Document(str(path)).paragraphs)[:8000]
        except Exception:
            sample_text = ""
    lang_result = detect_language(sample_text, mode=router.settings.language_detection_mode)

    return DocumentProfile(
        doc_id=doc_id,
        doc_name=path.name,
        sha256=sha,
        origin_type=origin,
        layout_complexity=layout,
        language_hint=LanguageHint(language=lang_result["language"], confidence=float(lang_result["confidence"])),
        domain_hint="general",
        estimated_extraction_cost=cost,
        page_count=1,
        avg_char_density=0.0,
        image_area_ratio=0.0,
        whitespace_ratio=0.0,
    )
