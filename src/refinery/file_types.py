from __future__ import annotations

from pathlib import Path

from .lang_detect import detect_language
from .file_router import FileRouter
from .models import DocumentProfile, LanguageHint
from .runtime_rules import load_runtime_rules
from .utils import deterministic_id, sha256_file


def doc_id_from_file(path: Path) -> str:
    return deterministic_id("doc", {"name": path.name, "sha256": sha256_file(path)})


def build_non_pdf_profile(path: Path, router: FileRouter) -> DocumentProfile:
    ftype = router.detect_type(path)
    sha = sha256_file(path)
    doc_id = deterministic_id("doc", {"name": path.name, "sha256": sha})
    rules = load_runtime_rules(router.settings)
    configured_profiles = rules.get("file_type_profiles", {}) if isinstance(rules.get("file_type_profiles", {}), dict) else {}
    profile_cfg = configured_profiles.get(ftype) or configured_profiles.get("default") or {}

    origin = str(profile_cfg.get("origin_type", "mixed"))
    layout = str(profile_cfg.get("layout_complexity", "mixed"))
    cost = str(profile_cfg.get("estimated_extraction_cost", "needs_layout_model"))
    origin_confidence = float(profile_cfg.get("origin_confidence", 0.72))
    layout_confidence = float(profile_cfg.get("layout_confidence", 0.70))
    extraction_cost_confidence = float(profile_cfg.get("extraction_cost_confidence", 0.70))
    domain_hint = str(profile_cfg.get("domain_hint", "general"))
    domain_confidence = float(profile_cfg.get("domain_confidence", 0.55))

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
        domain_hint=domain_hint,
        estimated_extraction_cost=cost,
        origin_confidence=origin_confidence,
        layout_confidence=layout_confidence,
        domain_confidence=domain_confidence,
        extraction_cost_confidence=extraction_cost_confidence,
        page_count=1,
        avg_char_density=0.0,
        image_area_ratio=0.0,
        whitespace_ratio=0.0,
    )
